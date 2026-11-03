#ifndef LBM_TESTS_REGRESSION_POISEUILLE_POISEUILLESHORT_H
#define LBM_TESTS_REGRESSION_POISEUILLE_POISEUILLESHORT_H

#include "common/ShortRunSupport.H"
#include "common/MacroSampling.H"

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::regression::poiseuille {

enum class Variant { HbbVelocity, GuoVelocity, GuoPressure, GuoPrescribed };

struct Controls {
    Variant variant = Variant::HbbVelocity;
    std::array<int, 2> intervals{40, 10};
    int depthCells = 1;
    int maxGridSize = 16;
    amrex::Real rhoLb = 1.0;
    amrex::Real uMaxLb = 0.1;
    amrex::Real uRefPhys = 1.0;
    amrex::Real tau = 0.75;
    amrex::Real rhoInLb = 1.04;
    amrex::Real rhoOutLb = 0.96;
    std::string pressureScaling = "fixed_re_fixed_tau";
    int maxStep = 2;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 1;
    std::string runKind = "short";
    std::string runVariant;
    ShortOutputConfig output;
};

inline Variant parse_variant(std::string const& value)
{
    if (value == "hbb_velocity") return Variant::HbbVelocity;
    if (value == "guo_velocity") return Variant::GuoVelocity;
    if (value == "guo_pressure") return Variant::GuoPressure;
    if (value == "guo_prescribed_state") return Variant::GuoPrescribed;
    amrex::Abort(
        "poiseuille_flow.variant must be hbb_velocity, guo_velocity, "
        "guo_pressure or guo_prescribed_state");
    return Variant::HbbVelocity;
}

inline bool uses_guo(Variant variant) noexcept
{
    return variant != Variant::HbbVelocity;
}

inline bool density_driven(Variant variant) noexcept
{
    return variant == Variant::GuoPressure ||
        variant == Variant::GuoPrescribed;
}

inline Controls read_direct_controls()
{
    Controls controls;
    amrex::ParmParse flow("poiseuille_flow");
    std::string variant = "hbb_velocity";
    flow.query("variant", variant);
    controls.variant = parse_variant(variant);
    flow.query("intervals", controls.intervals);
    flow.query("depth_cells", controls.depthCells);
    flow.query("rho_lb", controls.rhoLb);
    flow.query("u_max_lb", controls.uMaxLb);
    flow.query("u_ref_phys", controls.uRefPhys);
    flow.query("tau", controls.tau);
    flow.query("rho_in_lb", controls.rhoInLb);
    flow.query("rho_out_lb", controls.rhoOutLb);
    flow.query("pressure_scaling", controls.pressureScaling);
    amrex::ParmParse("mesh").query("max_grid_size", controls.maxGridSize);
    amrex::ParmParse run("run");
    run.query("max_step", controls.maxStep);
    run.query("final_time_phys", controls.finalTimePhys);
    run.query("kind", controls.runKind);
    run.query("variant", controls.runVariant);
    amrex::ParmParse("diagnostics").query(
        "interval", controls.diagnosticsInterval);
    if (controls.intervals[0] <= 0 || controls.intervals[1] <= 0 ||
        controls.depthCells <= 0 || controls.maxGridSize <= 0) {
        amrex::Abort(
            "Poiseuille intervals, depth_cells and max_grid_size must be positive");
    }
    if (controls.pressureScaling != "fixed_re_fixed_tau" &&
        controls.pressureScaling != "fixed_mach_characterization") {
        amrex::Abort(
            "Poiseuille pressure_scaling must be fixed_re_fixed_tau or "
            "fixed_mach_characterization");
    }
    controls.output = {
        direct_output_directory(),
        direct_log_directory(),
        "Failed to prepare direct Poiseuille output",
        "case_name = poiseuille_flow\ncase_variant = " +
            controls.runVariant + "\n"};
    return controls;
}

struct ExactReference {
    initialization::PoiseuilleInitializationParams params;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        int,
        int,
        int) const noexcept
    {
        const amrex::Real eta = position[1];
        const amrex::Real ux = 4.0 * params.uMaxLb * eta * (1.0 - eta);
        amrex::Real rho = params.rhoLb;
        if (params.linearDensity != 0) {
            const amrex::Real xi = position[0] / 4.0;
            rho = params.rhoXLoLb +
                xi * (params.rhoXHiLb - params.rhoXLoLb);
        }
        return {rho, ux, 0.0, 0.0};
    }
};

struct FullFieldObservables {
    amrex::Real maxProfileError = 0.0;
    amrex::Real maxStreamwiseUxSpread = 0.0;
    amrex::Real maxPressureBoundaryRhoError = 0.0;
};

inline FullFieldObservables reduce_full_field_observables(
    Controls const& controls,
    MacroField const& macro,
    amrex::Box const& domain)
{
    const bool guo = uses_guo(controls.variant);
    const int ny = controls.intervals[1] + (guo ? 1 : 0);
    std::vector<amrex::Real> uxMinimum(
        ny, std::numeric_limits<amrex::Real>::max());
    std::vector<amrex::Real> uxMaximum(
        ny, std::numeric_limits<amrex::Real>::lowest());
    FullFieldObservables result;
    auto const field = lbm::tests::host_field_snapshot(macro.multifab());
    for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
        amrex::Box const& box = mfi.validbox();
        auto const values = field.const_array(mfi);
        for (amrex::BoxIterator index(box); index.ok(); ++index) {
            auto const& cell = index();
            const int j = cell[1] - domain.smallEnd(1);
            const amrex::Real eta = guo
                ? static_cast<amrex::Real>(j) / controls.intervals[1]
                : (static_cast<amrex::Real>(j) + 0.5) /
                    controls.intervals[1];
            const amrex::Real exact =
                4.0 * controls.uMaxLb * eta * (1.0 - eta);
            const amrex::Real ux = values(cell, MacroField::Ux);
            const bool strictInterior =
                !guo || (j != 0 && j != ny - 1);
            if (strictInterior) {
                result.maxProfileError = std::max(
                    result.maxProfileError, std::abs(ux - exact));
            }
            uxMinimum[j] = std::min(uxMinimum[j], ux);
            uxMaximum[j] = std::max(uxMaximum[j], ux);
            if (density_driven(controls.variant) && strictInterior &&
                (cell[0] == domain.smallEnd(0) ||
                 cell[0] == domain.bigEnd(0))) {
                const amrex::Real target =
                    cell[0] == domain.smallEnd(0)
                    ? controls.rhoInLb : controls.rhoOutLb;
                result.maxPressureBoundaryRhoError = std::max(
                    result.maxPressureBoundaryRhoError,
                    std::abs(values(cell, MacroField::Rho) - target));
            }
        }
    }
    amrex::ParallelDescriptor::ReduceRealMax(result.maxProfileError);
    amrex::ParallelDescriptor::ReduceRealMax(
        result.maxPressureBoundaryRhoError);
    for (int j = 0; j < ny; ++j) {
        amrex::ParallelDescriptor::ReduceRealMin(uxMinimum[j]);
        amrex::ParallelDescriptor::ReduceRealMax(uxMaximum[j]);
        result.maxStreamwiseUxSpread = std::max(
            result.maxStreamwiseUxSpread, uxMaximum[j] - uxMinimum[j]);
    }
    return result;
}

inline void write_artifacts(
    Controls const& controls,
    config::ResolvedLbmConfig const& lbm,
    runtime::DiagnosticRow const& row,
    MacroField const& macro,
    amrex::Box const& domain)
{
    const bool guo = uses_guo(controls.variant);
    const bool pressure = controls.variant == Variant::GuoPressure;
    const bool prescribed = controls.variant == Variant::GuoPrescribed;
    const int nx = controls.intervals[0] + (guo ? 1 : 0);
    const int ny = controls.intervals[1] + (guo ? 1 : 0);
    const amrex::Real dx = 1.0 / controls.intervals[1];
    const amrex::Real rawReference = controls.rhoLb *
        static_cast<amrex::Real>(nx * ny * controls.depthCells);
    const amrex::Real physicalReference = controls.rhoLb *
        static_cast<amrex::Real>(
            controls.intervals[0] * controls.intervals[1] *
            controls.depthCells);
    const amrex::Real densityDelta = density_driven(controls.variant)
        ? controls.rhoInLb - controls.rhoOutLb
        : 0.0;
    constexpr amrex::Real cs2 = 1.0 / 3.0;

    const int sampleI = domain.smallEnd(0) + nx / 2;
    std::vector<amrex::Real> observedUx(ny);
    std::vector<amrex::Real> observedUy(ny);
    std::vector<amrex::Real> observedUz(ny);
    for (int j = 0; j < ny; ++j) {
        const int sampleJ = domain.smallEnd(1) + j;
        observedUx[j] = sample_z_average(
            macro, sampleI, sampleJ, MacroField::Ux, domain);
        observedUy[j] = sample_z_max_abs(
            macro, sampleI, sampleJ, MacroField::Uy, domain);
        observedUz[j] = sample_z_max_abs(
            macro, sampleI, sampleJ, MacroField::Uz, domain);
    }
    std::vector<amrex::Real> observedRho(nx);
    for (int i = 0; i < nx; ++i) {
        observedRho[i] = sample_yz_trapezoidal_average(
            macro,
            domain.smallEnd(0) + i,
            MacroField::Rho,
            domain,
            guo);
    }
    const ComponentStatistics rho =
        component_statistics(macro, MacroField::Rho, controls.rhoLb);
    const ComponentStatistics uy =
        component_statistics(macro, MacroField::Uy);
    const ComponentStatistics uz =
        component_statistics(macro, MacroField::Uz);
    const FullFieldObservables full =
        reduce_full_field_observables(controls, macro, domain);
    amrex::Real interiorErrorSquared = 0.0;
    amrex::Real interiorExactSquared = 0.0;
    amrex::Real fullErrorSquared = 0.0;
    amrex::Real fullExactSquared = 0.0;
    amrex::Real eInfProfile = 0.0;
    amrex::Real eInfProfileFull = 0.0;
    for (int j = 0; j < ny; ++j) {
        const amrex::Real eta = guo
            ? static_cast<amrex::Real>(j) / controls.intervals[1]
            : (static_cast<amrex::Real>(j) + 0.5) /
                controls.intervals[1];
        const amrex::Real exact =
            4.0 * controls.uMaxLb * eta * (1.0 - eta);
        const amrex::Real error = observedUx[j] - exact;
        fullErrorSquared += error * error;
        fullExactSquared += exact * exact;
        eInfProfileFull = std::max(eInfProfileFull, std::abs(error));
        if (!guo || (j != 0 && j != ny - 1)) {
            interiorErrorSquared += error * error;
            interiorExactSquared += exact * exact;
            eInfProfile = std::max(eInfProfile, std::abs(error));
        }
    }
    const amrex::Real e2Profile = std::sqrt(
        interiorErrorSquared /
        (interiorExactSquared > 0.0 ? interiorExactSquared : 1.0));
    const amrex::Real e2ProfileFull = std::sqrt(
        fullErrorSquared /
        (fullExactSquared > 0.0 ? fullExactSquared : 1.0));
    eInfProfile /= controls.uMaxLb;
    eInfProfileFull /= controls.uMaxLb;
    const amrex::Real physicalWeightedMass = trapezoidal_weighted_sum_xy(
        macro, MacroField::Rho, domain, guo, guo);
    const amrex::Real physicalWeightedRelativeMassError = std::abs(
        physicalWeightedMass - physicalReference) / physicalReference;
    const amrex::Real rhoMean = rho.sum / rho.count;
    const amrex::Real rhoMeanError = rhoMean - controls.rhoLb;
    const amrex::Real rhoVariance = std::max(
        rho.squaredDeviationSum / rho.count -
            rhoMeanError * rhoMeanError,
        0.0);
    amrex::Real xiMean = 0.0;
    amrex::Real rhoMeanStreamwise = 0.0;
    for (int i = 0; i < nx; ++i) {
        const amrex::Real xi = guo
            ? static_cast<amrex::Real>(i) / controls.intervals[0]
            : (static_cast<amrex::Real>(i) + 0.5) /
                controls.intervals[0];
        xiMean += xi;
        rhoMeanStreamwise += observedRho[i];
    }
    xiMean /= nx;
    rhoMeanStreamwise /= nx;
    amrex::Real xiVariance = 0.0;
    amrex::Real rhoStreamwiseVariance = 0.0;
    amrex::Real covariance = 0.0;
    for (int i = 0; i < nx; ++i) {
        const amrex::Real xi = guo
            ? static_cast<amrex::Real>(i) / controls.intervals[0]
            : (static_cast<amrex::Real>(i) + 0.5) /
                controls.intervals[0];
        const amrex::Real dxValue = xi - xiMean;
        const amrex::Real drho = observedRho[i] - rhoMeanStreamwise;
        xiVariance += dxValue * dxValue;
        rhoStreamwiseVariance += drho * drho;
        covariance += dxValue * drho;
    }
    const amrex::Real rhoSlope = covariance / xiVariance;
    const amrex::Real rhoR2 = rhoStreamwiseVariance > 0.0
        ? covariance * covariance /
            (xiVariance * rhoStreamwiseVariance)
        : 1.0;
    const amrex::Real rhoDeltaMeasured =
        observedRho.front() - observedRho.back();
    const amrex::Real rhoDeltaRelativeError = densityDelta > 0.0
        ? std::abs(rhoDeltaMeasured - densityDelta) / densityDelta
        : 0.0;
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream profile(
        controls.output.directory + "/poiseuille_profile.csv",
        std::ios::out);
    profile << std::setprecision(17)
            << "j,y_phys,eta,ux_exact,ux_center_zmean,uy_abs_max,uz_abs_max,"
               "sample_i,is_physical_boundary_node\n";
    for (int j = 0; j < ny; ++j) {
        const amrex::Real eta = guo
            ? static_cast<amrex::Real>(j) / controls.intervals[1]
            : (static_cast<amrex::Real>(j) + 0.5) / controls.intervals[1];
        const amrex::Real ux = 4.0 * controls.uMaxLb * eta * (1.0 - eta);
        profile << j << ',' << eta << ',' << eta << ',' << ux << ','
                << observedUx[j] << ',' << observedUy[j] << ','
                << observedUz[j]
                << ',' << controls.intervals[0] / 2 << ','
                << (guo && (j == 0 || j == ny - 1) ? 1 : 0) << '\n';
    }

    std::ofstream density(
        controls.output.directory + "/poiseuille_density_pressure.csv",
        std::ios::out);
    density << std::setprecision(17)
            << "i,x_phys,x_over_L,rho_cross_section_weighted_mean,"
               "pressure_lb_cross_section_weighted_mean\n";
    for (int i = 0; i < nx; ++i) {
        const amrex::Real xi = guo
            ? static_cast<amrex::Real>(i) / controls.intervals[0]
            : (static_cast<amrex::Real>(i) + 0.5) / controls.intervals[0];
        density << i << ',' << 4.0 * xi << ',' << xi << ','
                << observedRho[i] << ','
                << cs2 * observedRho[i] << '\n';
    }

    std::ofstream metrics(
        controls.output.directory + "/poiseuille_profile_metrics.csv",
        std::ios::out);
    const int scalingCode = density_driven(controls.variant) ? 2 : 0;
    metrics << std::setprecision(17)
            << "step,nx,ny,nz,n_intervals_y,on_lattice_boundary_nodes,"
               "u_max_lb,E_2_profile,E_inf_profile,E_2_profile_full,"
               "E_inf_profile_full,max_abs_uy_over_umax,max_abs_uz_over_umax,"
               "relative_mass_error,raw_lattice_mass,raw_lattice_reference_mass,"
               "physical_weighted_mass,physical_weighted_reference_mass,"
               "physical_weighted_relative_mass_error,rho_mean_minus_rho0,"
               "rho_std_over_rho0,rho_min,rho_max,rho_streamwise_xlo,"
               "rho_streamwise_xhi,pressure_streamwise_xlo,"
               "pressure_streamwise_xhi,rho_streamwise_slope,"
               "pressure_streamwise_slope,rho_streamwise_linear_r2,"
               "pressure_driven,prescribed_state,pressure_scaling_code,"
               "re_effective,rho_in_target_lb,rho_out_target_lb,"
               "density_delta_target_lb,density_delta_expected_lb,"
               "density_delta_over_rho_ref,mach,"
               "max_abs_pressure_boundary_rho_error,"
               "rho_streamwise_delta_measured,"
               "rho_streamwise_delta_relative_error,"
               "max_abs_profile_error_all_x_over_umax,"
               "max_streamwise_ux_spread_over_umax\n"
            << row.step << ',' << nx << ',' << ny << ','
            << controls.depthCells << ',' << controls.intervals[1] << ','
            << (guo ? 1 : 0) << ',' << controls.uMaxLb << ','
            << e2Profile << ',' << eInfProfile << ',' << e2ProfileFull << ','
            << eInfProfileFull << ','
            << std::max(std::abs(uy.minimum), std::abs(uy.maximum)) /
                   controls.uMaxLb
            << ',' << std::max(std::abs(uz.minimum), std::abs(uz.maximum)) /
                   controls.uMaxLb
            << ',' << row.relativeMassError << ',' << row.mass << ','
            << rawReference << ',' << physicalWeightedMass << ','
            << physicalReference << ',' << physicalWeightedRelativeMassError
            << ',' << rhoMean - controls.rhoLb << ','
            << std::sqrt(rhoVariance) / controls.rhoLb << ','
            << rho.minimum << ',' << rho.maximum << ','
            << observedRho.front() << ',' << observedRho.back() << ','
            << cs2 * observedRho.front() << ','
            << cs2 * observedRho.back() << ',' << rhoSlope << ','
            << cs2 * rhoSlope << ',' << rhoR2 << ',' << (pressure ? 1 : 0)
            << ',' << (prescribed ? 1 : 0) << ',' << scalingCode << ','
            << controls.uMaxLb * controls.intervals[1] / lbm.nuLb << ','
            << (density_driven(controls.variant) ? controls.rhoInLb
                                                 : controls.rhoLb)
            << ',' << (density_driven(controls.variant) ? controls.rhoOutLb
                                                        : controls.rhoLb)
            << ',' << densityDelta << ',' << densityDelta << ','
            << densityDelta / controls.rhoLb << ','
            << controls.uMaxLb / std::sqrt(cs2) << ','
            << full.maxPressureBoundaryRhoError << ','
            << rhoDeltaMeasured << ',' << rhoDeltaRelativeError << ','
            << full.maxProfileError / controls.uMaxLb << ','
            << full.maxStreamwiseUxSpread / controls.uMaxLb << '\n';
}

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;
    explicit RunSession(Controls controls) : controls_(std::move(controls)) {}
    RunSession(RunSession const&) = delete;
    RunSession& operator=(RunSession const&) = delete;
    RunSession(RunSession&&) noexcept = default;
    RunSession& operator=(RunSession&&) noexcept = default;

    runtime::SingleLevelRunSessionResult start(
        runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        lbm_ = context.lbm;
        domain_ = context.geometry.Domain();
        output_ = std::make_unique<ShortRunOutput>(controls_.output);
        std::string error = output_->prepare();
        if (!error.empty()) {
            return {runtime::SingleLevelRunStatus::OutputFailure, error};
        }
        diagnostics_ = std::make_unique<Diagnostics>(
            runtime::SingleLevelDiagnosticsConfig{
                controls_.maxStep, controls_.diagnosticsInterval, 0, 0, 0},
            lbm_, context.geometry, context.boxArray,
            context.distributionMapping,
            runtime::SingleLevelSteadyObservationPolicy{}, *output_);
        output_->print_identity();
        return {};
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = true;
        plan.observeFinal = true;
        plan.macroAtInitialized = true;
        plan.macroAtFinal = true;
        plan.completed.addInterval(controls_.diagnosticsInterval);
        plan.completed.includeMaxStep = true;
        plan.completedMacro = plan.completed;
        return plan;
    }

    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        diagnostics_->initialize(state.macro(), exact_reference());
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        if (state.hasMacro()) {
            diagnostics_->onStep(state.macro(), exact_reference(), state.step());
        }
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        diagnostics_->ensureFinalObservation(
            state.macro(), exact_reference(), state.step());
        write_artifacts(
            controls_, lbm_, output_->last_row(), state.macro(), domain_);
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT(!record.has_value());
        return {};
    }

private:
    using Diagnostics = runtime::SingleLevelDiagnostics<
        Descriptor, ShortRunOutput>;

    ExactReference exact_reference() const noexcept
    {
        initialization::PoiseuilleInitializationParams params;
        params.rhoLb = controls_.rhoLb;
        params.uMaxLb = controls_.uMaxLb;
        params.linearDensity = density_driven(controls_.variant) ? 1 : 0;
        params.rhoXLoLb = density_driven(controls_.variant)
            ? controls_.rhoInLb : controls_.rhoLb;
        params.rhoXHiLb = density_driven(controls_.variant)
            ? controls_.rhoOutLb : controls_.rhoLb;
        return {params, controls_.rhoLb};
    }

    Controls controls_;
    config::ResolvedLbmConfig lbm_;
    amrex::Box domain_;
    std::unique_ptr<ShortRunOutput> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = poiseuille::RunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        if constexpr (Descriptor::d == 3) {
            if (view.mesh.nCell[2] < 2) {
                result.diagnostics.push_back({
                    "invalid_poiseuille_depth",
                    "mesh.n_cell.z",
                    "D3Q27 Poiseuille regression requires at least two z cells"});
                return result;
            }
        }
        controls.maxStep = view.run.maxStep;
        result.session = RunSession<Descriptor>(std::move(controls));
        return result;
    }
};

inline boundary::DomainBoundarySpec x_face(
    Controls const& controls,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    const bool inlet = side == boundary::DomainSide::XLo;
    spec.name = inlet ? "poiseuille_xlo" : "poiseuille_xhi";
    spec.location = side;
    spec.scheme = uses_guo(controls.variant)
        ? boundary::BoundaryScheme::GuoExtrapolation
        : boundary::BoundaryScheme::HalfwayBounceBack;
    if (controls.variant == Variant::HbbVelocity ||
        controls.variant == Variant::GuoVelocity) {
        spec.physical = inlet
            ? boundary::BoundaryPhysical::VelocityInlet
            : boundary::BoundaryPhysical::VelocityOutlet;
        spec.profile = boundary::BoundaryProfileKind::Poiseuille;
        spec.profileAxis = 1;
        spec.hasProfileAxis = true;
        spec.uMaxLb = controls.uMaxLb;
        spec.hasUMaxLb = true;
    } else if (controls.variant == Variant::GuoPressure) {
        spec.physical = inlet
            ? boundary::BoundaryPhysical::PressureInlet
            : boundary::BoundaryPhysical::PressureOutlet;
        spec.profile = boundary::BoundaryProfileKind::Constant;
        spec.rhoLb = inlet ? controls.rhoInLb : controls.rhoOutLb;
        spec.hasRhoLb = true;
    } else {
        spec.physical = boundary::BoundaryPhysical::PrescribedState;
        spec.profile = boundary::BoundaryProfileKind::Poiseuille;
        spec.profileAxis = 1;
        spec.hasProfileAxis = true;
        spec.rhoLb = inlet ? controls.rhoInLb : controls.rhoOutLb;
        spec.hasRhoLb = true;
        spec.prescribedULb = {controls.uMaxLb, 0.0, 0.0};
        spec.hasPrescribedULb = true;
    }
    return spec;
}

inline boundary::DomainBoundarySpec wall(
    Controls const& controls,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    spec.name = side == boundary::DomainSide::YLo
        ? "poiseuille_ylo" : "poiseuille_yhi";
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = uses_guo(controls.variant)
        ? boundary::BoundaryScheme::GuoExtrapolation
        : boundary::BoundaryScheme::HalfwayBounceBack;
    return spec;
}

template <class Descriptor>
auto make_program(Controls controls)
{
    using Profile = initialization::PoiseuilleInitializationProfile;
    using Initialization =
        initialization::ProfileInitializationSelection<Profile>;
    using Program = runtime::SingleLevelProgram<
        Descriptor, BGKDynamicsSelection, Initialization, Preparation>;
    Program program;
    const bool guo = uses_guo(controls.variant);
    const amrex::Real dx = 1.0 / controls.intervals[1];
    program.mesh.nCell = {
        controls.intervals[0] + (guo ? 1 : 0),
        controls.intervals[1] + (guo ? 1 : 0),
        controls.depthCells};
    program.mesh.envelopeLo = {
        guo ? -0.5 * dx : 0.0,
        guo ? -0.5 * dx : 0.0,
        0.0};
    program.mesh.envelopeHi = {
        guo ? 4.0 + 0.5 * dx : 4.0,
        guo ? 1.0 + 0.5 * dx : 1.0,
        dx * controls.depthCells};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;
    program.lbm.parameterizationKind = config::LbmParameterizationKind::FromTau;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.uMaxLb;
    program.lbm.uRefPhys = controls.uRefPhys;
    program.lbm.lengthRefLb = controls.intervals[1];
    program.lbm.lengthRefPhys = 1.0;
    program.lbm.tau = controls.tau;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "poiseuille_flow";
    program.run.variant = controls.runVariant;
    initialization::PoiseuilleInitializationParams params;
    params.rhoLb = controls.rhoLb;
    params.uMaxLb = controls.uMaxLb;
    params.linearDensity = density_driven(controls.variant) ? 1 : 0;
    params.rhoXLoLb = density_driven(controls.variant)
        ? controls.rhoInLb : controls.rhoLb;
    params.rhoXHiLb = density_driven(controls.variant)
        ? controls.rhoOutLb : controls.rhoLb;
    program.initialization = initialization::from_profile(Profile{params});
    program.domainBoundaries = {
        x_face(controls, boundary::DomainSide::XLo),
        x_face(controls, boundary::DomainSide::XHi),
        wall(controls, boundary::DomainSide::YLo),
        wall(controls, boundary::DomainSide::YHi)};
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::tests::regression::poiseuille

#endif
