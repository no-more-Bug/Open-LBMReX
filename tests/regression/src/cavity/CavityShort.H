#ifndef LBM_TESTS_REGRESSION_CAVITY_CAVITYSHORT_H
#define LBM_TESTS_REGRESSION_CAVITY_CAVITYSHORT_H

#include "common/ShortRunSupport.H"
#include "common/MacroSampling.H"

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::regression::cavity {

enum class WallScheme { HalfwayBounceBack, GuoExtrapolation };

struct Controls {
    std::array<amrex::Real, 2> physicalLo{0.0, 0.0};
    std::array<amrex::Real, 2> physicalHi{1.0, 1.0};
    std::array<int, 2> intervals{16, 16};
    int depthCells = 1;
    WallScheme wallScheme = WallScheme::GuoExtrapolation;
    int maxGridSize = 8;
    amrex::Real rhoLb = 1.0;
    amrex::Real reynolds = 1000.0;
    amrex::Real lidVelocityLb = 0.1;
    amrex::Real uRefPhys = 1.0;
    int maxStep = 2;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 1;
    std::string runKind = "regression";
    std::string runVariant;
    ShortOutputConfig output;
};

inline bool uses_guo(Controls const& controls) noexcept
{
    return controls.wallScheme == WallScheme::GuoExtrapolation;
}

inline Controls read_direct_controls()
{
    Controls controls;
    amrex::ParmParse cavity("cavity");
    cavity.query("physical_lo", controls.physicalLo);
    cavity.query("physical_hi", controls.physicalHi);
    cavity.query("intervals", controls.intervals);
    cavity.query("depth_cells", controls.depthCells);
    std::string scheme = "guo_extrapolation";
    cavity.query("wall_scheme", scheme);
    controls.wallScheme = scheme == "halfway_bounce_back"
        ? WallScheme::HalfwayBounceBack
        : WallScheme::GuoExtrapolation;
    cavity.query("rho_lb", controls.rhoLb);
    cavity.query("re", controls.reynolds);
    cavity.query("lid_velocity_lb", controls.lidVelocityLb);
    cavity.query("u_ref_phys", controls.uRefPhys);
    amrex::ParmParse("mesh").query("max_grid_size", controls.maxGridSize);
    amrex::ParmParse run("run");
    run.query("max_step", controls.maxStep);
    run.query("final_time_phys", controls.finalTimePhys);
    run.query("kind", controls.runKind);
    run.query("variant", controls.runVariant);
    amrex::ParmParse("diagnostics").query(
        "interval", controls.diagnosticsInterval);
    controls.output = {
        direct_output_directory(),
        direct_log_directory(),
        "Failed to prepare direct cavity output",
        "case_name = lid_driven_cavity\ncase_variant = " +
            controls.runVariant + "\n"};
    return controls;
}

struct ExactReference {
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return {massReferenceDensity, 0.0, 0.0, 0.0};
    }
};

inline void write_artifacts(
    Controls const& controls,
    runtime::DiagnosticRow const& row,
    MacroField const& macro,
    amrex::Box const& domain)
{
    const bool guo = uses_guo(controls);
    const int nx = controls.intervals[0] + (guo ? 1 : 0);
    const int ny = controls.intervals[1] + (guo ? 1 : 0);
    const amrex::Real dx =
        (controls.physicalHi[0] - controls.physicalLo[0]) /
        controls.intervals[0];
    const amrex::Real envelopeLoX = controls.physicalLo[0] -
        (guo ? 0.5 * dx : 0.0);
    const amrex::Real envelopeHiX = controls.physicalHi[0] +
        (guo ? 0.5 * dx : 0.0);
    const amrex::Real envelopeLoY = controls.physicalLo[1] -
        (guo ? 0.5 * dx : 0.0);
    const amrex::Real envelopeHiY = controls.physicalHi[1] +
        (guo ? 0.5 * dx : 0.0);

    std::vector<amrex::Real> uxValues(ny);
    for (int j = 0; j < ny; ++j) {
        uxValues[j] = sample_z_average(
            macro,
            domain.smallEnd(0) + nx / 2,
            domain.smallEnd(1) + j,
            MacroField::Ux,
            domain);
    }
    std::vector<amrex::Real> uyValues(nx);
    for (int i = 0; i < nx; ++i) {
        uyValues[i] = sample_z_average(
            macro,
            domain.smallEnd(0) + i,
            domain.smallEnd(1) + ny / 2,
            MacroField::Uy,
            domain);
    }
    if (!amrex::ParallelDescriptor::IOProcessor()) return;

    std::ofstream ux(
        controls.output.directory + "/cavity_centerline_x0p5_ux.csv",
        std::ios::out);
    ux << std::setprecision(17) << "y_over_L,ux_lb,ux_over_Ulid\n";
    for (int j = 0; j < ny; ++j) {
        const amrex::Real eta = guo
            ? static_cast<amrex::Real>(j) / controls.intervals[1]
            : (static_cast<amrex::Real>(j) + 0.5) / controls.intervals[1];
        const amrex::Real value = uxValues[j];
        ux << eta << ',' << value << ','
           << value / controls.lidVelocityLb << '\n';
    }

    std::ofstream uy(
        controls.output.directory + "/cavity_centerline_y0p5_uy.csv",
        std::ios::out);
    uy << std::setprecision(17) << "x_over_L,uy_lb,uy_over_Ulid\n";
    for (int i = 0; i < nx; ++i) {
        const amrex::Real xi = guo
            ? static_cast<amrex::Real>(i) / controls.intervals[0]
            : (static_cast<amrex::Real>(i) + 0.5) / controls.intervals[0];
        const amrex::Real value = uyValues[i];
        uy << xi << ',' << value << ','
           << value / controls.lidVelocityLb << '\n';
    }

    std::ofstream metadata(
        controls.output.directory + "/cavity_centerline_metadata.txt",
        std::ios::out);
    metadata << std::setprecision(17)
             << "boundary_geometry="
             << (guo ? "on_lattice_valid_node" : "domain_face_halfway")
             << "\nsampling=" << (guo ? "direct_node" : "interpolated_cell_center")
             << "\ninterpolation=" << (guo ? "false" : "true")
             << "\nendpoint_injection=" << (guo ? "false" : "true")
             << "\nvalid_nodes_x=" << nx
             << "\nvalid_nodes_y=" << ny
             << "\nlattice_intervals_x=" << controls.intervals[0]
             << "\nlattice_intervals_y=" << controls.intervals[1]
             << "\nz_policy=average"
             << "\nphysical_lo_x=" << controls.physicalLo[0]
             << "\nphysical_hi_x=" << controls.physicalHi[0]
             << "\nphysical_lo_y=" << controls.physicalLo[1]
             << "\nphysical_hi_y=" << controls.physicalHi[1]
             << "\nenvelope_lo_x=" << envelopeLoX
             << "\nenvelope_hi_x=" << envelopeHiX
             << "\nenvelope_lo_y=" << envelopeLoY
             << "\nenvelope_hi_y=" << envelopeHiY << '\n';

    if (guo) {
        std::ofstream metrics(
            controls.output.directory + "/cavity_validation_metrics.csv",
            std::ios::out);
        metrics << std::setprecision(17)
                << "step,relative_mass_error,rho_min,rho_max,l2_uz,linf_uz\n"
                << row.step << ',' << row.relativeMassError << ','
                << controls.rhoLb - row.maxRhoError << ','
                << controls.rhoLb + row.maxRhoError << ','
                << row.l2Uz << ',' << row.linfUz << '\n';
    }
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
        diagnostics_->initialize(state.macro(), ExactReference{controls_.rhoLb});
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        if (state.hasMacro()) {
            diagnostics_->onStep(
                state.macro(), ExactReference{controls_.rhoLb}, state.step());
        }
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        diagnostics_->ensureFinalObservation(
            state.macro(), ExactReference{controls_.rhoLb}, state.step());
        write_artifacts(
            controls_, output_->last_row(), state.macro(), domain_);
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

    Controls controls_;
    config::ResolvedLbmConfig lbm_;
    amrex::Box domain_;
    std::unique_ptr<ShortRunOutput> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = cavity::RunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        controls.maxStep = view.run.maxStep;
        result.session = RunSession<Descriptor>(std::move(controls));
        return result;
    }
};

inline boundary::DomainBoundarySpec wall(
    Controls const& controls,
    boundary::DomainSide side,
    bool moving)
{
    boundary::DomainBoundarySpec spec;
    if (moving) {
        spec.name = "moving_lid";
    } else if (side == boundary::DomainSide::XLo) {
        spec.name = "left_wall";
    } else if (side == boundary::DomainSide::XHi) {
        spec.name = "right_wall";
    } else {
        spec.name = "bottom_wall";
    }
    spec.location = side;
    spec.physical = moving
        ? boundary::BoundaryPhysical::MovingWall
        : boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = uses_guo(controls)
        ? boundary::BoundaryScheme::GuoExtrapolation
        : boundary::BoundaryScheme::HalfwayBounceBack;
    if (moving) {
        spec.profile = boundary::BoundaryProfileKind::Uniform;
        spec.prescribedULb = {controls.lidVelocityLb, 0.0, 0.0};
        spec.hasPrescribedULb = true;
    }
    return spec;
}

template <class Descriptor>
auto make_program(Controls controls)
{
    using Profile = initialization::ConstantInitializationProfile;
    using Initialization =
        initialization::ProfileInitializationSelection<Profile>;
    using Program = runtime::SingleLevelProgram<
        Descriptor, BGKDynamicsSelection, Initialization, Preparation>;
    Program program;
    const bool guo = uses_guo(controls);
    const amrex::Real dx =
        (controls.physicalHi[0] - controls.physicalLo[0]) /
        controls.intervals[0];
    program.mesh.nCell = {
        controls.intervals[0] + (guo ? 1 : 0),
        controls.intervals[1] + (guo ? 1 : 0),
        controls.depthCells};
    program.mesh.envelopeLo = {
        controls.physicalLo[0] - (guo ? 0.5 * dx : 0.0),
        controls.physicalLo[1] - (guo ? 0.5 * dx : 0.0),
        0.0};
    program.mesh.envelopeHi = {
        controls.physicalHi[0] + (guo ? 0.5 * dx : 0.0),
        controls.physicalHi[1] + (guo ? 0.5 * dx : 0.0),
        dx * controls.depthCells};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;
    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.lidVelocityLb;
    program.lbm.uRefPhys = controls.uRefPhys;
    program.lbm.lengthRefLb = controls.intervals[0];
    program.lbm.lengthRefPhys =
        controls.physicalHi[0] - controls.physicalLo[0];
    program.lbm.reynolds = controls.reynolds;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "lid_driven_cavity";
    program.run.variant = controls.runVariant;
    program.initialization = initialization::from_profile(
        Profile{{controls.rhoLb, 0.0, 0.0, 0.0}});
    program.domainBoundaries = {
        wall(controls, boundary::DomainSide::XLo, false),
        wall(controls, boundary::DomainSide::XHi, false),
        wall(controls, boundary::DomainSide::YLo, false),
        wall(controls, boundary::DomainSide::YHi, true)};
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::tests::regression::cavity

#endif
