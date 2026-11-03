#ifndef LBM_TESTS_REGRESSION_UNIFORM_UNIFORMSHORT_H
#define LBM_TESTS_REGRESSION_UNIFORM_UNIFORMSHORT_H

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
#include <AMReX_Vector.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace lbm::tests::regression::uniform {

enum class Variant { HbbVelocityIo, GuoVelocityIo, GuoFreeOutflow };

struct Controls {
    Variant variant = Variant::HbbVelocityIo;
    amrex::Array<int, 2> intervals{10, 10};
    int depthCells = 1;
    int maxGridSize = 5;
    amrex::Real rhoLb = 1.0;
    amrex::Real velocityLb = 0.1;
    amrex::Real velocityReferencePhys = 1.0;
    amrex::Real nuLb = 1.0 / 12.0;
    int maxStep = 2;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 1;
    std::string runKind = "regression";
    std::string runVariant;
    ShortOutputConfig output;
};

inline Variant parse_variant(std::string const& value)
{
    if (value == "hbb_velocity_io") return Variant::HbbVelocityIo;
    if (value == "guo_velocity_io") return Variant::GuoVelocityIo;
    if (value == "guo_velocity_free_outflow") return Variant::GuoFreeOutflow;
    amrex::Abort("uniform_flow.variant must be hbb_velocity_io, guo_velocity_io or guo_velocity_free_outflow");
    return Variant::HbbVelocityIo;
}

inline bool uses_guo(Variant variant) noexcept
{
    return variant != Variant::HbbVelocityIo;
}

inline Controls read_direct_controls()
{
    Controls controls;
    amrex::ParmParse uniform("uniform_flow");
    std::string variant;
    uniform.get("variant", variant);
    controls.variant = parse_variant(variant);
    amrex::Vector<int> intervals;
    uniform.getarr("intervals", intervals, 0, 2);
    controls.intervals = {intervals[0], intervals[1]};
    uniform.get("depth_cells", controls.depthCells);
    uniform.get("rho_lb", controls.rhoLb);
    uniform.get("velocity_lb", controls.velocityLb);
    uniform.get("velocity_reference_phys", controls.velocityReferencePhys);
    uniform.get("nu_lb", controls.nuLb);
    amrex::ParmParse("mesh").get("max_grid_size", controls.maxGridSize);
    amrex::ParmParse run("run");
    run.get("max_step", controls.maxStep);
    run.get("final_time_phys", controls.finalTimePhys);
    run.get("kind", controls.runKind);
    run.get("variant", controls.runVariant);
    amrex::ParmParse("diagnostics").get(
        "interval", controls.diagnosticsInterval);
    controls.output = {
        direct_output_directory(),
        direct_log_directory(),
        "Failed to prepare direct uniform-flow output",
        "case_name = uniform_flow\ncase_variant = " +
            controls.runVariant + "\n"};
    return controls;
}

struct ExactReference {
    initialization::MacroscopicState state;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return state;
    }
};

inline void write_metrics(
    Controls const& controls,
    config::ResolvedLbmConfig const& lbm,
    runtime::DiagnosticRow const& row,
    MacroField const& macro)
{
    const amrex::Real rawReference = controls.rhoLb *
        static_cast<amrex::Real>(
            (controls.intervals[0] + (uses_guo(controls.variant) ? 1 : 0)) *
            controls.intervals[1] * controls.depthCells);
    const amrex::Real physicalReference = controls.rhoLb *
        static_cast<amrex::Real>(
            controls.intervals[0] * controls.intervals[1] *
            controls.depthCells);
    const amrex::Real flowThroughs =
        static_cast<amrex::Real>(row.step) * std::abs(lbm.uRefLb) /
        lbm.lengthRefLb;
    const ComponentStatistics rho =
        component_statistics(macro, MacroField::Rho, controls.rhoLb);
    const ComponentStatistics ux =
        component_statistics(macro, MacroField::Ux, controls.velocityLb);
    const ComponentStatistics uy =
        component_statistics(macro, MacroField::Uy);
    const ComponentStatistics uz =
        component_statistics(macro, MacroField::Uz);
    const amrex::Real meanUx = ux.sum / ux.count;
    const amrex::Real meanUxError = meanUx - controls.velocityLb;
    const amrex::Real uxVariance = std::max(
        ux.squaredDeviationSum / ux.count - meanUxError * meanUxError,
        0.0);
    const amrex::Real rhoMean = rho.sum / rho.count;
    const amrex::Real rhoMeanError = rhoMean - controls.rhoLb;
    const amrex::Real rhoVariance = std::max(
        rho.squaredDeviationSum / rho.count -
            rhoMeanError * rhoMeanError,
        0.0);
    const amrex::Real maxAbsUy =
        std::max(std::abs(uy.minimum), std::abs(uy.maximum));
    const amrex::Real maxAbsUz =
        std::max(std::abs(uz.minimum), std::abs(uz.maximum));
    const amrex::Box domain = macro.multifab().boxArray().minimalBox();
    const amrex::Real physicalMass = trapezoidal_weighted_sum_xy(
        macro, MacroField::Rho, domain, uses_guo(controls.variant), false);
    const amrex::Real physicalMassError =
        std::abs(physicalMass - physicalReference) / physicalReference;
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream stream(
        controls.output.directory + "/uniform_flow_metrics.csv",
        std::ios::out);
    stream << std::setprecision(17)
           << "step,flow_throughs,E_2_u,E_inf_u,mean_ux,std_ux_over_u0,"
              "max_abs_uy_over_u0,max_abs_uz_over_u0,relative_mass_error,"
              "rho_mean_minus_rho0,rho_std_over_rho0,raw_lattice_mass,"
              "raw_lattice_reference_mass,physical_weighted_mass,"
              "physical_weighted_reference_mass,"
              "physical_weighted_relative_mass_error\n"
           << row.step << ',' << flowThroughs << ',' << row.relativeL2U << ','
           << row.linfU / controls.velocityLb << ',' << meanUx << ','
           << std::sqrt(uxVariance) / controls.velocityLb << ','
           << maxAbsUy / controls.velocityLb << ','
           << maxAbsUz / controls.velocityLb << ','
           << row.relativeMassError << ','
           << rhoMean - controls.rhoLb << ','
           << std::sqrt(rhoVariance) / controls.rhoLb << ',' << row.mass << ','
           << rawReference << ',' << physicalMass << ','
           << physicalReference << ',' << physicalMassError << '\n';
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
        write_metrics(controls_, lbm_, output_->last_row(), state.macro());
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
        return {
            {controls_.rhoLb, controls_.velocityLb, 0.0, 0.0},
            controls_.rhoLb};
    }

    Controls controls_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<ShortRunOutput> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = uniform::RunSession<Descriptor>;

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
                    "invalid_uniform_flow_depth",
                    "mesh.n_cell.z",
                    "D3Q27 uniform-flow regression requires at least two z cells"});
                return result;
            }
        }
        controls.maxStep = view.run.maxStep;
        result.session = RunSession<Descriptor>(std::move(controls));
        return result;
    }
};

inline boundary::DomainBoundarySpec inlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "uniform_inlet";
    spec.location = boundary::DomainSide::XLo;
    spec.physical = boundary::BoundaryPhysical::VelocityInlet;
    spec.scheme = uses_guo(controls.variant)
        ? boundary::BoundaryScheme::GuoExtrapolation
        : boundary::BoundaryScheme::HalfwayBounceBack;
    spec.profile = boundary::BoundaryProfileKind::Uniform;
    spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
    spec.hasPrescribedULb = true;
    return spec;
}

inline boundary::DomainBoundarySpec outlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "uniform_outlet";
    spec.location = boundary::DomainSide::XHi;
    spec.physical = controls.variant == Variant::GuoFreeOutflow
        ? boundary::BoundaryPhysical::FreeOutflow
        : boundary::BoundaryPhysical::VelocityOutlet;
    spec.scheme = uses_guo(controls.variant)
        ? boundary::BoundaryScheme::GuoExtrapolation
        : boundary::BoundaryScheme::HalfwayBounceBack;
    if (spec.physical == boundary::BoundaryPhysical::VelocityOutlet) {
        spec.profile = boundary::BoundaryProfileKind::Uniform;
        spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
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
    const bool guo = uses_guo(controls.variant);
    const amrex::Real dx = 1.0 / controls.intervals[0];
    program.mesh.nCell = {
        controls.intervals[0] + (guo ? 1 : 0),
        controls.intervals[1],
        controls.depthCells};
    program.mesh.envelopeLo = {guo ? -0.5 * dx : 0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        guo ? 1.0 + 0.5 * dx : 1.0,
        1.0,
        dx * controls.depthCells};
    program.mesh.isPeriodic = {0, 1, 1};
    program.mesh.maxGridSize = controls.maxGridSize;
    program.lbm.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.velocityLb;
    program.lbm.uRefPhys = controls.velocityReferencePhys;
    program.lbm.lengthRefLb = controls.intervals[0];
    program.lbm.lengthRefPhys = 1.0;
    program.lbm.nuLb = controls.nuLb;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "uniform_flow";
    program.run.variant = controls.runVariant;
    program.initialization = initialization::from_profile(
        Profile{{controls.rhoLb, controls.velocityLb, 0.0, 0.0}});
    program.domainBoundaries = {inlet(controls), outlet(controls)};
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::tests::regression::uniform

#endif
