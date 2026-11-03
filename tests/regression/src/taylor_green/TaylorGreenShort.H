#ifndef LBM_TESTS_REGRESSION_TAYLORGREEN_TAYLORGREENSHORT_H
#define LBM_TESTS_REGRESSION_TAYLORGREEN_TAYLORGREENSHORT_H

#include "common/ShortRunSupport.H"

#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/TaylorGreenVortexParams.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_Math.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace lbm::tests::regression::taylor_green {

struct Controls {
    amrex::Array<int, 3> nCell{16, 16, 1};
    int maxGridSize = 8;
    amrex::Real rhoLb = 1.0;
    amrex::Real uRefLb = 0.01;
    amrex::Real uRefPhys = 1.0;
    amrex::Real nuLb = 0.01;
    amrex::Real velocityAmplitudeLb = 0.01;
    int maxStep = 2;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 1;
    std::string runKind = "regression";
    std::string runVariant;
    ShortOutputConfig output;
};

inline Controls read_direct_controls()
{
    Controls controls;
    amrex::ParmParse tgv("taylor_green");
    amrex::Vector<int> nCell;
    tgv.getarr("n_cell", nCell, 0, 3);
    controls.nCell = {nCell[0], nCell[1], nCell[2]};
    tgv.get("rho_lb", controls.rhoLb);
    tgv.get("u_ref_lb", controls.uRefLb);
    tgv.get("u_ref_phys", controls.uRefPhys);
    tgv.get("nu_lb", controls.nuLb);
    tgv.get("velocity_amplitude_lb", controls.velocityAmplitudeLb);
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
        "Failed to prepare direct Taylor-Green output",
        "case_name = taylor_green_vortex\ncase_variant = " +
            controls.runVariant + "\n"};
    return controls;
}

struct ExactReference {
    initialization::TaylorGreenVortexParams params;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        int,
        int,
        int) const noexcept
    {
        const amrex::Real x =
            2.0 * amrex::Math::pi<amrex::Real>() * position[0];
        const amrex::Real y =
            2.0 * amrex::Math::pi<amrex::Real>() * position[1];
        const auto sxCx = amrex::Math::sincos(x);
        const auto syCy = amrex::Math::sincos(y);
        const auto sx2Cx2 = amrex::Math::sincos(2.0 * x);
        const auto sy2Cy2 = amrex::Math::sincos(2.0 * y);
        return {
            params.rho0 -
                0.25 * params.rho0 * params.u0 * params.u0 / params.cs2 *
                    (sx2Cx2.second + sy2Cy2.second) * params.pressureDecay,
            -params.u0 * sxCx.second * syCy.first * params.decay,
            params.u0 * sxCx.first * syCy.second * params.decay,
            0.0};
    }
};

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;

    RunSession(
        Controls controls,
        initialization::TaylorGreenVortexParams initial)
        : controls_(std::move(controls)), initial_(initial)
    {}

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
                controls_.maxStep,
                controls_.diagnosticsInterval,
                0,
                0,
                0},
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            runtime::SingleLevelSteadyObservationPolicy{},
            *output_);
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
        diagnostics_->initialize(state.macro(), exact_reference(0));
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        if (state.hasMacro()) {
            diagnostics_->onStep(
                state.macro(), exact_reference(state.step()), state.step());
        }
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        diagnostics_->ensureFinalObservation(
            state.macro(), exact_reference(state.step()), state.step());
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

    ExactReference exact_reference(int step) const
    {
        auto params = initial_;
        params.decay = std::exp(
            -params.nu * (params.kx * params.kx + params.ky * params.ky) *
            static_cast<amrex::Real>(step));
        params.pressureDecay = params.decay * params.decay;
        return {params, initial_.rho0};
    }

    Controls controls_;
    initialization::TaylorGreenVortexParams initial_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<ShortRunOutput> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = taylor_green::RunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        controls.maxStep = view.run.maxStep;
        controls.finalTimePhys = view.run.finalTimePhys;
        controls.runVariant = view.run.variant;
        result.session = RunSession<Descriptor>(
            std::move(controls), view.initialization.resolved().taylorGreenParams);
        return result;
    }
};

template <class Descriptor>
auto make_program(Controls controls)
{
    using Program = runtime::SingleLevelProgram<
        Descriptor,
        BGKDynamicsSelection,
        initialization::ConfiguredInitializationSelection,
        Preparation>;
    Program program;
    program.mesh.nCell = controls.nCell;
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        static_cast<amrex::Real>(controls.nCell[2]) /
            static_cast<amrex::Real>(controls.nCell[0])};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = controls.maxGridSize;
    program.lbm.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.uRefLb;
    program.lbm.uRefPhys = controls.uRefPhys;
    program.lbm.nuLb = controls.nuLb;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "taylor_green_vortex";
    program.run.variant = controls.runVariant;
    program.initialization.intent.kind =
        initialization::InitializationKind::TaylorGreen;
    program.initialization.intent.rhoLb = controls.rhoLb;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb =
        controls.velocityAmplitudeLb;
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::tests::regression::taylor_green

#endif
