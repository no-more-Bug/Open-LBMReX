#ifndef LBM_CASES_LIDDRIVENCAVITY_RUNSESSION_H
#define LBM_CASES_LIDDRIVENCAVITY_RUNSESSION_H

#include "lid_driven_cavity/config/Controls.H"
#include "lid_driven_cavity/observation/CavityObserver.H"
#include "lid_driven_cavity/runtime/RunOutput.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

struct QuiescentMacroscopicReference {
    amrex::Real rho = 1.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

struct QuiescentExactReference {
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    QuiescentMacroscopicReference operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return {massReferenceDensity, 0.0, 0.0, 0.0};
    }
};

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;

    RunSession(Controls controls, CavityObservationConfig cavity)
        : controls_(std::move(controls)),
          cavityConfig_(std::move(cavity)),
          observationPlan_(resolveObservationPlan())
    {}

    RunSession(RunSession const&) = delete;
    RunSession& operator=(RunSession const&) = delete;
    RunSession(RunSession&&) noexcept = default;
    RunSession& operator=(RunSession&&) noexcept = default;

    runtime::SingleLevelRunSessionResult start(
        runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        lbm_ = context.lbm;
        output_ = std::make_unique<RunOutput<Descriptor>>(controls_, lbm_);
        std::string error = output_->prepare();
        if (!error.empty()) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                std::move(error)};
        }

        cavity_ = std::make_unique<CavityObserver>(
            cavityConfig_,
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping);
        error = cavity_->prepare(output_->runOutputDir());
        if (!error.empty()) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                std::move(error)};
        }

        diagnostics_ = std::make_unique<Diagnostics>(
            diagnosticsConfig(),
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            steadyPolicy(),
            *output_);
        output_->printRunSummary(
            context,
            cavity_->runSummary(
                controls_.steadyCheckInterval,
                controls_.steadyL2Tolerance,
                controls_.stopOnSteady));
        return {};
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        return observationPlan_;
    }

    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && state.step() == 0 && state.hasMacro() &&
                state.forceBatch().empty(),
            "Direct cavity initialized observation contract changed");
        diagnostics_->initialize(state.macro(), exactReference());
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && state.step() > 0 && state.hasMacro() &&
                state.forceBatch().empty(),
            "Direct cavity completed observation contract changed");
        const bool stopRequested = diagnostics_->onStep(
            state.macro(), exactReference(), state.step());
        return {{}, stopRequested};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && cavity_ != nullptr &&
                state.hasMacro() && state.forceBatch().empty(),
            "Direct cavity final observation contract changed");
        diagnostics_->ensureFinalObservation(
            state.macro(), exactReference(), state.step());
        std::string const summary = cavity_->writeFinal(
            state.macro(),
            state.step(),
            diagnostics_->lastSteadyResidual(),
            controls_.steadyL2Tolerance,
            diagnostics_->steadyReached());
        if (!summary.empty()) {
            output_->printAndLog(summary);
        }
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !record.has_value(),
            "Direct cavity profiling is not exposed by its case controls");
        return {};
    }

private:
    using Diagnostics = runtime::SingleLevelDiagnostics<
        Descriptor,
        RunOutput<Descriptor>>;

    runtime::SingleLevelObservationPlan resolveObservationPlan() const
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = true;
        plan.observeFinal = true;
        plan.macroAtInitialized = true;
        plan.macroAtFinal = true;
        plan.completed.addInterval(controls_.diagnosticsInterval);
        plan.completed.addInterval(controls_.plotInterval);
        plan.completed.includeMaxStep =
            controls_.diagnosticsInterval > 0 || controls_.plotInterval > 0;
        plan.completedMacro = plan.completed;
        if (controls_.steadyCheckInterval > 0) {
            plan.completed.everyStep = true;
            plan.completedMacro.everyStep = true;
        }
        return plan;
    }

    runtime::SingleLevelDiagnosticsConfig diagnosticsConfig() const noexcept
    {
        return {
            controls_.maxStep,
            controls_.diagnosticsInterval,
            controls_.plotInterval,
            controls_.writeFinalPlotfile,
            0};
    }

    runtime::SingleLevelSteadyObservationPolicy steadyPolicy() const noexcept
    {
        return {
            controls_.steadyCheckInterval > 0,
            "cavity_steady",
            controls_.steadyCheckInterval,
            controls_.stopOnSteady,
            controls_.steadyL2Tolerance};
    }

    QuiescentExactReference exactReference() const noexcept
    {
        return {lbm_.rhoRefLb};
    }

    Controls controls_;
    CavityObservationConfig cavityConfig_;
    runtime::SingleLevelObservationPlan observationPlan_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<RunOutput<Descriptor>> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
    std::unique_ptr<CavityObserver> cavity_;
};

} // namespace lbm::cases::lid_driven_cavity

#endif
