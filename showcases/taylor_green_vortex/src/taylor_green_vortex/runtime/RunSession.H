#ifndef LBM_CASES_TAYLORGREENVORTEX_RUNSESSION_H
#define LBM_CASES_TAYLORGREENVORTEX_RUNSESSION_H

#include "taylor_green_vortex/config/Controls.H"
#include "taylor_green_vortex/observation/TaylorGreenExactReference.H"
#include "taylor_green_vortex/runtime/RunOutput.H"
#include "lbm/initialization/TaylorGreenVortexParams.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>

#include <memory>
#include <optional>
#include <utility>

namespace lbm::cases::taylor_green_vortex {

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;

    RunSession(
        Controls controls,
        initialization::TaylorGreenVortexParams params)
        : controls_(std::move(controls)),
          params_(params),
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
        diagnostics_ = std::make_unique<Diagnostics>(
            diagnosticsConfig(),
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            runtime::SingleLevelSteadyObservationPolicy{},
            *output_);
        output_->printRunSummary(context);
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
            "Direct Taylor-Green initialized observation contract changed");
        diagnostics_->initialize(state.macro(), exactReference(0));
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && state.step() > 0 && state.hasMacro() &&
                state.forceBatch().empty(),
            "Direct Taylor-Green completed observation contract changed");
        const bool stopRequested = diagnostics_->onStep(
            state.macro(), exactReference(state.step()), state.step());
        AMREX_ALWAYS_ASSERT(!stopRequested);
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && state.hasMacro() &&
                state.forceBatch().empty(),
            "Direct Taylor-Green final observation contract changed");
        diagnostics_->ensureFinalObservation(
            state.macro(), exactReference(state.step()), state.step());
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !record.has_value(),
            "Direct Taylor-Green profiling is not exposed by its case controls");
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
        return plan;
    }

    runtime::SingleLevelDiagnosticsConfig diagnosticsConfig() const noexcept
    {
        return {
            controls_.maxStep,
            controls_.diagnosticsInterval,
            controls_.plotInterval,
            controls_.writeFinalPlotfile,
            controls_.convergenceEnabled};
    }

    TaylorGreenExactReference exactReference(int step) const
    {
        return make_exact_reference(params_, step);
    }

    Controls controls_;
    initialization::TaylorGreenVortexParams params_;
    runtime::SingleLevelObservationPlan observationPlan_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<RunOutput<Descriptor>> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
};

} // namespace lbm::cases::taylor_green_vortex

#endif
