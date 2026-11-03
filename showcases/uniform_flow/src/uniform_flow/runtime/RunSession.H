#ifndef LBM_CASES_UNIFORMFLOW_RUNSESSION_H
#define LBM_CASES_UNIFORMFLOW_RUNSESSION_H

#include "uniform_flow/config/Controls.H"
#include "uniform_flow/observation/UniformExactReference.H"
#include "uniform_flow/observation/UniformFlowObserver.H"
#include "uniform_flow/runtime/RunOutput.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace lbm::cases::uniform_flow {

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;
    explicit RunSession(Controls controls)
        : controls_(std::move(controls)),
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
            return {runtime::SingleLevelRunStatus::OutputFailure, std::move(error)};
        }
        observer_ = std::make_unique<Observer>(
            controls_.observation,
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            context.postStreamingPlan,
            output_->runOutputDir(),
            *output_);
        error = observer_->prepare();
        if (!error.empty()) {
            return {runtime::SingleLevelRunStatus::OutputFailure, std::move(error)};
        }
        diagnostics_ = std::make_unique<Diagnostics>(
            diagnosticsConfig(),
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            runtime::SingleLevelSteadyObservationPolicy{},
            *output_);
        output_->printRunSummary(context, observer_->runSummary());
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
            "Direct uniform-flow initialized observation contract changed");
        diagnostics_->initialize(state.macro(), exactReference());
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && state.step() > 0 &&
                state.forceBatch().empty(),
            "Direct uniform-flow completed observation contract changed");
        bool stopRequested = false;
        if (state.hasMacro()) {
            stopRequested = diagnostics_->onStep(
                state.macro(), exactReference(), state.step());
        }
        return {{}, stopRequested};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && observer_ != nullptr &&
                state.hasMacro() && state.forceBatch().empty(),
            "Direct uniform-flow final observation contract changed");
        diagnostics_->ensureFinalObservation(
            state.macro(), exactReference(), state.step());
        ObservationResult const observed =
            observer_->writeFinal(state.macro(), state.step());
        if (!observed.ok()) {
            return {
                observed.outputFailure ?
                    runtime::SingleLevelRunStatus::OutputFailure :
                    runtime::SingleLevelRunStatus::NumericalFailure,
                observed.error};
        }
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !record.has_value(),
            "Direct uniform-flow profiling is not exposed by case controls");
        return {};
    }

private:
    using Diagnostics = runtime::SingleLevelDiagnostics<
        Descriptor, RunOutput<Descriptor>>;
    using Observer = UniformFlowObserver<Descriptor, RunOutput<Descriptor>>;

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
            0};
    }

    UniformExactReference exactReference() const noexcept
    {
        return make_exact_reference(initialization::MacroscopicState{
            controls_.rhoLb, controls_.velocityLb, 0.0, 0.0});
    }

    Controls controls_;
    runtime::SingleLevelObservationPlan observationPlan_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<RunOutput<Descriptor>> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
    std::unique_ptr<Observer> observer_;
};

} // namespace lbm::cases::uniform_flow

#endif
