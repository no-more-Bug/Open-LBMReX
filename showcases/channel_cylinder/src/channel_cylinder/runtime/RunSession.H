#ifndef LBM_CASES_CHANNELCYLINDER_RUNSESSION_H
#define LBM_CASES_CHANNELCYLINDER_RUNSESSION_H

#include "channel_cylinder/config/Controls.H"
#include "channel_cylinder/config/ObservationConfig.H"
#include "channel_cylinder/observation/Observer.H"
#include "channel_cylinder/runtime/RunOutput.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace lbm::cases::channel_cylinder {

struct ConstantMacroscopicReference {
    amrex::Real rho = 1.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

struct ConstantExactReference {
    amrex::Real massReferenceDensity = 1.0;
    amrex::Real ux = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ConstantMacroscopicReference operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return {massReferenceDensity, ux, 0.0, 0.0};
    }
};

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;

    RunSession(
        Controls controls,
        ChannelCylinderRuntimeConfig observation)
        : controls_(std::move(controls)),
          observation_(std::move(observation)),
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

        observer_ = std::make_unique<ChannelCylinderObserver<Descriptor>>(
            observation_, lbm_, output_->runOutputDir());
        if (!observer_->outputError().empty()) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                observer_->outputError()};
        }

        diagnostics_ = std::make_unique<Diagnostics>(
            diagnosticsConfig(),
            lbm_,
            context.geometry,
            context.boxArray,
            context.distributionMapping,
            steadyPolicy(),
            *output_);
        output_->printRunSummary(context, observation_);
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
            diagnostics_ != nullptr && observer_ != nullptr &&
                state.step() == 0 && state.hasMacro() &&
                state.forceBatch().empty(),
            "Direct channel-cylinder initialized observation contract changed");
        diagnostics_->initialize(state.macro(), exactReference());
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            diagnostics_ != nullptr && observer_ != nullptr &&
                state.step() > 0,
            "Direct channel-cylinder completed observation contract changed");
        observer_->consume(state.forceBatch());
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
                state.hasMacro(),
            "Direct channel-cylinder final observation contract changed");
        observer_->consume(state.forceBatch());
        diagnostics_->ensureFinalObservation(
            state.macro(), exactReference(), state.step());

        BenchmarkObservationResult result = observer_->finalize(
            state.populations(), diagnostics_->steadyReached());
        if (!result.summary.empty()) {
            output_->printAndLog(result.summary);
        }
        if (!result.ok()) {
            return {
                result.outputFailure ?
                    runtime::SingleLevelRunStatus::OutputFailure :
                    runtime::SingleLevelRunStatus::NumericalFailure,
                std::move(result.error)};
        }
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !record.has_value(),
            "Direct channel-cylinder profiling is not exposed by case controls");
        return {};
    }

private:
    using Diagnostics = runtime::SingleLevelDiagnostics<
        Descriptor, RunOutput<Descriptor>>;

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
            controls_.diagnosticsInterval > 0 ||
            controls_.plotInterval > 0;
        plan.completedMacro = plan.completed;
        if (controls_.steadyCheckInterval > 0) {
            plan.completed.everyStep = true;
            plan.completedMacro.everyStep = true;
        }
        add_channel_cylinder_observation_requests(observation_, plan);
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
            "channel_cylinder_steady",
            controls_.steadyCheckInterval,
            controls_.stopOnSteady,
            controls_.steadyL2Tolerance};
    }

    ConstantExactReference exactReference() const noexcept
    {
        return {
            lbm_.rhoRefLb,
            uses_uniform_initialization(controls_.variant) ?
                controls_.uMeanLb :
                amrex::Real(0.0)};
    }

    Controls controls_;
    ChannelCylinderRuntimeConfig observation_;
    runtime::SingleLevelObservationPlan observationPlan_;
    config::ResolvedLbmConfig lbm_;
    std::unique_ptr<RunOutput<Descriptor>> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
    std::unique_ptr<ChannelCylinderObserver<Descriptor>> observer_;
};

} // namespace lbm::cases::channel_cylinder

#endif
