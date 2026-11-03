#ifndef LBM_RUNTIME_SINGLELEVELRUN_HH
#define LBM_RUNTIME_SINGLELEVELRUN_HH

#include "lbm/operators/level/CollisionOperator.H"
#include "lbm/operators/level/MacroscopicOperator.H"
#include "lbm/operators/level/PullStreamingOperator.H"
#include "lbm/solvers/BoundaryStepContext.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>

#include <memory>
#include <optional>
#include <utility>

namespace lbm::runtime {

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
SingleLevelRunReport run_single_level(
    PreparedSingleLevelRun<
        Descriptor,
        Dynamics,
        Initialization,
        RunSession> prepared)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        prepared.state_ != nullptr,
        "Single-level run requires one unconsumed prepared token");
    auto state = std::move(prepared.state_);
    SingleLevelRunReport report;
    SingleLevelRunContext<Descriptor> const context{
        state->lbm,
        state->geometry,
        state->boxArray,
        state->distributionMapping,
        state->boundarySystem.postStreamingPlan(),
        state->onLatticeReferenceAxis};

    SingleLevelRunSessionResult sessionResult =
        state->runSession.start(context);
    if (!sessionResult.ok()) {
        report.status = sessionResult.status;
        report.message = std::move(sessionResult.message);
        return report;
    }
    SingleLevelObservationPlan const observationPlan =
        state->runSession.observationPlan();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !observationPlan.macroAtInitialized ||
            observationPlan.observeInitialized,
        "Initialized macro observation requires an initialized published state");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !observationPlan.macroAtFinal || observationPlan.observeFinal,
        "Final macro observation requires a final published state");
    std::unique_ptr<MacroField> observationMacro;
    if (observationPlan.requiresMacroStorage()) {
        observationMacro = std::make_unique<MacroField>(
            "observation_macro", state->boxArray, state->distributionMapping);
    }
    auto refreshMacro = [&](LatticeField<Descriptor> const& current) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            observationMacro != nullptr,
            "Observation plan requested macro reconstruction without storage");
        operators::MacroscopicOperator<Descriptor>{}.apply(
            current, *observationMacro);
    };
    auto latticeTime = [](int step) noexcept {
        return static_cast<amrex::Real>(step);
    };
    auto physicalTimeAtStep = [&](int step) noexcept {
        return latticeTime(step) * state->lbm.uRefLb *
            (state->lbm.lengthRefPhys / state->lbm.lengthRefLb) /
            state->lbm.uRefPhys;
    };
    amrex::Real const dtPhys = physicalTimeAtStep(1);
    boundary::InternalForceBatch emptyForceBatch;

    state->boundarySystem.writeDiagnostics();
    initialization::initialize_populations<Descriptor>(
        state->initialization, *state->src, state->geometry);
    state->boundarySystem.initializeValidBoundaries(*state->src, 0.0);
    if (observationPlan.observeInitialized) {
        if (observationPlan.macroAtInitialized) {
            refreshMacro(*state->src);
        }
        SingleLevelPublishedState<Descriptor> const published(
            0,
            latticeTime(0),
            physicalTimeAtStep(0),
            *state->src,
            context,
            observationPlan.macroAtInitialized ? observationMacro.get() : nullptr,
            emptyForceBatch);
        sessionResult = state->runSession.observeInitialized(published);
        if (!sessionResult.ok()) {
            report.status = sessionResult.status;
            report.message = std::move(sessionResult.message);
            return report;
        }
    }

    operators::CollisionOperator<Descriptor, Dynamics> collision(
        std::move(state->dynamics));
    operators::PullStreamingOperator<Descriptor> streaming;

    for (int step = 1; step <= state->run.maxStep; ++step) {
        report.finalStep = step;
        const solvers::BoundaryStepContext boundaryStep(
            physicalTimeAtStep(step - 1), dtPhys);
        const profiling::TimestepProfileStep profileStep =
            state->profiler.beginStep(step);

        state->profiler.measureStage(
            profileStep,
            profiling::TimestepStage::Collision,
            [&] { collision.apply(*state->src); });
        state->profiler.measureStage(
            profileStep,
            profiling::TimestepStage::PreStream,
            [&] {
                boundaryStep.runPreStreaming(
                    [&](amrex::Real physicalTime) {
                        if (profileStep.stageBreakdownActive()) {
                            boundary::PreStreamingTimingSample timing;
                            state->boundarySystem.fillPreStreaming(
                                *state->src, physicalTime, &timing);
                            state->profiler.addPreStreamingBreakdown(
                                profileStep,
                                timing.fillBoundary,
                                timing.domainSetup,
                                timing.physicalGhosts);
                        } else {
                            state->boundarySystem.fillPreStreaming(
                                *state->src, physicalTime);
                        }
                    });
            });
        state->profiler.measureStage(
            profileStep,
            profiling::TimestepStage::Streaming,
            [&] { streaming.apply(*state->src, *state->dst); });
        state->profiler.measureStage(
            profileStep,
            profiling::TimestepStage::PostStream,
            [&] {
                boundaryStep.runPostStreaming(
                    [&](amrex::Real physicalTime) {
                        if (profileStep.stageBreakdownActive()) {
                            boundary::PostStreamingTimingSample timing;
                            state->boundarySystem.applyPostStreamingValidNodes(
                                *state->src,
                                *state->dst,
                                physicalTime,
                                &timing);
                            state->profiler.addPostStreamingBreakdown(
                                profileStep,
                                timing.domain,
                                timing.internalSolid);
                        } else {
                            state->boundarySystem.applyPostStreamingValidNodes(
                                *state->src, *state->dst, physicalTime);
                        }
                    });
            });

        bool stopOnSteady = false;
        state->profiler.measureStage(
            profileStep,
            profiling::TimestepStage::Diagnostics,
            [&] {
                if (observationPlan.forceCalculationDue(step)) {
                    const auto beginPhysicalTime = physicalTimeAtStep(step - 1);
                    state->boundarySystem.recordInternalForceSample(
                        *state->src, *state->dst,
                        {0, step, step, beginPhysicalTime, beginPhysicalTime + dtPhys});
                }
                std::swap(state->src, state->dst);
                boundary::InternalForceBatch forceBatch;
                if (observationPlan.forceDeliveryDue(step)) {
                    forceBatch =
                        state->boundarySystem
                            .takeGlobalInternalForceBatch();
                }
                if (observationPlan.completedDue(step, state->run.maxStep)) {
                    const bool needsMacro =
                        observationPlan.completedMacroDue(
                            step, state->run.maxStep);
                    if (needsMacro) {
                        refreshMacro(*state->src);
                    }
                    SingleLevelPublishedState<Descriptor> const published(
                        step,
                        latticeTime(step),
                        physicalTimeAtStep(step),
                        *state->src,
                        context,
                        needsMacro ? observationMacro.get() : nullptr,
                        forceBatch);
                    SingleLevelCompletedObservationResult completed =
                        state->runSession.observeCompleted(published);
                    if (!completed.ok()) {
                        report.status = completed.result.status;
                        report.message = std::move(completed.result.message);
                    }
                    stopOnSteady = completed.stopRequested;
                }
            });
        state->profiler.completeStep(profileStep);
        if (!report.ok()) {
            return report;
        }
        if (stopOnSteady) {
            report.stoppedOnSteady = true;
            break;
        }
    }

    state->profiler.finalize();

    boundary::InternalForceBatch finalForceBatch;
    if (observationPlan.forceCalculationInterval > 0) {
        finalForceBatch =
            state->boundarySystem.takeGlobalInternalForceBatch();
    }
    if (observationPlan.observeFinal) {
        if (observationPlan.macroAtFinal) {
            refreshMacro(*state->src);
        }
        SingleLevelPublishedState<Descriptor> const published(
            report.finalStep,
            latticeTime(report.finalStep),
            physicalTimeAtStep(report.finalStep),
            *state->src,
            context,
            observationPlan.macroAtFinal ? observationMacro.get() : nullptr,
            finalForceBatch);
        sessionResult = state->runSession.observeFinal(published);
        if (!sessionResult.ok()) {
            report.status = sessionResult.status;
            report.message = std::move(sessionResult.message);
            return report;
        }
    }

    std::optional<profiling::TimestepPerformanceRecord> const performance =
        state->profiler.performanceRecord();
    sessionResult = state->runSession.writePerformance(performance);
    if (!sessionResult.ok()) {
        report.status = sessionResult.status;
        report.message = std::move(sessionResult.message);
    }
    return report;
}

} // namespace lbm::runtime

#endif
