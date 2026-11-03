#ifndef LBM_TESTS_INTEGRATION_APP_TIMESTEPPROFILERINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_APP_TIMESTEPPROFILERINTEGRATIONTESTS_H

#include "lbm/profiling/TimestepProfiler.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <chrono>
#include <optional>
#include <thread>

namespace lbm::app {

namespace timestep_profiler_integration_test_detail {

using lbm::profiling::TimestepProfileDurations;

inline void expect_duration_fields_equal(
    TimestepProfileDurations const& actual,
    TimestepProfileDurations const& expected)
{
    AMREX_ALWAYS_ASSERT(actual.collision == expected.collision);
    AMREX_ALWAYS_ASSERT(actual.preStream == expected.preStream);
    AMREX_ALWAYS_ASSERT(
        actual.preStreamFillBoundary ==
        expected.preStreamFillBoundary);
    AMREX_ALWAYS_ASSERT(
        actual.preStreamSetup == expected.preStreamSetup);
    AMREX_ALWAYS_ASSERT(
        actual.preStreamPhysicalGhosts ==
        expected.preStreamPhysicalGhosts);
    AMREX_ALWAYS_ASSERT(actual.streaming == expected.streaming);
    AMREX_ALWAYS_ASSERT(
        actual.domainPostStream == expected.domainPostStream);
    AMREX_ALWAYS_ASSERT(
        actual.internalSolidReconstruction ==
        expected.internalSolidReconstruction);
    AMREX_ALWAYS_ASSERT(actual.postStream == expected.postStream);
    AMREX_ALWAYS_ASSERT(actual.diagnostics == expected.diagnostics);
    AMREX_ALWAYS_ASSERT(actual.total == expected.total);
}

inline TimestepProfileDurations rank_dependent_durations(int rank)
{
    TimestepProfileDurations durations;
    durations.collision = 10.0 + rank;
    durations.preStream = 20.0 + rank;
    durations.preStreamFillBoundary = 30.0 + rank;
    durations.preStreamSetup = 40.0 + rank;
    durations.preStreamPhysicalGhosts = 50.0 + rank;
    durations.streaming = 60.0 + rank;
    durations.domainPostStream = 70.0 + rank;
    durations.internalSolidReconstruction = 80.0 + rank;
    durations.postStream = 90.0 + rank;
    durations.diagnostics = 100.0 + rank;
    durations.total = 110.0 + rank;
    return durations;
}

inline TimestepProfileDurations expected_average_durations(
    TimestepProfileDurations const& window,
    amrex::Real inverseSteps)
{
    TimestepProfileDurations expected;
    expected.collision = window.collision * inverseSteps;
    expected.preStream = window.preStream * inverseSteps;
    expected.preStreamFillBoundary =
        window.preStreamFillBoundary * inverseSteps;
    expected.preStreamSetup =
        window.preStreamSetup * inverseSteps;
    expected.preStreamPhysicalGhosts =
        window.preStreamPhysicalGhosts * inverseSteps;
    expected.streaming = window.streaming * inverseSteps;
    expected.domainPostStream = window.domainPostStream * inverseSteps;
    expected.internalSolidReconstruction =
        window.internalSolidReconstruction * inverseSteps;
    expected.postStream = window.postStream * inverseSteps;
    expected.diagnostics = window.diagnostics * inverseSteps;
    expected.total = window.total * inverseSteps;
    return expected;
}

} // namespace timestep_profiler_integration_test_detail

/**
 * @brief Check disabled behavior, window selection, and MPI-max record semantics.
 *
 * Every rank enters the same profiler collectives. Synthetic rank-dependent
 * totals make the expected all-rank maxima deterministic without depending on
 * wall-clock resolution.
 */
inline void run_timestep_profiler_integration_tests()
{
    using namespace timestep_profiler_integration_test_detail;
    using namespace lbm::profiling;

    {
        TimestepProfiler profiler(0, 0, 0);
        AMREX_ALWAYS_ASSERT(
            profiler.mode() == TimestepProfilingMode::Disabled);
        int operationCount = 0;
        for (int step = 1; step <= 3; ++step) {
            TimestepProfileStep const profileStep =
                profiler.beginStep(step);
            AMREX_ALWAYS_ASSERT(!profileStep.active);
            profiler.measureStage(
                profileStep,
                TimestepStage::Collision,
                [&] { ++operationCount; });
            profiler.completeStep(profileStep);
        }
        profiler.finalize();
        AMREX_ALWAYS_ASSERT(operationCount == 3);
        AMREX_ALWAYS_ASSERT(!profiler.hasSample());
        AMREX_ALWAYS_ASSERT(!profiler.performanceRecord().has_value());
    }

    {
        TimestepProfiler profiler(1, 2, 3);
        AMREX_ALWAYS_ASSERT(
            profiler.mode() == TimestepProfilingMode::StageBreakdown);
        int operationCount = 0;
        for (int step = 1; step <= 7; ++step) {
            TimestepProfileStep const profileStep =
                profiler.beginStep(step);
            AMREX_ALWAYS_ASSERT(
                profileStep.active == (step >= 3 && step <= 5));
            profiler.measureStage(
                profileStep,
                TimestepStage::Collision,
                [&] { ++operationCount; });
            profiler.addPreStreamingBreakdown(
                profileStep, 1.0, 2.0, 3.0);
            profiler.addPostStreamingBreakdown(
                profileStep, 4.0, 5.0);
            profiler.completeStep(profileStep);
        }

        AMREX_ALWAYS_ASSERT(operationCount == 7);
        TimestepProfileSample const& sample = profiler.sample();
        AMREX_ALWAYS_ASSERT(sample.sampleStart == 3);
        AMREX_ALWAYS_ASSERT(sample.sampleEnd == 5);
        AMREX_ALWAYS_ASSERT(sample.nSteps == 3);
        AMREX_ALWAYS_ASSERT(
            sample.window.preStreamFillBoundary == 3.0);
        AMREX_ALWAYS_ASSERT(sample.window.preStreamSetup == 6.0);
        AMREX_ALWAYS_ASSERT(
            sample.window.preStreamPhysicalGhosts == 9.0);
        AMREX_ALWAYS_ASSERT(sample.window.domainPostStream == 12.0);
        AMREX_ALWAYS_ASSERT(
            sample.window.internalSolidReconstruction == 15.0);
        AMREX_ALWAYS_ASSERT(sample.window.collision >= 0.0);
        AMREX_ALWAYS_ASSERT(sample.window.total >= 0.0);
        AMREX_ALWAYS_ASSERT(sample.synchronizedWindow >= 0.0);

        std::optional<TimestepPerformanceRecord> const record =
            profiler.performanceRecord();
        AMREX_ALWAYS_ASSERT(record.has_value());
        AMREX_ALWAYS_ASSERT(
            record->mode == TimestepProfilingMode::StageBreakdown &&
            !record->formalMlups && record->criticalRank >= 0);
        AMREX_ALWAYS_ASSERT(record->sampleStart == 3);
        AMREX_ALWAYS_ASSERT(record->sampleEnd == 5);
        AMREX_ALWAYS_ASSERT(record->nSteps == 3);
        AMREX_ALWAYS_ASSERT(
            record->rankMaxWindow.collision >=
            sample.window.collision);
        AMREX_ALWAYS_ASSERT(
            record->perStepAverage.collision ==
            record->rankMaxWindow.collision * (1.0 / 3.0));
        AMREX_ALWAYS_ASSERT(
            record->rankMaxWindow.preStreamFillBoundary == 3.0);
        AMREX_ALWAYS_ASSERT(
            record->perStepAverage.preStreamFillBoundary == 1.0);
        AMREX_ALWAYS_ASSERT(
            record->rankMaxWindow.domainPostStream == 12.0);
        AMREX_ALWAYS_ASSERT(
            record->perStepAverage.internalSolidReconstruction == 5.0);
        AMREX_ALWAYS_ASSERT(
            record->synchronizedWindow >= sample.synchronizedWindow);
    }

    {
        TimestepProfiler profiler(
            TimestepProfilingMode::Throughput, 1, 2);
        int operationCount = 0;
        for (int step = 1; step <= 4; ++step) {
            TimestepProfileStep const profileStep =
                profiler.beginStep(step);
            AMREX_ALWAYS_ASSERT(
                profileStep.active == (step == 2 || step == 3));
            AMREX_ALWAYS_ASSERT(!profileStep.stageBreakdownActive());
            profiler.measureStage(
                profileStep,
                TimestepStage::Collision,
                [&] { ++operationCount; });
            profiler.addPreStreamingBreakdown(
                profileStep, 1.0, 2.0, 3.0);
            profiler.addPostStreamingBreakdown(
                profileStep, 4.0, 5.0);
            profiler.completeStep(profileStep);
        }
        profiler.finalize();
        AMREX_ALWAYS_ASSERT(operationCount == 4);
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleStart == 2);
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleEnd == 3);
        AMREX_ALWAYS_ASSERT(profiler.sample().nSteps == 2);
        expect_duration_fields_equal(
            profiler.sample().window, TimestepProfileDurations{});
        auto const record = profiler.performanceRecord();
        AMREX_ALWAYS_ASSERT(record.has_value());
        AMREX_ALWAYS_ASSERT(
            record->mode == TimestepProfilingMode::Throughput &&
            record->formalMlups && record->criticalRank == -1 &&
            record->synchronizedWindow >= 0.0);
        expect_duration_fields_equal(
            record->rankMaxWindow, TimestepProfileDurations{});
        expect_duration_fields_equal(
            record->criticalRankWindow, TimestepProfileDurations{});
    }

    {
        TimestepProfiler profiler(1, 1, 0);
        for (int step = 1; step <= 4; ++step) {
            TimestepProfileStep const profileStep =
                profiler.beginStep(step);
            AMREX_ALWAYS_ASSERT(
                profileStep.active == (step >= 2));
            profiler.completeStep(profileStep);
        }
        profiler.finalize();
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleStart == 2);
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleEnd == 4);
        AMREX_ALWAYS_ASSERT(profiler.sample().nSteps == 3);
        AMREX_ALWAYS_ASSERT(profiler.sample().synchronizedWindow >= 0.0);
    }

    {
        TimestepProfiler profiler(
            TimestepProfilingMode::Throughput, 1, 0);
        for (int step = 1; step <= 4; ++step) {
            TimestepProfileStep const profileStep =
                profiler.beginStep(step);
            profiler.measureStage(
                profileStep, TimestepStage::Streaming, [] {});
            profiler.completeStep(profileStep);
        }
        profiler.finalize();
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleStart == 2);
        AMREX_ALWAYS_ASSERT(profiler.sample().sampleEnd == 4);
        AMREX_ALWAYS_ASSERT(profiler.sample().nSteps == 3);
        expect_duration_fields_equal(
            profiler.sample().window, TimestepProfileDurations{});
        auto const record = profiler.performanceRecord();
        AMREX_ALWAYS_ASSERT(
            record.has_value() && record->formalMlups &&
            record->mode == TimestepProfilingMode::Throughput &&
            record->synchronizedWindow >= 0.0);
    }

    {
        const int rank = amrex::ParallelDescriptor::MyProc();
        const int maxRank =
            amrex::ParallelDescriptor::NProcs() - 1;
        TimestepProfileSample sample;
        sample.mode = TimestepProfilingMode::StageBreakdown;
        sample.sampleStart = 4;
        sample.sampleEnd = 5;
        sample.nSteps = 2;
        sample.window = rank_dependent_durations(rank);
        sample.synchronizedWindow = 100.0 + rank;

        TimestepPerformanceRecord const record =
            make_timestep_performance_record(sample);
        AMREX_ALWAYS_ASSERT(record.sampleStart == 4);
        AMREX_ALWAYS_ASSERT(record.sampleEnd == 5);
        AMREX_ALWAYS_ASSERT(record.nSteps == 2);
        TimestepProfileDurations const expectedWindow =
            rank_dependent_durations(maxRank);
        expect_duration_fields_equal(
            record.rankMaxWindow, expectedWindow);
        expect_duration_fields_equal(
            record.perStepAverage,
            expected_average_durations(expectedWindow, 0.5));
        AMREX_ALWAYS_ASSERT(record.criticalRank == maxRank);
        expect_duration_fields_equal(
            record.criticalRankWindow, expectedWindow);
        AMREX_ALWAYS_ASSERT(
            record.synchronizedWindow == 100.0 + maxRank);
    }

    {
        const int rank = amrex::ParallelDescriptor::MyProc();
        TimestepProfileSample sample;
        sample.mode = TimestepProfilingMode::StageBreakdown;
        sample.sampleStart = 1;
        sample.sampleEnd = 1;
        sample.nSteps = 1;
        sample.window = rank_dependent_durations(rank);
        sample.window.total = 123.0;
        sample.synchronizedWindow = 123.0;

        TimestepPerformanceRecord const record =
            make_timestep_performance_record(sample);
        TimestepProfileDurations expected =
            rank_dependent_durations(0);
        expected.total = 123.0;
        AMREX_ALWAYS_ASSERT(record.criticalRank == 0);
        expect_duration_fields_equal(
            record.criticalRankWindow, expected);
    }

    if (amrex::ParallelDescriptor::NProcs() >= 2) {
        TimestepProfiler profiler(1, 0, 1);
        TimestepProfileStep const profileStep = profiler.beginStep(1);
        const int slowRank = amrex::ParallelDescriptor::NProcs() - 1;
        if (amrex::ParallelDescriptor::MyProc() == slowRank) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        profiler.completeStep(profileStep);

        amrex::Real minimumWindow = profiler.sample().synchronizedWindow;
        amrex::ParallelDescriptor::ReduceRealMin(minimumWindow);
        AMREX_ALWAYS_ASSERT(minimumWindow >= 0.025);
        AMREX_ALWAYS_ASSERT(
            profiler.performanceRecord()->synchronizedWindow >=
            minimumWindow);
    }

    amrex::Print() << "Timestep profiler integration tests passed.\n";
}

} // namespace lbm::app

#endif
