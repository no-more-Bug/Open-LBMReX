#ifndef LBM_PROFILING_TIMESTEPPROFILER_H
#define LBM_PROFILING_TIMESTEPPROFILER_H

#include <AMReX_BLassert.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <array>
#include <optional>
#include <utility>

namespace lbm::profiling {

enum class TimestepProfilingMode {
    Disabled,
    Throughput,
    StageBreakdown
};

inline bool is_valid(TimestepProfilingMode mode) noexcept
{
    return mode == TimestepProfilingMode::Disabled ||
        mode == TimestepProfilingMode::Throughput ||
        mode == TimestepProfilingMode::StageBreakdown;
}

inline char const* to_string(TimestepProfilingMode mode) noexcept
{
    switch (mode) {
    case TimestepProfilingMode::Disabled:
        return "disabled";
    case TimestepProfilingMode::Throughput:
        return "throughput";
    case TimestepProfilingMode::StageBreakdown:
        return "synchronized_stage_breakdown";
    }
    return "unknown";
}

/**
 * @brief Complete solver stages that can contribute to a timestep sample.
 *
 * The solver, not the profiler, owns these stage boundaries. In particular,
 * PostStream surrounds the complete boundary-system call; this enum must not
 * be used to push timing state into boundary executors or reconstruction
 * kernels.
 */
enum class TimestepStage {
    Collision,
    PreStream,
    Streaming,
    PostStream,
    Diagnostics
};

/**
 * @brief Timing totals accumulated over one local-rank measurement window.
 *
 * Main-stage fields cover complete solver stages. PreStream and PostStream
 * detail fields preserve instrumentation inside BoundarySystem and are not
 * expected to sum exactly to their outer stage after independent MPI maxima.
 */
struct TimestepProfileDurations {
    amrex::Real collision = 0.0;
    amrex::Real preStream = 0.0;
    amrex::Real preStreamFillBoundary = 0.0;
    amrex::Real preStreamSetup = 0.0;
    amrex::Real preStreamPhysicalGhosts = 0.0;
    amrex::Real streaming = 0.0;
    amrex::Real domainPostStream = 0.0;
    amrex::Real internalSolidReconstruction = 0.0;
    amrex::Real postStream = 0.0;
    amrex::Real diagnostics = 0.0;
    amrex::Real total = 0.0;
};

/**
 * @brief Local-rank sampling window and its cumulative timing totals.
 *
 * sampleStart/sampleEnd are one-based solver iteration numbers. nSteps counts
 * completed sampled iterations, so a measurement limit is enforced only after
 * completeStep() records the current iteration.
 */
struct TimestepProfileSample {
    TimestepProfilingMode mode = TimestepProfilingMode::Disabled;
    int sampleStart = -1;
    int sampleEnd = -1;
    int nSteps = 0;
    TimestepProfileDurations window;
    amrex::Real synchronizedWindow = 0.0;
};

/**
 * @brief Immutable-by-convention data prepared for performance serialization.
 *
 * Each rankMaxWindow field is reduced independently with an all-rank maximum,
 * matching the historical performance.csv semantics. synchronizedWindow is
 * the independently reduced wall time between the two measurement-window
 * barriers. perStepAverage is the reduced local window total divided by
 * nSteps; it is not an average of per-rank or per-step maxima.
 */
struct TimestepPerformanceRecord {
    TimestepProfilingMode mode = TimestepProfilingMode::Disabled;
    bool formalMlups = false;
    int sampleStart = -1;
    int sampleEnd = -1;
    int nSteps = 0;
    TimestepProfileDurations perStepAverage;
    TimestepProfileDurations rankMaxWindow;
    amrex::Real synchronizedWindow = 0.0;
    int criticalRank = -1;
    TimestepProfileDurations criticalRankWindow;
};

/**
 * @brief Handle for one solver iteration's optional measurement.
 *
 * A handle is created at the step boundary before collision and completed
 * after diagnostics. It carries profiling wall-clock state only; it is
 * unrelated to the physical boundary evaluation time t_n.
 */
struct TimestepProfileStep {
    int step = 0;
    bool active = false;
    TimestepProfilingMode mode = TimestepProfilingMode::Disabled;
    amrex::Real windowStart = 0.0;

    bool stageBreakdownActive() const noexcept
    {
        return active && mode == TimestepProfilingMode::StageBreakdown;
    }
};

namespace detail {

inline amrex::Real&
duration_for_stage(
    TimestepProfileDurations& durations,
    TimestepStage stage) noexcept
{
    switch (stage) {
    case TimestepStage::Collision:
        return durations.collision;
    case TimestepStage::PreStream:
        return durations.preStream;
    case TimestepStage::Streaming:
        return durations.streaming;
    case TimestepStage::PostStream:
        return durations.postStream;
    case TimestepStage::Diagnostics:
        return durations.diagnostics;
    }
    AMREX_ASSERT(false);
    return durations.collision;
}

inline TimestepProfileDurations
scaled_durations(
    TimestepProfileDurations const& durations,
    amrex::Real scale) noexcept
{
    TimestepProfileDurations result;
    result.collision = durations.collision * scale;
    result.preStream = durations.preStream * scale;
    result.preStreamFillBoundary =
        durations.preStreamFillBoundary * scale;
    result.preStreamSetup = durations.preStreamSetup * scale;
    result.preStreamPhysicalGhosts =
        durations.preStreamPhysicalGhosts * scale;
    result.streaming = durations.streaming * scale;
    result.domainPostStream = durations.domainPostStream * scale;
    result.internalSolidReconstruction =
        durations.internalSolidReconstruction * scale;
    result.postStream = durations.postStream * scale;
    result.diagnostics = durations.diagnostics * scale;
    result.total = durations.total * scale;
    return result;
}

inline std::array<amrex::Real, 11>
duration_values(TimestepProfileDurations const& durations) noexcept
{
    return {
        durations.collision,
        durations.preStream,
        durations.preStreamFillBoundary,
        durations.preStreamSetup,
        durations.preStreamPhysicalGhosts,
        durations.streaming,
        durations.domainPostStream,
        durations.internalSolidReconstruction,
        durations.postStream,
        durations.diagnostics,
        durations.total};
}

inline TimestepProfileDurations durations_from_values(
    std::array<amrex::Real, 11> const& values) noexcept
{
    TimestepProfileDurations durations;
    durations.collision = values[0];
    durations.preStream = values[1];
    durations.preStreamFillBoundary = values[2];
    durations.preStreamSetup = values[3];
    durations.preStreamPhysicalGhosts = values[4];
    durations.streaming = values[5];
    durations.domainPostStream = values[6];
    durations.internalSolidReconstruction = values[7];
    durations.postStream = values[8];
    durations.diagnostics = values[9];
    durations.total = values[10];
    return durations;
}

inline void reduce_duration_maxima(TimestepProfileDurations& reduced)
{
    amrex::ParallelDescriptor::ReduceRealMax(reduced.collision);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.preStream);
    amrex::ParallelDescriptor::ReduceRealMax(
        reduced.preStreamFillBoundary);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.preStreamSetup);
    amrex::ParallelDescriptor::ReduceRealMax(
        reduced.preStreamPhysicalGhosts);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.streaming);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.domainPostStream);
    amrex::ParallelDescriptor::ReduceRealMax(
        reduced.internalSolidReconstruction);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.postStream);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.diagnostics);
    amrex::ParallelDescriptor::ReduceRealMax(reduced.total);
}

} // namespace detail

/**
 * @brief Reduce one completed local sample into a serialization-ready record.
 *
 * All ranks in the active AMReX communicator must call this function in the
 * same order with the same sample indices and nSteps. AMReX ReduceRealMax()
 * performs an all-reduce, so every rank receives the same timing maxima.
 */
inline TimestepPerformanceRecord
make_timestep_performance_record(TimestepProfileSample const& sample)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        sample.nSteps > 0,
        "A timestep performance record requires at least one sampled step");

    TimestepProfileDurations reduced;
    int criticalRank = -1;
    TimestepProfileDurations criticalRankWindow;
    if (sample.mode == TimestepProfilingMode::StageBreakdown) {
        reduced = sample.window;
        detail::reduce_duration_maxima(reduced);
        int candidateRank = sample.window.total == reduced.total
            ? amrex::ParallelDescriptor::MyProc()
            : amrex::ParallelDescriptor::NProcs();
        amrex::ParallelDescriptor::ReduceIntMin(candidateRank);
        criticalRank = candidateRank;
        std::array<amrex::Real, 11> criticalValues =
            detail::duration_values(sample.window);
        amrex::ParallelDescriptor::Bcast(
            criticalValues.data(), criticalValues.size(), criticalRank);
        criticalRankWindow =
            detail::durations_from_values(criticalValues);
    }
    amrex::Real synchronizedWindow = sample.synchronizedWindow;
    amrex::ParallelDescriptor::ReduceRealMax(synchronizedWindow);

    TimestepPerformanceRecord record;
    record.mode = sample.mode;
    record.formalMlups = sample.mode == TimestepProfilingMode::Throughput;
    record.sampleStart = sample.sampleStart;
    record.sampleEnd = sample.sampleEnd;
    record.nSteps = sample.nSteps;
    record.rankMaxWindow = reduced;
    record.synchronizedWindow = synchronizedWindow;
    record.criticalRank = criticalRank;
    record.criticalRankWindow = criticalRankWindow;
    record.perStepAverage = detail::scaled_durations(
        reduced,
        1.0 / static_cast<amrex::Real>(sample.nSteps));
    return record;
}

/**
 * @brief Own profiling-window state, stage timing, and final MPI reduction.
 *
 * The profiler is define-once/run-many state owned by one solver run. All
 * ranks must receive identical enabled/warmup/measurement controls. For an
 * active step, the solver must call completeStep() exactly once after all
 * measured stages; performanceRecord() is collective when it returns data and
 * must therefore be called from a rank-symmetric branch.
 * beginStep() and completeStep()/finalize() place one barrier at each end of
 * the complete measurement window, never inside its measured iterations.
 *
 * StageBreakdown synchronizes the active AMReX GPU stream after each measured
 * operation. Throughput executes stages directly and synchronizes only at the
 * two complete-window boundaries. Disabled performs no profiling timing,
 * synchronization, or collective.
 */
class TimestepProfiler {
public:
    /**
     * @brief Configure one solver-run measurement window.
     *
     * @pre enabled is 0 or 1, and both step counts are non-negative. These
     * controls have already been resolved identically on every rank before the
     * solver starts.
     */
    TimestepProfiler(
        int enabled,
        int warmupSteps,
        int measurementSteps) noexcept
        : TimestepProfiler(
              enabled != 0
                  ? TimestepProfilingMode::StageBreakdown
                  : TimestepProfilingMode::Disabled,
              warmupSteps,
              measurementSteps)
    {}

    TimestepProfiler(
        TimestepProfilingMode mode,
        int warmupSteps,
        int measurementSteps) noexcept
        : mode_(mode),
          warmupSteps_(warmupSteps),
          measurementSteps_(measurementSteps)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            is_valid(mode_), "Unknown timestep profiling mode");
        sample_.mode = mode_;
    }

    bool enabled() const noexcept
    {
        return mode_ == TimestepProfilingMode::Throughput ||
            mode_ == TimestepProfilingMode::StageBreakdown;
    }

    TimestepProfilingMode mode() const noexcept
    {
        return mode_;
    }

    bool stepActive(int step) const noexcept
    {
        if (!enabled() || step <= warmupSteps_) {
            return false;
        }
        return measurementSteps_ == 0 ||
               sample_.nSteps < measurementSteps_;
    }

    /**
     * @brief Open the optional timing envelope before collision for one step.
     *
     * The first active call drains prior backend work, performs the single
     * start barrier, and opens the synchronized measurement window. Later
     * measured steps perform no start-side synchronization.
     */
    TimestepProfileStep beginStep(int step)
    {
        const bool active = stepActive(step);
        if (active && !windowOpen_) {
            synchronize();
            amrex::ParallelDescriptor::Barrier();
            synchronizedWindowStart_ = now();
            windowOpen_ = true;
        }
        return TimestepProfileStep{
            step,
            active,
            active ? mode_ : TimestepProfilingMode::Disabled,
            active && mode_ == TimestepProfilingMode::StageBreakdown
                ? now()
                : 0.0};
    }

    /**
     * @brief Execute one solver-owned stage and accumulate its local duration.
     *
     * The operation always executes exactly once. StageBreakdown stops the
     * timer only after synchronizing the active AMReX GPU stream; Throughput
     * and Disabled add no stage timer or synchronization.
     */
    template <class F>
    void measureStage(
        TimestepProfileStep const& profileStep,
        TimestepStage stage,
        F&& operation)
    {
        if (!profileStep.stageBreakdownActive()) {
            std::forward<F>(operation)();
            return;
        }

        addStageDuration(profileStep, stage, measureOperation(true, std::forward<F>(operation)));
    }

    /** Shared completed-operation timer, including optional AMR local samples. */
    template<class F>
    static amrex::Real measureOperation(bool active, F&& operation)
    {
        if (!active) {
            std::forward<F>(operation)();
            return 0;
        }
        amrex::Real const start = now();
        std::forward<F>(operation)();
        synchronize();
        return now()-start;
    }

    /**
     * @brief Preserve the existing pre-stream sub-stage measurements.
     *
     * Scalar inputs intentionally avoid any dependency on BoundarySystem or
     * PreStreamingTimingSample. The boundary runtime continues to own how each
     * sub-duration is measured.
     */
    void addPreStreamingBreakdown(
        TimestepProfileStep const& profileStep,
        amrex::Real fillBoundary,
        amrex::Real setup,
        amrex::Real physicalGhosts) noexcept
    {
        if (!profileStep.stageBreakdownActive()) {
            return;
        }
        sample_.window.preStreamFillBoundary += fillBoundary;
        sample_.window.preStreamSetup += setup;
        sample_.window.preStreamPhysicalGhosts += physicalGhosts;
    }

    /** Add local timings measured inside the complete post-stream stage. */
    void addPostStreamingBreakdown(
        TimestepProfileStep const& profileStep,
        amrex::Real domain,
        amrex::Real internalSolid) noexcept
    {
        if (!profileStep.stageBreakdownActive()) {
            return;
        }
        sample_.window.domainPostStream += domain;
        sample_.window.internalSolidReconstruction += internalSolid;
    }

    /**
     * @brief Close the whole-step envelope after diagnostics.
     *
     * @pre For each active handle returned by beginStep(), the solver calls
     * this function exactly once and only after all five measured stages.
     */
    void completeStep(TimestepProfileStep const& profileStep)
    {
        if (!profileStep.active) {
            return;
        }

        if (profileStep.stageBreakdownActive()) {
            synchronize();
            sample_.window.total += now() - profileStep.windowStart;
        }
        if (sample_.nSteps == 0) {
            sample_.sampleStart = profileStep.step;
        }
        sample_.sampleEnd = profileStep.step;
        ++sample_.nSteps;
        if (measurementSteps_ > 0 &&
            sample_.nSteps == measurementSteps_) {
            closeWindow();
        }
    }

    /** Close an open-ended or early-stopped window before final observations. */
    void finalize()
    {
        if (windowOpen_) {
            closeWindow();
        }
    }

    bool hasSample() const noexcept
    {
        return sample_.nSteps > 0;
    }

    TimestepProfileSample const& sample() const noexcept
    {
        return sample_;
    }

    /**
     * @brief Prepare the all-rank max-window record, or return no record.
     *
     * The no-sample path performs no collective. Once a sample exists this is
     * an MPI collective and must be called by every rank in the same order.
     */
    std::optional<TimestepPerformanceRecord> performanceRecord() const
    {
        if (!hasSample()) {
            return std::nullopt;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            !windowOpen_,
            "TimestepProfiler::finalize must close the measurement window "
            "before performanceRecord");
        return make_timestep_performance_record(sample_);
    }

private:
    void addStageDuration(
        TimestepProfileStep const& profileStep,
        TimestepStage stage,
        amrex::Real duration) noexcept
    {
        if (profileStep.stageBreakdownActive()) {
            detail::duration_for_stage(sample_.window, stage) += duration;
        }
    }

    static amrex::Real now() noexcept
    {
        return static_cast<amrex::Real>(
            amrex::ParallelDescriptor::second());
    }

    static void synchronize()
    {
        amrex::Gpu::streamSynchronize();
    }

    void closeWindow()
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            windowOpen_,
            "Cannot close an inactive timestep measurement window");
        if (mode_ == TimestepProfilingMode::Throughput) {
            synchronize();
        }
        amrex::ParallelDescriptor::Barrier();
        sample_.synchronizedWindow = now() - synchronizedWindowStart_;
        windowOpen_ = false;
    }

    TimestepProfilingMode mode_ = TimestepProfilingMode::Disabled;
    int warmupSteps_ = 0;
    int measurementSteps_ = 0;
    TimestepProfileSample sample_;
    bool windowOpen_ = false;
    amrex::Real synchronizedWindowStart_ = 0.0;
};

} // namespace lbm::profiling

#endif
