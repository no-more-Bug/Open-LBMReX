#ifndef LBM_PROFILING_AMRTIMESTEPPROFILER_H
#define LBM_PROFILING_AMRTIMESTEPPROFILER_H

#include "lbm/profiling/TimestepProfiler.H"
#include <vector>

namespace lbm::profiling {

enum class AmrLevelStage { Collision, Communication, Streaming, BoundaryPre, BoundaryPost, Management };
enum class AmrPairStage { CoarseToFineCommunication, FineToCoarseCommunication, Explode, Coalesce };

struct AmrStageTiming {
    amrex::Real seconds = 0;
    int calls = 0;
};

struct AmrProfileDurations {
    std::vector<std::array<AmrStageTiming,6>> levels;
    std::vector<std::array<AmrStageTiming,4>> pairs;
    AmrStageTiming hierarchy;
    // Runtime redefinitions only; a nested subset of hierarchy, not an extra total.
    AmrStageTiming boundaryPreparation;
};

struct AmrPerformanceRecord {
    TimestepPerformanceRecord window;
    std::vector<int> levelSteps;
    // Empty in Throughput: unmeasured stages are not zero-valued measurements.
    AmrProfileDurations criticalRank;
    AmrProfileDurations rankMax;
};

AmrPerformanceRecord make_amr_performance_record(
    TimestepPerformanceRecord const& window, std::vector<int> const& levelSteps,
    AmrProfileDurations const& local);

class AmrTimestepProfiler;

/** Borrowed handle for one complete L0 step, shared by all recursive calls. */
struct AmrProfileStep {
    AmrTimestepProfiler* profiler = nullptr;
    TimestepProfileStep window;
    bool stageBreakdownActive() const noexcept;
    void advanceLevel(int level) const;
    template<class F> void measure(int level, AmrLevelStage stage, F&& operation) const;
    template<class F> void measureHierarchy(F&& operation) const;
    template<class F> void measureBoundaryPreparation(F&& operation) const;
    void addPair(int coarseLevel, AmrPairStage stage, amrex::Real seconds) const;
};

/** Case-owned AMR extension of the existing single-window profiler.
 * Capacity covers all allowed levels, including inactive levels with no calls.
 * The hierarchy size and controls agree on every rank. Keep this context across
 * segmented evolve calls; finalize it before an open-ended record is consumed.
 */
class AmrTimestepProfiler {
public:
    AmrTimestepProfiler(TimestepProfilingMode mode, int warmupSteps, int measurementSteps, int levels);
    int numLevels() const noexcept { return int(levelSteps_.size()); }
    AmrProfileStep beginStep(int step) { return {this, window_.beginStep(step)}; }
    void completeStep(AmrProfileStep const& step) { window_.completeStep(step.window); }
    void finalize() { window_.finalize(); }
    std::optional<AmrPerformanceRecord> performanceRecord() const;
    AmrProfileDurations const& localDurations() const noexcept { return durations_; }
    std::vector<int> const& levelSteps() const noexcept { return levelSteps_; }
private:
    friend struct AmrProfileStep;
    TimestepProfiler window_;
    std::vector<int> levelSteps_;
    AmrProfileDurations durations_;
};
} // namespace lbm::profiling

#include "lbm/profiling/AmrTimestepProfiler.HH"
#endif
