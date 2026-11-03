#include "lbm/profiling/AmrTimestepProfiler.H"

namespace lbm::profiling {
namespace {
std::vector<AmrStageTiming*> entries(AmrProfileDurations& durations)
{
    std::vector<AmrStageTiming*> result;
    for (auto& level : durations.levels) for (auto& item : level) result.push_back(&item);
    for (auto& pair : durations.pairs) for (auto& item : pair) result.push_back(&item);
    result.push_back(&durations.hierarchy);
    result.push_back(&durations.boundaryPreparation);
    return result;
}
} // namespace

AmrPerformanceRecord make_amr_performance_record(
    TimestepPerformanceRecord const& window, std::vector<int> const& levelSteps,
    AmrProfileDurations const& local)
{
    AmrPerformanceRecord result;
    result.window = window;
    result.levelSteps = levelSteps;
    if (window.mode != TimestepProfilingMode::StageBreakdown) return result;
    result.criticalRank = local;
    result.rankMax = local;
    auto critical = entries(result.criticalRank);
    auto maxima = entries(result.rankMax);
    std::vector<amrex::Real> seconds;
    std::vector<int> calls;
    for (auto const* item : critical) { seconds.push_back(item->seconds); calls.push_back(item->calls); }
    amrex::ParallelDescriptor::ReduceRealMax(seconds.data(), int(seconds.size()));
    amrex::ParallelDescriptor::ReduceIntMax(calls.data(), int(calls.size()));
    for (int i=0; i<int(maxima.size()); ++i) *maxima[i] = {seconds[i], calls[i]};
    for (int i=0; i<int(critical.size()); ++i) {
        seconds[i] = critical[i]->seconds;
        calls[i] = critical[i]->calls;
    }
    amrex::ParallelDescriptor::Bcast(seconds.data(), seconds.size(), window.criticalRank);
    amrex::ParallelDescriptor::Bcast(calls.data(), calls.size(), window.criticalRank);
    for (int i=0; i<int(critical.size()); ++i) *critical[i] = {seconds[i], calls[i]};
    return result;
}

AmrTimestepProfiler::AmrTimestepProfiler(
    TimestepProfilingMode mode, int warmupSteps, int measurementSteps, int levels)
    : window_(mode, warmupSteps, measurementSteps)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(levels > 0 && warmupSteps >= 0 && measurementSteps >= 0,
                                    "AMR profiling requires a hierarchy and nonnegative window counts");
    levelSteps_.resize(levels, 0);
    durations_.levels.resize(levels);
    durations_.pairs.resize(levels-1);
}

std::optional<AmrPerformanceRecord> AmrTimestepProfiler::performanceRecord() const
{
    auto const record = window_.performanceRecord();
    if (!record) return std::nullopt;
    return make_amr_performance_record(*record, levelSteps_, durations_);
}

bool AmrProfileStep::stageBreakdownActive() const noexcept
{
    return profiler && window.stageBreakdownActive();
}

void AmrProfileStep::advanceLevel(int level) const
{
    if (profiler && window.active) ++profiler->levelSteps_[level];
}

void AmrProfileStep::addPair(int coarseLevel, AmrPairStage stage, amrex::Real seconds) const
{
    if (!stageBreakdownActive()) return;
    auto& timing = profiler->durations_.pairs[coarseLevel][static_cast<int>(stage)];
    timing.seconds += seconds;
    ++timing.calls;
}
} // namespace lbm::profiling
