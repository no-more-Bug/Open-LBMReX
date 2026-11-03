#ifndef LBM_PROFILING_AMRTIMESTEPPROFILER_HH
#define LBM_PROFILING_AMRTIMESTEPPROFILER_HH

namespace lbm::profiling {
template<class F>
void AmrProfileStep::measure(int level, AmrLevelStage stage, F&& operation) const
{
    if (!stageBreakdownActive()) {
        std::forward<F>(operation)();
        return;
    }
    auto& timing = profiler->durations_.levels[level][static_cast<int>(stage)];
    timing.seconds += TimestepProfiler::measureOperation(true, std::forward<F>(operation));
    ++timing.calls;
}

template<class F>
void AmrProfileStep::measureHierarchy(F&& operation) const
{
    if (!stageBreakdownActive()) {
        std::forward<F>(operation)();
        return;
    }
    auto& timing = profiler->durations_.hierarchy;
    timing.seconds += TimestepProfiler::measureOperation(true, std::forward<F>(operation));
    ++timing.calls;
}
template<class F>
void AmrProfileStep::measureBoundaryPreparation(F&& operation) const
{
    if (!stageBreakdownActive()) {
        std::forward<F>(operation)();
        return;
    }
    auto& timing = profiler->durations_.boundaryPreparation;
    timing.seconds += TimestepProfiler::measureOperation(true, std::forward<F>(operation));
    ++timing.calls;
}
} // namespace lbm::profiling
#endif
