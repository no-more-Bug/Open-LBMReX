#ifndef LBM_TESTS_PERFORMANCE_PERFORMANCERUNSESSION_H
#define LBM_TESTS_PERFORMANCE_PERFORMANCERUNSESSION_H

#include "PerformanceControls.H"
#include "PerformanceCsv.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::performance {

namespace detail {

inline bool valid_mode(std::string const& mode) noexcept
{
    return mode == "smoke" || mode == "calibration" || mode == "baseline";
}

} // namespace detail

template <class Descriptor>
class PerformanceRunSession {
public:
    PerformanceRunSession() = default;

    PerformanceRunSession(
        PerformanceIdentity identity,
        PerformanceControls controls)
        : identity_(std::move(identity)), controls_(std::move(controls))
    {}

    runtime::SingleLevelRunSessionResult start(
        runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        for (int direction = 0; direction < 3; ++direction) {
            grid_[direction] = context.geometry.Domain().length(direction);
        }
        if (grid_ != identity_.grid ||
            context.boxArray.size() != identity_.boxCount) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "Performance workload grid or box identity mismatch"};
        }
        int const rankCount = amrex::ParallelDescriptor::NProcs();
        auto const& processorMap =
            context.distributionMapping.ProcessorMap();
        bool oneBoxPerRank =
            rankCount == identity_.boxCount &&
            processorMap.size() == context.boxArray.size();
        std::vector<int> boxesPerRank(rankCount, 0);
        for (int owner : processorMap) {
            if (owner < 0 || owner >= rankCount) {
                oneBoxPerRank = false;
                break;
            }
            ++boxesPerRank[owner];
        }
        for (int count : boxesPerRank) {
            oneBoxPerRank = oneBoxPerRank && count == 1;
        }
        if (!oneBoxPerRank) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "Performance workload requires exactly one box per MPI rank"};
        }
        globalValidCells_ = 1;
        for (int length : grid_) {
            globalValidCells_ *= static_cast<std::int64_t>(length);
        }
        started_ = true;
        return {};
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        return {};
    }

    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        if (!started_ || !record.has_value()) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "Performance run completed without a profiling record"};
        }
        if (record->nSteps != controls_.measurementSteps ||
            record->sampleStart != controls_.warmupSteps + 1 ||
            record->sampleEnd != controls_.maxStep ||
            record->sampleEnd - record->sampleStart + 1 != record->nSteps ||
            !std::isfinite(record->synchronizedWindow) ||
            record->synchronizedWindow <= 0.0) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "Performance profiling window is incomplete or invalid"};
        }
        try {
            write_performance_csv<Descriptor>(
                identity_,
                controls_,
                grid_,
                globalValidCells_,
                *record);
        } catch (std::exception const& exception) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                std::string("Failed to write performance CSV: ") +
                    exception.what()};
        }
        return {};
    }

private:
    PerformanceIdentity identity_;
    PerformanceControls controls_;
    std::array<int, 3> grid_{};
    std::int64_t globalValidCells_ = 0;
    bool started_ = false;
};

struct PerformancePreparation {
    PerformanceIdentity identity;
    PerformanceControls controls;

    template <class Descriptor>
    using RunSession = PerformanceRunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        auto reject = [&](char const* code, char const* subject, char const* message) {
            result.diagnostics.push_back({code, subject, message});
        };

        if (controls.workload != identity.workload) {
            reject(
                "performance_workload_mismatch",
                "performance.workload",
                "input workload does not match the compiled performance target");
        }
        if (!detail::valid_mode(controls.mode)) {
            reject(
                "invalid_performance_mode",
                "performance.mode",
                "performance mode must be smoke, calibration or baseline");
        }
        if (controls.repeat <= 0 || controls.outputCsv.empty()) {
            reject(
                "invalid_performance_output",
                "performance.output",
                "repeat must be positive and output_csv must be non-empty");
        }
        if (view.profiling.enabled != 1 ||
            view.profiling.measurementSteps <= 0 ||
            view.run.maxStep != view.profiling.warmupSteps +
                    view.profiling.measurementSteps) {
            reject(
                "invalid_performance_window",
                "profiling.window",
                "performance requires max_step = warmup_steps + measurement_steps");
        }
        for (int direction = 0; direction < 3; ++direction) {
            if (view.mesh.nCell[direction] != identity.grid[direction]) {
                reject(
                    "performance_grid_mismatch",
                    "mesh.n_cell",
                    "compiled performance grid identity was changed");
                break;
            }
        }
        if (view.mesh.maxGridSize != identity.maxGridSize) {
            reject(
                "performance_box_mismatch",
                "mesh.max_grid_size",
                "compiled performance max_grid_size identity was changed");
        }
        if (result.diagnostics.empty()) {
            result.session = RunSession<Descriptor>(
                std::move(identity), std::move(controls));
        }
        return result;
    }
};

} // namespace lbm::tests::performance

#endif
