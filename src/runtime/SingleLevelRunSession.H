#ifndef LBM_RUNTIME_SINGLELEVELRUNSESSION_H
#define LBM_RUNTIME_SINGLELEVELRUNSESSION_H

#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/fields/MacroField.H"
#include "lbm/profiling/TimestepProfiler.H"
#include "runtime/SingleLevelRunReport.H"
#include "runtime/SingleLevelSetupDiagnostic.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

#include <array>
#include <optional>
#include <vector>

namespace lbm::runtime {

template <class Descriptor>
struct SingleLevelRunContext {
    config::ResolvedLbmConfig const& lbm;
    amrex::Geometry const& geometry;
    amrex::BoxArray const& boxArray;
    amrex::DistributionMapping const& distributionMapping;
    boundary::PostStreamingBoundaryPlan const& postStreamingPlan;
    int onLatticeReferenceAxis = -1;
};

/** Fixed, allocation-free cadence for completed published states. */
struct SingleLevelObservationCadence {
    bool everyStep = false;
    bool includeMaxStep = false;

    void addInterval(int interval)
    {
        if (interval <= 0) {
            return;
        }
        for (int index = 0; index < intervalCount_; ++index) {
            if (intervals_[index] == interval) {
                return;
            }
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            intervalCount_ < static_cast<int>(intervals_.size()),
            "Single-level observation cadence capacity exceeded");
        intervals_[intervalCount_++] = interval;
    }

    bool due(int step, int maxStep) const noexcept
    {
        if (everyStep || (includeMaxStep && step == maxStep)) {
            return true;
        }
        for (int index = 0; index < intervalCount_; ++index) {
            if (step % intervals_[index] == 0) {
                return true;
            }
        }
        return false;
    }

    bool hasIntervals() const noexcept { return intervalCount_ != 0; }

private:
    std::array<int, 4> intervals_{};
    int intervalCount_ = 0;
};

/** Immutable work plan resolved before the first published-state observation. */
struct SingleLevelObservationPlan {
    bool observeInitialized = false;
    bool observeFinal = false;
    bool macroAtInitialized = false;
    bool macroAtFinal = false;
    SingleLevelObservationCadence completed;
    SingleLevelObservationCadence completedMacro;
    int forceCalculationInterval = 0;
    int forceDeliveryInterval = 0;

    bool completedDue(int step, int maxStep) const noexcept
    {
        return completed.due(step, maxStep) ||
            completedMacro.due(step, maxStep) || forceDeliveryDue(step);
    }

    bool completedMacroDue(int step, int maxStep) const noexcept
    {
        return completedMacro.due(step, maxStep);
    }

    bool forceCalculationDue(int step) const noexcept
    {
        return forceCalculationInterval > 0 && step > 0 &&
            step % forceCalculationInterval == 0;
    }

    bool forceDeliveryDue(int step) const noexcept
    {
        return forceDeliveryInterval > 0 && step > 0 &&
            step % forceDeliveryInterval == 0;
    }

    bool requiresMacroStorage() const noexcept
    {
        return macroAtInitialized || macroAtFinal ||
            completedMacro.everyStep || completedMacro.includeMaxStep ||
            completedMacro.hasIntervals();
    }
};

/**
 * Read-only, call-scoped view of one fully published population state.
 *
 * The view cannot be copied or moved. Sessions must not retain references
 * obtained from it after the observation call returns.
 */
template <class Descriptor>
class SingleLevelPublishedState {
public:
    SingleLevelPublishedState(
        int step,
        amrex::Real latticeTime,
        amrex::Real physicalTime,
        LatticeField<Descriptor> const& populations,
        SingleLevelRunContext<Descriptor> const& context,
        MacroField const* macro,
        boundary::InternalForceBatch const& forceBatch)
        : step_(step),
          latticeTime_(latticeTime),
          physicalTime_(physicalTime),
          populations_(populations),
          context_(context),
          macro_(macro),
          forceBatch_(forceBatch)
    {}

    SingleLevelPublishedState(SingleLevelPublishedState const&) = delete;
    SingleLevelPublishedState& operator=(SingleLevelPublishedState const&) = delete;
    SingleLevelPublishedState(SingleLevelPublishedState&&) = delete;
    SingleLevelPublishedState& operator=(SingleLevelPublishedState&&) = delete;

    int step() const noexcept { return step_; }
    amrex::Real latticeTime() const noexcept { return latticeTime_; }
    amrex::Real physicalTime() const noexcept { return physicalTime_; }
    LatticeField<Descriptor> const& populations() const noexcept
    {
        return populations_;
    }
    SingleLevelRunContext<Descriptor> const& context() const noexcept
    {
        return context_;
    }
    bool hasMacro() const noexcept { return macro_ != nullptr; }
    MacroField const& macro() const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            macro_ != nullptr,
            "Published-state macro was not requested by the observation plan");
        return *macro_;
    }
    boundary::InternalForceBatch const& forceBatch() const noexcept
    {
        return forceBatch_;
    }

private:
    int step_ = 0;
    amrex::Real latticeTime_ = 0.0;
    amrex::Real physicalTime_ = 0.0;
    LatticeField<Descriptor> const& populations_;
    SingleLevelRunContext<Descriptor> const& context_;
    MacroField const* macro_ = nullptr;
    boundary::InternalForceBatch const& forceBatch_;
};

struct SingleLevelCompletedObservationResult {
    SingleLevelRunSessionResult result;
    bool stopRequested = false;

    bool ok() const noexcept { return result.ok(); }
};

/** No-output session for direct programs that request only numerical advance. */
template <class Descriptor>
class NoCaseRunSession {
public:
    SingleLevelRunSessionResult start(
        SingleLevelRunContext<Descriptor> const&)
    {
        return {};
    }

    SingleLevelObservationPlan observationPlan() const noexcept { return {}; }

    SingleLevelRunSessionResult observeInitialized(
        SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    SingleLevelCompletedObservationResult observeCompleted(
        SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    SingleLevelRunSessionResult observeFinal(
        SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const&)
    {
        return {};
    }
};

} // namespace lbm::runtime

#endif
