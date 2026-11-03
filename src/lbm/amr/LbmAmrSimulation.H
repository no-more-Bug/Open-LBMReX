#ifndef LBM_AMR_LBMAMRSIMULATION_H
#define LBM_AMR_LBMAMRSIMULATION_H

#include "lbm/amr/StaticRefinement.H"
#include "lbm/amr/DynamicRefinement.H"
#include "lbm/amr/AmrBoundarySystem.H"
#include "lbm/amr/CoarseFineCoupling.H"
#include "lbm/amr/PreparedAmrEvolution.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/fields/MacroField.H"
#include "lbm/profiling/AmrTimestepProfiler.H"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

namespace lbm::amr {

struct AmrRunResult {
    std::string error;
    bool ok() const noexcept { return error.empty(); }
};

template<class Descriptor>
struct LbmAmrLevelView {
    amrex::Geometry const& geometry;
    amrex::BoxArray const& boxes;
    amrex::DistributionMapping const& mapping;
    LatticeField<Descriptor> const& populations;
    MacroField const* macro;
    int step;
    amrex::Real physicalTime;
};

/** Borrowed data, valid only during the observer call. */
template<class Descriptor>
struct LbmAmrView {
    std::vector<LbmAmrLevelView<Descriptor>> levels;
    std::vector<amrex::IntVect> refRatios;
    int step;
    amrex::Real physicalTime;
    boundary::InternalForceBatch const& forceBatch;
};

enum class FineSubstep { First, Second };

/** One completed fine step; only this level's valid result is borrowed. */
template<class Descriptor>
struct FineSubstepView {
    int level;
    LbmAmrLevelView<Descriptor> fine;
    int coarsestStep;
    FineSubstep substep;
};

template<class Descriptor, class Profile, class Tagging>
struct HierarchyPreparation {
    std::unique_ptr<LbmAmrSimulation<Descriptor, Profile, Tagging>> simulation;
    std::vector<std::string> errors;
    bool ok() const noexcept { return simulation != nullptr && errors.empty(); }
    std::unique_ptr<LbmAmrSimulation<Descriptor, Profile, Tagging>> take() { return std::move(simulation); }
};

template<class Descriptor, class Profile>
using StaticHierarchyPreparation = HierarchyPreparation<Descriptor, Profile, StaticTagging>;

template<class Descriptor, class Profile, class Tagging>
HierarchyPreparation<Descriptor, Profile, Tagging> prepare_dynamic_hierarchy(
    DynamicHierarchySpec const& spec, Profile profile, Tagging tagging,
    AmrBoundarySpec const& boundaries = {});

template<class Descriptor, class Profile>
StaticHierarchyPreparation<Descriptor, Profile> prepare_static_hierarchy(
    StaticHierarchySpec const& spec, Profile profile, AmrBoundarySpec const& boundaries = {});

/** Owns the mesh and fields; only a completed hierarchy may be observed. */
template<class Descriptor, class Profile, class Tagging>
class LbmAmrSimulation : private amrex::AmrCore {
    using Layout = std::conditional_t<std::is_same_v<Tagging, StaticTagging>,
                                      StaticHierarchyLayout, DynamicHierarchyLayout>;
public:
    LbmAmrSimulation(LbmAmrSimulation const&) = delete;
    LbmAmrSimulation& operator=(LbmAmrSimulation const&) = delete;
    ~LbmAmrSimulation() override;

    int numLevels() const noexcept { return finestLevel() + 1; }
    int numAllowedLevels() const noexcept { return maxLevel() + 1; }
    amrex::Geometry const& geometry(int lev) const { return Geom(lev); }
    boundary::ResolvedBoundarySet const& resolvedBoundaries() const noexcept { return boundaries_.resolved(); }
    int internalBoundaryLevel(boundary::BoundaryId id) const noexcept { return boundaries_.internalBoundaryLevel(id); }

    template<class Observer>
    void observe(bool withMacro, Observer&& observer);

    template<class Collision, class Observer, class FineObserver = std::nullptr_t>
    AmrRunResult evolve(PreparedAmrEvolution<Collision> const& evolution, int numSteps,
                        bool withMacro, Observer&& observer, FineObserver&& fineObserver = nullptr,
                        profiling::AmrTimestepProfiler* profiler = nullptr);

protected:
    LbmAmrSimulation(Layout layout, Profile profile, Tagging tagging = {});
    std::vector<std::string> initialize(AmrBoundarySpec const& boundaries);
    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;
    void MakeNewLevelFromScratch(int lev, amrex::Real time,
        amrex::BoxArray const& ba, amrex::DistributionMapping const& dm) override;
    void ClearLevel(int lev) override;
    void MakeNewLevelFromCoarse(int, amrex::Real, amrex::BoxArray const&,
        amrex::DistributionMapping const&) override;
    void RemakeLevel(int, amrex::Real, amrex::BoxArray const&,
        amrex::DistributionMapping const&) override;
    void regrid(int, amrex::Real, bool) override;
    // Install one synchronized candidate; L0 and the physical domain remain fixed.
    void installHierarchy(std::vector<amrex::BoxArray> const& candidate,
                          std::vector<amrex::DistributionMapping> const& mappings, amrex::Real time);

private:
    template<class D, class P> friend StaticHierarchyPreparation<D, P> prepare_static_hierarchy(
        StaticHierarchySpec const&, P, AmrBoundarySpec const&);
    template<class D, class P, class T> friend HierarchyPreparation<D, P, T> prepare_dynamic_hierarchy(
        DynamicHierarchySpec const&, P, T, AmrBoundarySpec const&);
    template<class D, class P, class T, class S>
    friend runtime::LbmAmrEvolutionPreparation<D, S> runtime::prepare_amr_evolution(
        ::lbm::amr::LbmAmrSimulation<D, P, T> const&, config::LbmConfigIntent const&, S const&);
    enum class StepPhase { Coarsest, First, Second };
    struct ObserverGuard {
        bool& active;
        explicit ObserverGuard(bool& flag) : active(flag) { active = true; }
        ~ObserverGuard() { active = false; }
    };
    struct RegridProfileGuard {
        profiling::AmrProfileStep const*& slot;
        RegridProfileGuard(profiling::AmrProfileStep const*& target,
                           profiling::AmrProfileStep const& timing) : slot(target) { slot = &timing; }
        ~RegridProfileGuard() { slot = nullptr; }
    };
    template<class Observer>
    void publish(bool withMacro, boundary::InternalForceBatch const& forceBatch, Observer&& observer);
    template<class Collision, class FineObserver>
    AmrRunResult advanceHierarchy(PreparedAmrEvolution<Collision> const& evolution,
                                 FineObserver& fineObserver, profiling::AmrProfileStep const& timing);
    template<class Collision, class FineObserver>
    void prepareLevelStep(int lev, StepPhase phase,
                          PreparedAmrEvolution<Collision> const& evolution,
                          FineObserver& fineObserver, profiling::AmrProfileStep const& timing);
    void commitLevelStep(int lev, amrex::Real dtPhys);
    template<class FineObserver>
    void observeFine(int lev, FineSubstep substep, amrex::Real dtPhys,
                     FineObserver& observer);
    template<class Collision>
    AmrRunResult synchronizeLevelTimes(amrex::Real begin,
                                      PreparedAmrEvolution<Collision> const& evolution);
    struct LevelFields {
        std::unique_ptr<LatticeField<Descriptor>> current, next;
        std::unique_ptr<MacroField> macro;
        int step = 0;
        amrex::Real physicalTime = 0;
    };
    std::unique_ptr<LevelFields> makeLevelFields(
        int lev, amrex::BoxArray const& ba, amrex::DistributionMapping const& dm);
    Layout layout_;
    Profile profile_;
    Tagging tagging_;
    // Current event only; collate stores full source-level requests on the IO rank.
    std::vector<amrex::Gpu::PinnedVector<amrex::IntVect>> eventDynamicTags_;
    AmrBoundarySpec boundarySpec_;
    std::vector<std::pair<boundary::BoundaryId, int>> initialBoundaryOwners_;
    AmrBoundarySystem<Descriptor> boundaries_;
    // Borrowed only while evolve calls the fixed AMReX regrid virtual interface.
    // The step/profiler remain caller-owned; no profiling state survives the scope.
    profiling::AmrProfileStep const* regridTiming_ = nullptr;
    std::vector<std::unique_ptr<LevelFields>> fields_;
    std::shared_ptr<void const> hierarchyIdentity_ = std::make_shared<char const>(0);
    std::shared_ptr<void const> boundEvolution_;
    std::vector<std::unique_ptr<CoarseFineCoupling<Descriptor>>> couplings_;
    enum class RunState { Uninitialized, Ready, Failed };
    RunState state_ = RunState::Uninitialized;
    bool observing_ = false;
};

} // namespace lbm::amr

#include "lbm/amr/LbmAmrSimulation.HH"
#endif
