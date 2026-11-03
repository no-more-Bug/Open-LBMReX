#ifndef LBM_AMR_LBMAMRSIMULATION_HH
#define LBM_AMR_LBMAMRSIMULATION_HH

#include "lbm/operators/level/EquilibriumPopulationInitializationOperator.H"
#include "lbm/operators/level/MacroscopicOperator.H"
#include "lbm/operators/level/PullStreamingOperator.H"
#include "lbm/solvers/BoundaryStepContext.H"
#include "lbm/amr/SynchronizedVelocity.H"
#include "lbm/amr/PopulationRemap.H"

#include <AMReX_Gpu.H>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <limits>
#include <utility>

namespace lbm::amr {
namespace detail {

// Bound positive, repeated Real additions before any population is modified.
// gamma_n = n*u/(1-n*u), u = epsilon/2. The conservative upper endpoint must
// still resolve dt strictly above half an ULP, including round-to-even ties.
// Bounds round outwards; no assumption that long double is wider than Real.
// Preparation and the caller supply finite begin >= 0, dt > 0 and count > 0.
template<class Real>
bool representable_evolution_time(Real begin, Real dt, int count)
{
    if (count == 1) return std::isfinite(begin + dt) && begin + dt > begin;
    auto const up = [](long double value) {
        return std::nextafter(value, std::numeric_limits<long double>::infinity());
    };
    long double const error = up(static_cast<long double>(count) *
                                std::numeric_limits<Real>::epsilon() / 2);
    if (error >= 1) return false;
    long double const denominator = std::nextafter(1 - error, 0.L);
    long double const upper = up(up(static_cast<long double>(begin) +
                                     up(static_cast<long double>(count) * dt)) / denominator);
    if (!std::isfinite(upper) || upper >= std::numeric_limits<Real>::max()) return false;
    Real const rounded = static_cast<Real>(upper);
    if (rounded >= std::numeric_limits<Real>::max()) return false;
    Real const infinity = std::numeric_limits<Real>::infinity();
    Real const high = std::nextafter(rounded, infinity);
    if (high >= std::numeric_limits<Real>::max()) return false;
    Real const gap = std::nextafter(high, infinity) - high;
    return dt > gap / 2;
}

// A local check for one coarse interval. Return the tolerance for failure
// reporting without storing a second clock or a persistent tolerance.
template<class Real>
std::pair<bool,Real> synchronized_evolution_time(
    Real begin, Real target, Real coarseDt, Real actual, Real levelDt, int substeps)
{
    Real const scale = std::max({std::abs(begin),std::abs(target),coarseDt});
    Real const tolerance = std::min(Real(4)*std::numeric_limits<Real>::epsilon()*
                                    (Real(substeps)+Real(1))*scale, levelDt/4);
    return {std::isfinite(actual) && std::isfinite(target) && actual > begin &&
            target > begin && std::abs(actual-target) <= tolerance, tolerance};
}

} // namespace detail

template<class Descriptor, class Profile, class Tagging>
LbmAmrSimulation<Descriptor, Profile, Tagging>::LbmAmrSimulation(Layout layout, Profile profile, Tagging tagging)
    : amrex::AmrCore(layout.geometry, layout.mesh), layout_(std::move(layout)),
      profile_(std::move(profile)), tagging_(std::move(tagging)), fields_(maxLevel() + 1)
{}

template<class Descriptor, class Profile, class Tagging>
LbmAmrSimulation<Descriptor, Profile, Tagging>::~LbmAmrSimulation()
{
    amrex::Gpu::streamSynchronize();
}

template<class Descriptor, class Profile, class Tagging>
std::vector<std::string> LbmAmrSimulation<Descriptor, Profile, Tagging>::initialize(AmrBoundarySpec const& boundaries)
{
    if (state_ != RunState::Uninitialized) return {"already_initialized: create a new simulation for a new run"};
    int passes = 0;
    if constexpr (std::is_same_v<Tagging, StaticTagging>) {
        for (;;) {
            InitFromScratch(0.0);
            ++passes;
            auto update = update_static_layout(layout_, *this);
            if (!update.errors.empty()) return std::move(update.errors);
            if (!update.rebuild) break;
        }
    } else {
        eventDynamicTags_.clear();
        if (layout_.constraints) eventDynamicTags_.resize(maxLevel());
        InitFromScratch(0.0);
        // InitFromScratch may revisit sources and subsequently remove a tail.
        // Keep only the last callback of sources still present in its raw result.
        if (layout_.constraints) eventDynamicTags_.resize(std::min(numLevels(), maxLevel()));
        std::vector<amrex::BoxArray> candidate;
        for (int lev = 0; lev < numLevels(); ++lev) candidate.push_back(boxArray(lev));
        auto final = finalize_dynamic_grids(std::move(candidate), layout_, eventDynamicTags_);
        eventDynamicTags_.clear();
        if (!final.ok()) return {final.error};
        candidate = std::move(final.grids);
        int const oldCount = numLevels();
        int const newCount = static_cast<int>(candidate.size());
        for (int lev = 1; lev < newCount; ++lev) {
            if (lev < oldCount && candidate[lev] == boxArray(lev)) continue;
            auto mapping = MakeDistributionMap(lev, candidate[lev]);
            MakeNewLevelFromScratch(lev, 0.0, candidate[lev], mapping);
            SetBoxArray(lev, candidate[lev]);
            SetDistributionMap(lev, mapping);
        }
        for (int lev = oldCount - 1; lev >= newCount; --lev) {
            ClearLevel(lev);
            ClearBoxArray(lev);
            ClearDistributionMap(lev);
        }
        finest_level = newCount - 1;
    }
    std::vector<std::string> errors;
    for (int lev = 0; lev < numLevels(); ++lev) {
        if (!fields_[lev] || fields_[lev]->current->multifab().boxArray() != boxArray(lev) ||
            fields_[lev]->current->multifab().DistributionMap() != DistributionMap(lev)) {
            errors.push_back("field_layout_mismatch: level " + std::to_string(lev));
        }
    }
    if (errors.empty()) {
        std::vector<BoundaryLevelLayout> levels;
        levels.reserve(numLevels());
        for (int lev = 0; lev < numLevels(); ++lev) {
            levels.push_back({Geom(lev), boxArray(lev), DistributionMap(lev), layout_.mesh.ref_ratio[lev]});
        }
        std::string boundaryError;
        if (!boundaries_.tryDefine(boundaries, levels, boundaryError)) {
            return {"boundary_preparation: " + boundaryError};
        }
        boundarySpec_ = boundaries;
        for (auto const& item : boundaries_.resolved().internal)
            initialBoundaryOwners_.emplace_back(
                item.boundaryId, boundaries_.internalBoundaryLevel(item.boundaryId));
        boundaries_.initializeValidBoundaries(*fields_[0]->current, 0.0);
        couplings_.reserve(numLevels()-1);
        for (int lev = 0; lev+1 < numLevels(); ++lev) {
            couplings_.push_back(std::make_unique<CoarseFineCoupling<Descriptor>>(
                Geom(lev), *fields_[lev]->current, Geom(lev+1), *fields_[lev+1]->current,
                refRatio(lev)));
        }
        state_ = RunState::Ready;
        if constexpr (std::is_same_v<Tagging, StaticTagging>)
            report_static_layout(layout_, *this, passes);
    }
    return errors;
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::MakeNewLevelFromScratch(
    int lev, amrex::Real time, amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_ == RunState::Uninitialized && time == 0.0,
        "scratch is only supported during t=0 initialization");
    auto data = makeLevelFields(lev,ba,dm);
    operators::EquilibriumPopulationInitializationOperator<Descriptor, Profile>
        initialization(profile_, Geom(lev));
    initialization.apply(*data->current, time);
    // A repeated scratch may replace storage without a preceding ClearLevel.
    amrex::Gpu::streamSynchronize();
    fields_[lev] = std::move(data);
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::ClearLevel(int lev)
{
    amrex::Gpu::streamSynchronize();
    fields_[lev].reset();
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::ErrorEst(
    int lev, amrex::TagBoxArray& tags, amrex::Real time, int)
{
    if constexpr (std::is_same_v<Tagging, StaticTagging>) {
        if (lev < static_cast<int>(layout_.tagCells.size()))
            tags.setVal(layout_.tagCells[lev], amrex::TagBox::SET);
    } else {
        std::unique_ptr<SynchronizedVelocity<Descriptor>> velocity;
        if constexpr (Tagging::needsVelocity) {
            std::vector<VelocitySourceLevel> sources;
            for (int level = 0; level < numLevels(); ++level)
                sources.push_back({Geom(level), fields_[level]->current->multifab()});
            velocity = std::make_unique<SynchronizedVelocity<Descriptor>>(sources, refRatio(0));
        }
        auto const& populations = fields_[lev]->current->multifab();
        auto covered = amrex::makeFineMask(populations.boxArray(), populations.DistributionMap(),
            lev+1 < numLevels() ? boxArray(lev+1) : amrex::BoxArray{}, refRatio(lev), 0, 1);
        std::unique_ptr<amrex::TagBoxArray> optionalTags;
        if (layout_.constraints) {
            optionalTags = std::make_unique<amrex::TagBoxArray>(
                populations.boxArray(), populations.DistributionMap(), 0);
            optionalTags->setVal(amrex::TagBox::CLEAR);
        }
        tag_valid_cells<Descriptor>(populations,velocity ? &velocity->level(lev) : nullptr,
            covered,Geom(lev),lev,time,tagging_.rule(lev),tags,
            layout_.constraints ? &*layout_.constraints : nullptr, optionalTags.get());
        if (optionalTags) optionalTags->collate(eventDynamicTags_[lev]);
        else amrex::Gpu::streamSynchronize();
    }
}

template<class Descriptor, class Profile, class Tagging>
std::unique_ptr<typename LbmAmrSimulation<Descriptor, Profile, Tagging>::LevelFields>
LbmAmrSimulation<Descriptor, Profile, Tagging>::makeLevelFields(
    int lev, amrex::BoxArray const& ba, amrex::DistributionMapping const& dm)
{
    auto data = std::make_unique<LevelFields>();
    auto const grow = lev == 0 ? amrex::IntVect(Descriptor::vicinity)
                               : stencil_extent<Descriptor>() * 2;
    data->current = std::make_unique<LatticeField<Descriptor>>(ba, dm, grow);
    data->next = std::make_unique<LatticeField<Descriptor>>(ba, dm, grow);
    return data;
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::MakeNewLevelFromCoarse(
    int lev, amrex::Real time, amrex::BoxArray const& ba, amrex::DistributionMapping const& dm)
{
    fields_[lev] = makeLevelFields(lev,ba,dm);
    fields_[lev]->step = fields_[0]->step*(1<<lev);
    fields_[lev]->physicalTime = time;
    // regrid fills valid populations immediately after this allocation.
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::RemakeLevel(
    int lev, amrex::Real time, amrex::BoxArray const& ba, amrex::DistributionMapping const& dm)
{
    MakeNewLevelFromCoarse(lev,time,ba,dm);
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::regrid(
    int lbase, amrex::Real time, bool)
{
    if constexpr (std::is_same_v<Tagging,StaticTagging>) {
        amrex::Abort("runtime regrid is unavailable for the static hierarchy");
    } else {
        AMREX_ALWAYS_ASSERT(lbase == 0);
        if (maxLevel() == 0) return;
        int const oldCount = numLevels();
        int traceStep = -1;
        if (auto const* requested = std::getenv("LBM_REGRID_TRACE_STEPS")) {
            std::istringstream steps(requested);
            int selected;
            while (steps >> selected)
                if (selected == fields_[0]->step) traceStep = selected;
        }
        auto trace = [&](auto const& grids, char const* stage) {
            if (traceStep < 0 || !amrex::ParallelDescriptor::IOProcessor()) return;
            for (int lev = 0; lev < static_cast<int>(grids.size()); ++lev) {
                amrex::Print() << "REGRID_STAGE " << traceStep << ' ' << stage << ' '
                               << lev << " -1 0 " << grids[lev].size() << '\n';
                for (int i = 0; i < grids[lev].size(); ++i) {
                    std::ostringstream line;
                    line << "REGRID_BOX " << traceStep << ' ' << stage << ' ' << lev;
                    for (int d = 0; d < 3; ++d)
                        line << ' ' << grids[lev][i].smallEnd(d) << ' ' << grids[lev][i].bigEnd(d);
                    amrex::Print() << line.str() << '\n';
                }
            }
        };
        if (traceStep >= 0) amrex::Gpu::streamSynchronize();
        double const begin = traceStep >= 0 ? amrex::second() : 0;
        int newFinest;
        eventDynamicTags_.clear();
        if (layout_.constraints) eventDynamicTags_.resize(maxLevel());
        amrex::Vector<amrex::BoxArray> generated(oldCount+1);
        MakeNewGrids(0,time,newFinest,generated);
        std::vector<amrex::BoxArray> candidate(generated.begin(),generated.begin()+newFinest+1);
        candidate[0] = boxArray(0);
        // Keep the raw candidate for the existing opt-in trace, outside timed regions.
        auto const initial = traceStep >= 0 ? candidate : std::vector<amrex::BoxArray>{};
        auto final = finalize_dynamic_grids(std::move(candidate),layout_,eventDynamicTags_);
        eventDynamicTags_.clear();
        if (!final.ok()) amrex::Abort(final.error);
        candidate = std::move(final.grids);
        newFinest = static_cast<int>(candidate.size()) - 1;
        if (traceStep >= 0) amrex::Gpu::streamSynchronize();
        double const clustered = traceStep >= 0 ? amrex::second() : 0;
        std::vector<amrex::DistributionMapping> mappings(newFinest+1);
        mappings[0] = DistributionMap(0);
        for (int lev = 1; lev <= newFinest; ++lev)
            mappings[lev] = lev < oldCount && candidate[lev] == boxArray(lev)
                ? DistributionMap(lev) : MakeDistributionMap(lev,candidate[lev]);
        installHierarchy(candidate,mappings,time);
        if (traceStep >= 0) {
            amrex::Gpu::streamSynchronize();
            amrex::Real seconds[]{clustered-begin, amrex::second()-clustered};
            amrex::ParallelDescriptor::ReduceRealMax(seconds,2);
            amrex::Print() << "REGRID_EVENT " << traceStep << ' ' << std::setprecision(17) << time << '\n'
                           << "REGRID_TIMING " << traceStep << ' ' << seconds[0] << ' ' << seconds[1] << '\n';
            trace(initial,"generated");
            trace(candidate,"final");
        }
    }
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::installHierarchy(
    std::vector<amrex::BoxArray> const& candidate,
    std::vector<amrex::DistributionMapping> const& mappings, amrex::Real time)
{
    int const oldCount = numLevels();
    int const newCount = static_cast<int>(candidate.size());
    std::vector<bool> changed(newCount,false);
    bool anyChange = oldCount != newCount;
    for (int lev = 1; lev < newCount; ++lev) {
        changed[lev] = lev >= oldCount || candidate[lev] != boxArray(lev)
            || mappings[lev] != DistributionMap(lev);
        anyChange = anyChange || changed[lev];
    }
    if (!anyChange) return;

    amrex::Gpu::streamSynchronize();
    auto oldFields = std::move(fields_);
    fields_.resize(numAllowedLevels());
    for (int lev = 0; lev < oldCount; ++lev) {
        if (lev >= newCount || changed[lev]) {
            oldFields[lev]->next.reset();
            oldFields[lev]->macro.reset();
        }
    }
    std::vector<amrex::MultiFab const*> sources;
    for (int lev = 0; lev < oldCount; ++lev)
        sources.push_back(&oldFields[lev]->current->multifab());
    PopulationRemap<Descriptor> transfer(sources,refRatio(0));
    for (int lev = 0; lev < newCount; ++lev) {
        auto const nextBoxes = lev+1 < newCount ? candidate[lev+1] : amrex::BoxArray{};
        if (!changed[lev]) {
            fields_[lev] = std::move(oldFields[lev]);
            transfer.backfillUncovered(lev,fields_[lev]->current->multifab(),nextBoxes);
        } else {
            if (lev < oldCount) RemakeLevel(lev,time,candidate[lev],mappings[lev]);
            else MakeNewLevelFromCoarse(lev,time,candidate[lev],mappings[lev]);
            transfer.fillNewLevel(lev,fields_[lev]->current->multifab(),
                &fields_[lev-1]->current->multifab(),nextBoxes);
            SetBoxArray(lev,candidate[lev]);
            SetDistributionMap(lev,mappings[lev]);
        }
    }
    for (int lev = newCount; lev < oldCount; ++lev) {
        fields_[lev] = std::move(oldFields[lev]);
        ClearLevel(lev);
        ClearBoxArray(lev);
        ClearDistributionMap(lev);
    }
    finest_level = newCount-1;

    couplings_.resize(newCount-1);
    for (int lev = 0; lev+1 < newCount; ++lev) {
        if (!couplings_[lev] || changed[lev] || changed[lev+1])
            couplings_[lev] = std::make_unique<CoarseFineCoupling<Descriptor>>(
                Geom(lev),*fields_[lev]->current,Geom(lev+1),*fields_[lev+1]->current,refRatio(lev));
    }
    std::vector<BoundaryLevelLayout> levels;
    for (int lev = 0; lev < newCount; ++lev)
        levels.push_back({Geom(lev),boxArray(lev),DistributionMap(lev),refRatio(lev)});
    auto rebuild = [&] {
        std::string boundaryError;
        if (!boundaries_.tryDefine(boundarySpec_,levels,boundaryError))
            amrex::Abort("dynamic boundary preparation: "+boundaryError);
        for (auto const& [id, initialLevel] : initialBoundaryOwners_) {
            if (boundaries_.internalBoundaryLevel(id) != initialLevel)
                amrex::Abort("dynamic boundary owner changed: BoundaryId="+std::to_string(id.value));
        }
    };
    if (regridTiming_ && regridTiming_->stageBreakdownActive()) {
        // Finish preceding migration/coupling work outside this nested timer.
        amrex::Gpu::streamSynchronize();
        regridTiming_->measureBoundaryPreparation(rebuild);
    } else {
        rebuild();
        amrex::Gpu::streamSynchronize();
    }
}

template<class Descriptor, class Profile, class Tagging>
template<class Observer>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::observe(bool withMacro, Observer&& observer)
{
    boundary::InternalForceBatch const empty{};
    publish(withMacro, empty, std::forward<Observer>(observer));
}

template<class Descriptor, class Profile, class Tagging>
template<class Observer>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::publish(
    bool withMacro, boundary::InternalForceBatch const& forceBatch, Observer&& observer)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state_ == RunState::Ready && !observing_,
        "observe requires a prepared simulation and cannot be reentered");
    LbmAmrView<Descriptor> view{{}, {}, fields_[0]->step, fields_[0]->physicalTime, forceBatch};
    view.levels.reserve(numLevels());
    for (int lev = 0; lev+1 < numLevels(); ++lev) view.refRatios.push_back(refRatio(lev));
    for (int lev = 0; lev < numLevels(); ++lev) {
        auto& data = *fields_[lev];
        if (withMacro) {
            if (!data.macro) {
                data.macro = std::make_unique<MacroField>("macroscopic", boxArray(lev), DistributionMap(lev));
            }
            operators::MacroscopicOperator<Descriptor>{}.apply(*data.current, *data.macro);
        }
        view.levels.push_back({Geom(lev), boxArray(lev), DistributionMap(lev),
            *data.current, withMacro ? data.macro.get() : nullptr, data.step, data.physicalTime});
    }
    ObserverGuard guard(observing_);
    std::forward<Observer>(observer)(view);
}

template<class Descriptor, class Profile, class Tagging>
template<class Collision, class Observer, class FineObserver>
AmrRunResult LbmAmrSimulation<Descriptor, Profile, Tagging>::evolve(
    PreparedAmrEvolution<Collision> const& evolution,
    int numSteps, bool withMacro, Observer&& observer, FineObserver&& fineObserver,
    profiling::AmrTimestepProfiler* profiler)
{
    if (state_ != RunState::Ready || observing_) return {"invalid_run_state: simulation not ready or observer active"};
    if (numSteps < 0) return {"negative_steps: numSteps must be nonnegative"};
    if (numSteps == 0) return {};
    if (profiler && profiler->numLevels() != numAllowedLevels()) {
        return {"profiling_hierarchy_mismatch: context must match the hierarchy size"};
    }
    if (!evolution.identity_ || evolution.hierarchy_ != hierarchyIdentity_ ||
        evolution.numLevels() != numAllowedLevels()) {
        return {"hierarchy_mismatch: evolution was not prepared for this hierarchy"};
    }
    if (boundEvolution_ && boundEvolution_ != evolution.identity_) {
        return {"evolution_identity: continuation requires the same prepared evolution"};
    }
    if (numSteps > std::numeric_limits<int>::max() - fields_[0]->step) {
        return {"step_range: target coarsest step exceeds the integer range"};
    }
    int const targetStep = fields_[0]->step + numSteps;
    for (int lev = 0; lev < numAllowedLevels(); ++lev) {
        if (lev >= std::numeric_limits<int>::digits ||
            targetStep > std::numeric_limits<int>::max() / (1 << lev)) {
            return {"step_range: target level step exceeds the integer range"};
        }
        auto const dt = evolution.level(lev).dtPhys;
        if (!detail::representable_evolution_time(fields_[0]->physicalTime, dt, numSteps * (1 << lev))) {
            return {"time_range: repeated level steps exceed the representable time range"};
        }
    }
    boundEvolution_ = evolution.identity_;
    for (int i = 0; i < numSteps; ++i) {
        auto const timing = profiler ? profiler->beginStep(fields_[0]->step+1) : profiling::AmrProfileStep{};
        const auto beginPhysicalTime = fields_[0]->physicalTime;
        try {
            auto result = advanceHierarchy(evolution,fineObserver,timing);
            if (!result.ok()) {
                state_ = RunState::Failed;
                return result;
            }
            timing.measureHierarchy([&] {
                // Own the completed interval before any regrid destroys its collectors.
                auto forceBatch = boundaries_.takeGlobalInternalForceBatch(
                    evolution, fields_[0]->step, beginPhysicalTime, fields_[0]->physicalTime);
                if constexpr (!std::is_same_v<Tagging,StaticTagging>) {
                    if (layout_.regridInterval > 0 && fields_[0]->step%layout_.regridInterval == 0) {
                        RegridProfileGuard guard(regridTiming_, timing);
                        regrid(0,fields_[0]->physicalTime,false);
                    }
                }
                publish(withMacro, forceBatch, observer);
            });
            if (profiler) profiler->completeStep(timing);
        } catch (...) {
            // A partial step, preparation or publication cannot be resumed or observed.
            state_ = RunState::Failed;
            throw;
        }
    }
    return {};
}

template<class Descriptor, class Profile, class Tagging>
template<class Collision, class FineObserver>
AmrRunResult LbmAmrSimulation<Descriptor, Profile, Tagging>::advanceHierarchy(
    PreparedAmrEvolution<Collision> const& evolution, FineObserver& fineObserver,
    profiling::AmrProfileStep const& timing)
{
    auto const begin = fields_[0]->physicalTime;
    prepareLevelStep(0,StepPhase::Coarsest,evolution,fineObserver,timing);
    timing.measure(0,profiling::AmrLevelStage::Management,[&] {
        commitLevelStep(0,evolution.level(0).dtPhys);
    });
    AmrRunResult result;
    timing.measureHierarchy([&] {
        result = synchronizeLevelTimes(begin,evolution);
    });
    return result;
}

template<class Descriptor, class Profile, class Tagging>
template<class Collision, class FineObserver>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::prepareLevelStep(
    int lev, StepPhase phase, PreparedAmrEvolution<Collision> const& evolution,
    FineObserver& fineObserver, profiling::AmrProfileStep const& timing)
{
    using Stage = profiling::AmrLevelStage;
    using PairStage = profiling::AmrPairStage;
    timing.advanceLevel(lev);
    auto& level = *fields_[lev];
    auto const v = stencil_extent<Descriptor>();
    solvers::BoundaryStepContext context(level.physicalTime, evolution.level(lev).dtPhys);
    timing.measure(lev,Stage::Collision,[&] { evolution.level(lev).collision.apply(*level.current); });
    timing.measure(lev,Stage::Communication,[&] {
        if (phase == StepPhase::Coarsest) level.current->fillBoundary(Geom(lev));
        else level.current->fillBoundary(Geom(lev), phase == StepPhase::First ? 2*v : v);
    });
    if (phase == StepPhase::First) {
        CoarseFineTimingSample pairTiming;
        couplings_[lev-1]->explode(*fields_[lev-1]->current,*level.current,
                                  timing.stageBreakdownActive() ? &pairTiming : nullptr);
        if (pairTiming.performed) {
            timing.addPair(lev-1,PairStage::CoarseToFineCommunication,pairTiming.communication);
            timing.addPair(lev-1,PairStage::Explode,pairTiming.computation);
        }
    }
    if (lev == 0) {
        timing.measure(lev,Stage::BoundaryPre,[&] {
            context.runPreStreaming([&](amrex::Real physicalTime) {
                boundaries_.fillPhysicalGhosts(*level.current, physicalTime);
            });
        });
    }
    timing.measure(lev,Stage::Streaming,[&] {
        operators::PullStreamingOperator<Descriptor>{}.apply(
            *level.current,*level.next,phase == StepPhase::First ? v : amrex::IntVect(0));
    });
    timing.measure(lev,Stage::BoundaryPost,[&] {
        context.runPostStreaming([&](amrex::Real physicalTime) {
            boundaries_.applyPostStreamingValidNodes(lev, *level.current, *level.next, physicalTime);
        });
    });
    if (lev+1 < numLevels()) {
        int const finer = lev+1;
        auto const dt = evolution.level(finer).dtPhys;
        prepareLevelStep(finer,StepPhase::First,evolution,fineObserver,timing);
        timing.measure(finer,Stage::Management,[&] {
            commitLevelStep(finer,dt);
            observeFine(finer,FineSubstep::First,dt,fineObserver);
        });
    
        prepareLevelStep(finer,StepPhase::Second,evolution,fineObserver,timing);
        if constexpr (!std::is_same_v<std::decay_t<FineObserver>,std::nullptr_t>) {
            timing.measure(finer,Stage::Management,[&] { observeFine(finer,FineSubstep::Second,dt,fineObserver); });
        }
        CoarseFineTimingSample pairTiming;
        couplings_[lev]->coalesce(*fields_[finer]->current,*level.next,
                                 timing.stageBreakdownActive() ? &pairTiming : nullptr);
        if (pairTiming.performed) {
            timing.addPair(lev,PairStage::FineToCoarseCommunication,pairTiming.communication);
            timing.addPair(lev,PairStage::Coalesce,pairTiming.computation);
        }
        timing.measure(finer,Stage::Management,[&] { commitLevelStep(finer,dt); });
        // This level's next is complete; retain current until its caller commits.
    }
    timing.measure(lev,Stage::Management,[&] {
        boundaries_.recordInternalForceSample(
            {lev, level.step+1, fields_[0]->step+1, level.physicalTime,
             level.physicalTime+evolution.level(lev).dtPhys}, *level.current, *level.next);
    });
}

template<class Descriptor, class Profile, class Tagging>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::commitLevelStep(int lev, amrex::Real dtPhys)
{
    auto& level = *fields_[lev];
    std::swap(level.current,level.next);
    ++level.step;
    level.physicalTime += dtPhys;
}

template<class Descriptor, class Profile, class Tagging>
template<class FineObserver>
void LbmAmrSimulation<Descriptor, Profile, Tagging>::observeFine(
    int lev, FineSubstep substep, amrex::Real dtPhys, FineObserver& observer)
{
    if constexpr (!std::is_same_v<std::decay_t<FineObserver>,std::nullptr_t>) {
        auto const& level = *fields_[lev];
        bool const pending = substep == FineSubstep::Second;
        FineSubstepView<Descriptor> view{lev,
            {Geom(lev),boxArray(lev),DistributionMap(lev),
             pending ? *level.next : *level.current,nullptr,
             level.step+int(pending),level.physicalTime+(pending ? dtPhys : 0)},
            fields_[0]->step+1,substep};
        ObserverGuard guard(observing_);
        observer(view);
    }
}

template<class Descriptor, class Profile, class Tagging>
template<class Collision>
AmrRunResult LbmAmrSimulation<Descriptor, Profile, Tagging>::synchronizeLevelTimes(
    amrex::Real begin, PreparedAmrEvolution<Collision> const& evolution)
{
    auto const& coarse = *fields_[0];
    for (int lev = 0; lev < numLevels(); ++lev) {
        auto const& level = *fields_[lev];
        int const substeps = 1 << lev;
        if (level.step != coarse.step*substeps)
            return {"level_step_mismatch: level "+std::to_string(lev)+" step="+
                    std::to_string(level.step)+" expected="+std::to_string(coarse.step*substeps)};
        auto const [aligned,tolerance] = detail::synchronized_evolution_time(
            begin,coarse.physicalTime,evolution.level(0).dtPhys,
            level.physicalTime,evolution.level(lev).dtPhys,substeps);
        if (!aligned) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10)
                    << "level_time_mismatch: level " << lev << " actual=" << level.physicalTime
                    << " expected=" << coarse.physicalTime << " tolerance=" << tolerance;
            return {message.str()};
        }
    }
    for (int lev = 0; lev < numLevels(); ++lev)
        fields_[lev]->physicalTime = coarse.physicalTime;
    return {};
}

template<class Descriptor, class Profile>
StaticHierarchyPreparation<Descriptor, Profile> prepare_static_hierarchy(
    StaticHierarchySpec const& spec, Profile profile, AmrBoundarySpec const& boundaries)
{
    StaticHierarchyPreparation<Descriptor, Profile> result;
    auto layout = prepare_static_layout(spec, Descriptor::d);
    if (!layout.ok()) { result.errors = std::move(layout.errors); return result; }
    auto simulation = std::unique_ptr<LbmAmrSimulation<Descriptor, Profile>>(
        new LbmAmrSimulation<Descriptor, Profile>(std::move(*layout.layout), std::move(profile)));
    result.errors = simulation->initialize(boundaries);
    if (result.errors.empty()) result.simulation = std::move(simulation);
    return result;
}

template<class Descriptor, class Profile, class Tagging>
HierarchyPreparation<Descriptor, Profile, Tagging> prepare_dynamic_hierarchy(
    DynamicHierarchySpec const& spec, Profile profile, Tagging tagging,
    AmrBoundarySpec const& boundaries)
{
    HierarchyPreparation<Descriptor, Profile, Tagging> result;
    auto layout = prepare_dynamic_layout(spec, Descriptor::d);
    if (!layout.ok()) { result.errors = std::move(layout.errors); return result; }
    result.errors = tagging.validate(spec.maxLevel);
    if (!result.errors.empty()) return result;
    auto simulation = std::unique_ptr<LbmAmrSimulation<Descriptor, Profile, Tagging>>(
        new LbmAmrSimulation<Descriptor, Profile, Tagging>(
            std::move(*layout.layout), std::move(profile), std::move(tagging)));
    result.errors = simulation->initialize(boundaries);
    if (result.errors.empty()) result.simulation = std::move(simulation);
    return result;
}

} // namespace lbm::amr
#endif
