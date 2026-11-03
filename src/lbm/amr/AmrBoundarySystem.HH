#ifndef LBM_AMR_AMRBOUNDARYSYSTEM_HH
#define LBM_AMR_AMRBOUNDARYSYSTEM_HH

#include "lbm/amr/StaticRegionTopology.H"
#include "lbm/boundary/metadata/InternalCutLinkCompiler.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <utility>

namespace lbm::amr {

template <class Descriptor>
bool AmrBoundarySystem<Descriptor>::tryDefine(
    AmrBoundarySpec const& spec,
    std::vector<BoundaryLevelLayout> const& levels,
    std::string& error)
{
    internal_.clear();
    internalLevels_.clear();
    resolved_ = {};
    domain_ = boundary::DomainBoundarySystem<Descriptor>{};
    collectForces_ = spec.collectForces;
    error.clear();
    if (levels.empty()) {
        error = "AMR boundary preparation requires at least L0";
        return false;
    }
    auto definitions = boundary::resolve_boundary_definitions<Descriptor>(
        spec.domain, spec.internal, levels.front().geometry);
    if (!definitions.ok()) {
        error = definitions.error_message();
        return false;
    }
    resolved_ = std::move(definitions.resolved);
    if (!domain_.tryDefine(levels.front().geometry, resolved_.domain, error)) {
        error = "level=0 " + error;
        return false;
    }
    internal_.resize(levels.size());
    if (resolved_.internal.empty()) {
        return true;
    }
    auto query = boundary::build_internal_geometry_query(resolved_, levels.front().geometry);
    if (!query.ok()) {
        error = query.errorMessage();
        return false;
    }
    const int levelCount = static_cast<int>(levels.size());
    const int boundaryCount = static_cast<int>(resolved_.internal.size());
    std::vector<int> seen(static_cast<std::size_t>(levelCount * boundaryCount), 0);
    std::vector<std::vector<boundary::CanonicalCutLinkRecord>> candidates(levels.size());
    std::vector<boundary::CenterPhaseBuildReport> reports(levels.size());
    for (int level = 0; level < levelCount; ++level) {
        auto const& layout = levels[level];
        const amrex::BoxArray covered = level + 1 < levelCount ?
            detail::coarse_cells(levels[level + 1].boxes, layout.ratioToFiner) : amrex::BoxArray{};
        {
            boundary::CenterPhaseField<Descriptor> phase(layout.boxes, layout.mapping);
            boundary::InternalCutLinkCompiler<Descriptor> compiler;
            if (!phase.build(query.query, layout.geometry, error) ||
                !compiler.build(query.query, phase, layout.geometry, covered, error)) {
                error = "level " + std::to_string(level) + ": " + error;
                return false;
            }
            candidates[level] = compiler.records();
            reports[level] = phase.report();
        }
        // No phase/provenance fields survive into the next level.
        for (auto const& link : candidates[level]) {
            const auto found = std::find_if(resolved_.internal.begin(), resolved_.internal.end(),
                [&](auto const& boundary) { return boundary.boundaryId.value == link.boundaryId; });
            const auto index = found - resolved_.internal.begin();
            seen[index * levelCount + level] = 1;
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(seen.data(), static_cast<int>(seen.size()));
    internalLevels_.assign(resolved_.internal.size(), -1);
    for (int index = 0; index < boundaryCount; ++index) {
        const auto id = resolved_.internal[index].boundaryId;
        for (int level = 0; level < levelCount; ++level) {
            if (seen[index * levelCount + level] == 0) {
                continue;
            }
            if (internalLevels_[index] >= 0) {
                error = "BoundaryId=" + std::to_string(id.value) + " has links on multiple effective levels";
                return false;
            }
            internalLevels_[index] = level;
        }
        if (internalLevels_[index] < 0) {
            error = "BoundaryId=" + std::to_string(id.value) + " produced no effective cut links";
            return false;
        }
    }
    for (int level = 0; level < levelCount; ++level) {
        std::vector<boundary::BoundaryId> ownedIds;
        boundary::ResolvedBoundarySet ownedGeometry;
        for (int index = 0; index < boundaryCount; ++index) {
            if (internalLevels_[index] == level) {
                ownedIds.push_back(resolved_.internal[index].boundaryId);
                ownedGeometry.internal.push_back(resolved_.internal[index]);
            }
        }
        if (ownedIds.empty()) {
            continue;
        }
        auto const& layout = levels[level];
        auto errors = boundary::validate_internal_boundaries_on_level(resolved_, ownedIds, layout.geometry);
        if (!errors.empty()) {
            error = "level " + std::to_string(level) + ": " + errors.front();
            return false;
        }
        auto ownedQuery = boundary::build_internal_geometry_query(ownedGeometry, layout.geometry);
        if (!ownedQuery.ok()) {
            error = ownedQuery.errorMessage();
            return false;
        }
        boundary::EbGeometryBackend backend;
        auto prepared = std::make_unique<boundary::InternalSolidBoundarySystem<Descriptor>>();
        if (!backend.define(ownedQuery.query, layout.geometry, layout.boxes, layout.mapping, error) ||
            !boundary::check_internal_links_against_eb(*backend.factory(), candidates[level], error) ||
            !prepared->definePrepared(resolved_, ownedIds, layout.geometry, layout.boxes,
                                      layout.mapping, candidates[level], reports[level], error)) {
            error = "level " + std::to_string(level) + ": " + error;
            return false;
        }
        internal_[level] = std::move(prepared);
        std::vector<boundary::CanonicalCutLinkRecord>{}.swap(candidates[level]);
        // backend destroys its factory and then its IndexSpace before next level.
    }
    return true;
}

template <class Descriptor>
int AmrBoundarySystem<Descriptor>::internalBoundaryLevel(boundary::BoundaryId id) const noexcept
{
    for (std::size_t index = 0; index < resolved_.internal.size(); ++index) {
        if (resolved_.internal[index].boundaryId.value == id.value) {
            return internalLevels_[index];
        }
    }
    return -1;
}

template <class Descriptor>
void AmrBoundarySystem<Descriptor>::initializeValidBoundaries(Field& coarsest, amrex::Real physicalTime) const
{
    domain_.initializeValidBoundaries(coarsest, physicalTime);
}

template <class Descriptor>
void AmrBoundarySystem<Descriptor>::fillPhysicalGhosts(Field& coarsest, amrex::Real physicalTime) const
{
    domain_.fillPhysicalGhosts(coarsest, physicalTime);
}

template <class Descriptor>
void AmrBoundarySystem<Descriptor>::applyPostStreamingValidNodes(
    int level, Field const& collidedSource, Field& destination, amrex::Real physicalTime) const
{
    if (level == 0) {
        domain_.applyPostStreamingValidNodes(destination, physicalTime);
    }
    if (internal_[level]) {
        internal_[level]->applyPostStreamingValidNodes(collidedSource, destination, physicalTime);
    }
}

template<class Descriptor>
void AmrBoundarySystem<Descriptor>::recordInternalForceSample(
    boundary::InternalForceStep const& step, Field const& collidedSource, Field const& completedPopulation)
{
    if (collectForces_ && internal_[step.level])
        internal_[step.level]->recordInternalForceSample(collidedSource, completedPopulation, step);
}

template<class Descriptor>
template<class Collision>
boundary::InternalForceBatch AmrBoundarySystem<Descriptor>::takeGlobalInternalForceBatch(
    PreparedAmrEvolution<Collision> const& evolution, int coarsestStep,
    amrex::Real beginPhysicalTime, amrex::Real endPhysicalTime)
{
    boundary::InternalForceBatch batch;
    if (!collectForces_) return batch;
    std::vector<std::vector<boundary::ObjectId>> levelObjects(internal_.size());
    std::vector<boundary::ObjectId> intervalObjects;
    for (auto const& object : resolved_.internalObjects) {
        int owner = internalBoundaryLevel(resolved_.internalObjectBoundaries[object.boundaryOffset]);
        for (int n = 1; n < object.boundaryCount; ++n) {
            if (internalBoundaryLevel(resolved_.internalObjectBoundaries[object.boundaryOffset+n]) != owner) {
                owner = -1;
                break;
            }
        }
        if (owner < 0) intervalObjects.push_back(object.objectId);
        else levelObjects[owner].push_back(object.objectId);
    }
    boundary::InternalObjectForceIntervalSample interval{coarsestStep, beginPhysicalTime, endPhysicalTime, {}};
    for (auto id : intervalObjects) interval.objectForces.push_back({id});
    for (std::size_t level = 0; level < internal_.size(); ++level) {
        if (!internal_[level]) continue;
        auto raw = internal_[level]->takeGlobalBoundaryForceBatch();
        const auto& parameters = evolution.level(static_cast<int>(level));
        amrex::Real scale = parameters.dxLb();
        if constexpr (Descriptor::d == 3) scale *= parameters.dxLb();
        const auto weight = parameters.dtPhys / evolution.level(0).dtPhys;
        for (auto& sample : raw) {
            for (auto& force : sample.boundaryForces)
                for (auto& component : force.forceLb) component *= scale;
            if (!levelObjects[level].empty()) {
                batch.objectSteps.push_back({sample.step,
                    boundary::sum_internal_object_forces(sample.boundaryForces, resolved_, levelObjects[level])});
            }
            auto contributions = boundary::sum_internal_object_forces(sample.boundaryForces, resolved_, intervalObjects);
            for (std::size_t object = 0; object < contributions.size(); ++object)
                for (int d = 0; d < 3; ++d)
                    interval.objectForces[object].forceLb[d] += weight * contributions[object].forceLb[d];
            batch.boundarySteps.push_back({sample.step, std::move(sample.boundaryForces)});
        }
    }
    if (!intervalObjects.empty() && !batch.boundarySteps.empty()) batch.objectIntervals.push_back(std::move(interval));
    return batch;
}

} // namespace lbm::amr
#endif
