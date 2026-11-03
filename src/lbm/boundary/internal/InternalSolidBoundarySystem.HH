#ifndef LBM_BOUNDARY_INTERNAL_INTERNALSOLIDBOUNDARYSYSTEM_HH
#define LBM_BOUNDARY_INTERNAL_INTERNALSOLIDBOUNDARYSYSTEM_HH

#include "lbm/boundary/metadata/InternalCutLinkCompiler.H"

namespace lbm::boundary {

template <class Descriptor>
bool InternalSolidBoundarySystem<Descriptor>::define(
    ResolvedBoundarySet const& resolved,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::string& error)
{
    reset();
    error.clear();
    hasInternal_ = !resolved.internal.empty();
    if (!hasInternal_) {
        ready_ = true;
        return true;
    }
    InternalGeometryQueryBuildResult queryBuild =
        build_internal_geometry_query(resolved, geom);
    if (!queryBuild.ok()) {
        error = queryBuild.errorMessage();
        return false;
    }
    std::vector<CanonicalCutLinkRecord> records;
    CenterPhaseBuildReport phaseReport;
    {
        CenterPhaseField<Descriptor> phase(ba, dm);
        InternalCutLinkCompiler<Descriptor> compiler;
        if (!phase.build(queryBuild.query, geom, error) ||
            !compiler.build(queryBuild.query, phase, geom, {}, error)) {
            return false;
        }
        phaseReport = phase.report();
        records = compiler.records();
    }
    EbGeometryBackend backend;
    if (!backend.define(queryBuild.query, geom, ba, dm, error) ||
        !check_internal_links_against_eb(*backend.factory(), records, error)) {
        return false;
    }
    std::vector<BoundaryId> ownedBoundaryIds;
    for (auto const& boundary : resolved.internal) {
        ownedBoundaryIds.push_back(boundary.boundaryId);
    }
    return definePrepared(resolved, ownedBoundaryIds, geom, ba, dm, records, phaseReport, error);
}

template <class Descriptor>
bool InternalSolidBoundarySystem<Descriptor>::definePrepared(
    ResolvedBoundarySet const& resolved,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::vector<CanonicalCutLinkRecord> const& records,
    CenterPhaseBuildReport const& phaseReport,
    std::string& error)
{
    reset();
    error.clear();
    hasInternal_ = !ownedBoundaryIds.empty();
    if (!hasInternal_) {
        ready_ = true;
        return true;
    }
    auto next = std::make_unique<State>();
    if (!next->storage.define(records, resolved, geom, ba, dm, error) ||
        !forceCollector_.define(next->storage, ba, dm, ownedBoundaryIds, error)) {
        return false;
    }
    next->diagnostics = build_internal_metadata_diagnostics(
        resolved, ownedBoundaryIds, phaseReport, next->storage.report());
    state_ = std::move(next);
    ready_ = true;
    return true;
}

template <class Descriptor>
void InternalSolidBoundarySystem<Descriptor>::reset() noexcept
{
    forceCollector_.reset();
    state_.reset();
    ready_ = false;
    hasInternal_ = false;
}

template <class Descriptor>
InternalBoundaryMetadataStorage const&
InternalSolidBoundarySystem<Descriptor>::storage() const
{
    requireState("storage");
    return state_->storage;
}

template <class Descriptor>
std::string InternalSolidBoundarySystem<Descriptor>::formatDiagnostics() const
{
    return ready_ && hasInternal_ && state_
        ? format_internal_metadata_diagnostics(state_->diagnostics)
        : std::string{};
}

template <class Descriptor>
void InternalSolidBoundarySystem<Descriptor>::requireState(
    char const* requested) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ready_ && hasInternal_ && state_ != nullptr,
        (std::string("Internal metadata requested ") + requested +
         " before setup")
            .c_str());
}

template <class Descriptor>
void InternalSolidBoundarySystem<Descriptor>::applyPostStreamingValidNodes(
    LatticeField<Descriptor> const& collidedSource,
    LatticeField<Descriptor>& streamedDestination,
    amrex::Real /* physicalTime */) const
{
    if (!hasInternal_) {
        return;
    }
    executor_.apply(state_->storage, collidedSource, streamedDestination);
}

template <class Descriptor>
void InternalSolidBoundarySystem<Descriptor>::recordInternalForceSample(
    LatticeField<Descriptor> const& collidedSource,
    LatticeField<Descriptor> const& completedPopulation,
    InternalForceStep const& step)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ready_,
        "InternalSolidBoundarySystem::record force called before setup");
    if (!hasInternal_) {
        return;
    }
    forceCollector_.recordInternalForceSample(
        collidedSource, completedPopulation, step);
}

template <class Descriptor>
std::vector<InternalBoundaryForceValues>
InternalSolidBoundarySystem<Descriptor>::takeGlobalBoundaryForceBatch()
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        ready_,
        "InternalSolidBoundarySystem::take force batch called before setup");
    return forceCollector_.takeGlobalBoundaryForceBatch();
}

} // namespace lbm::boundary

#endif
