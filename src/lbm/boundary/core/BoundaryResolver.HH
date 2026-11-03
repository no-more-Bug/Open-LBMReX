#ifndef LBM_BOUNDARY_CORE_BOUNDARYRESOLVER_HH
#define LBM_BOUNDARY_CORE_BOUNDARYRESOLVER_HH

#include <set>
#include <string>
#include <utility>

namespace lbm::boundary {

template <class Descriptor>
BoundaryResolveResult resolve_boundary_definitions(
    std::vector<DomainBoundarySpec> const& domainSpecs,
    std::vector<InternalBoundarySpec> const& internalSpecs,
    amrex::Geometry const& geom)
{
    BoundaryResolveResult result =
        resolve_domain_boundary_specs<Descriptor>(domainSpecs, geom);

    std::set<std::string> domainNames;
    for (DomainBoundarySpec const& domain : domainSpecs) {
        if (!domain.name.empty()) {
            domainNames.insert(domain.name);
        }
    }
    InternalBoundaryResolveResult internal =
        resolve_internal_boundary_definitions<Descriptor>(
            internalSpecs,
            geom,
            static_cast<int>(result.resolved.domain.size()),
            domainNames,
            domainSpecs);
    result.resolved.internal = std::move(internal.boundaries);
    result.resolved.internalObjects = std::move(internal.objects);
    result.resolved.internalObjectBoundaries =
        std::move(internal.objectBoundaries);
    result.errors.insert(
        result.errors.end(), internal.errors.begin(), internal.errors.end());
    return result;
}

template <class Descriptor>
BoundaryResolveResult resolve_boundary_specs(
    std::vector<DomainBoundarySpec> const& domainSpecs,
    std::vector<InternalBoundarySpec> const& internalSpecs,
    amrex::Geometry const& geom)
{
    auto result = resolve_boundary_definitions<Descriptor>(domainSpecs, internalSpecs, geom);
    if (result.ok()) {
        std::vector<BoundaryId> ownedBoundaryIds;
        for (auto const& boundary : result.resolved.internal) {
            ownedBoundaryIds.push_back(boundary.boundaryId);
        }
        auto errors = validate_internal_boundaries_on_level(result.resolved, ownedBoundaryIds, geom);
        result.errors.insert(result.errors.end(), errors.begin(), errors.end());
    }
    return result;
}

} // namespace lbm::boundary

#endif
