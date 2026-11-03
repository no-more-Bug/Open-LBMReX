#ifndef LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYRESOLVER_H
#define LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYRESOLVER_H

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Geometry.H>

#include <set>
#include <string>
#include <vector>

namespace lbm::boundary {

/** Flat internal resolve result, independent of the domain resolver. */
struct InternalBoundaryResolveResult {
    std::vector<ResolvedInternalBoundary> boundaries;
    std::vector<ResolvedInternalObject> objects;
    std::vector<BoundaryId> objectBoundaries;
    std::vector<std::string> errors;
};

/** Resolve mutually disjoint static analytic-cylinder inputs. */
template <class Descriptor>
InternalBoundaryResolveResult resolve_internal_boundary_definitions(
    std::vector<InternalBoundarySpec> const& specs,
    amrex::Geometry const& geom,
    int boundaryIdOffset,
    std::set<std::string> const& reservedBoundaryNames,
    std::vector<DomainBoundarySpec> const& domainSpecs);

} // namespace lbm::boundary

#include "lbm/boundary/internal/InternalBoundaryResolver.HH"

#endif
