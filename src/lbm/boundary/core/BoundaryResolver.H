#ifndef LBM_BOUNDARY_CORE_BOUNDARYRESOLVER_H
#define LBM_BOUNDARY_CORE_BOUNDARYRESOLVER_H

#include "lbm/boundary/core/BoundaryResolveResult.H"
#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/boundary/domain/DomainBoundaryResolver.H"
#include "lbm/boundary/internal/InternalBoundaryResolver.H"

#include <AMReX_Geometry.H>

namespace lbm::boundary {

/** Resolve global geometry and identities without level-spacing constraints. */
template <class Descriptor>
BoundaryResolveResult resolve_boundary_definitions(
    std::vector<DomainBoundarySpec> const& domainSpecs,
    std::vector<InternalBoundarySpec> const& internalSpecs,
    amrex::Geometry const& geom);

/** Resolve typed domain and internal specifications as one coherent set. */
template <class Descriptor>
BoundaryResolveResult resolve_boundary_specs(
    std::vector<DomainBoundarySpec> const& domainSpecs,
    std::vector<InternalBoundarySpec> const& internalSpecs,
    amrex::Geometry const& geom);

} // namespace lbm::boundary

#include "lbm/boundary/core/BoundaryResolver.HH"

#endif
