#ifndef LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYRESOLVER_H
#define LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYRESOLVER_H

#include "lbm/boundary/core/BoundaryResolveResult.H"
#include "lbm/boundary/core/BoundarySpec.H"

#include <AMReX_Geometry.H>

#include <vector>

namespace lbm::boundary {

/** Resolve and validate all domain-face specifications for a descriptor. */
template <class Descriptor>
BoundaryResolveResult resolve_domain_boundary_specs(
    std::vector<DomainBoundarySpec> const& specs,
    amrex::Geometry const& geom);

} // namespace lbm::boundary

#include "lbm/boundary/domain/DomainBoundaryResolver.HH"

#endif
