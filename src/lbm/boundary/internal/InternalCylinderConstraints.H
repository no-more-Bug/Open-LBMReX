#ifndef LBM_BOUNDARY_INTERNAL_INTERNALCYLINDERCONSTRAINTS_H
#define LBM_BOUNDARY_INTERNAL_INTERNALCYLINDERCONSTRAINTS_H

#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Geometry.H>

#include <string>

namespace lbm::boundary {

bool below_internal_minimum_beyond_roundoff(
    amrex::Real actual,
    amrex::Real required,
    amrex::Real arithmeticScale) noexcept;

/** Check disjoint physical geometry, including nearest transverse periodic images. */
std::string cylinder_pair_constraint_failure(
    ResolvedInternalBoundary const& lhs,
    ResolvedInternalBoundary const& rhs,
    amrex::Geometry const& geom);

/** Check level spacing for owned IDs from a successful global definition. */
std::vector<std::string> validate_internal_boundaries_on_level(
    ResolvedBoundarySet const& resolved,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    amrex::Geometry const& geom);

} // namespace lbm::boundary

#endif
