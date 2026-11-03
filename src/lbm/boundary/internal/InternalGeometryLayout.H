#ifndef LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYLAYOUT_H
#define LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYLAYOUT_H

#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>

namespace lbm::boundary {

/** @brief A disjoint cell-centered layout inside the domain may be partial. */
inline bool is_canonical_internal_cell_layout(
    amrex::BoxArray const& ba,
    amrex::Geometry const& geom) noexcept
{
    return ba.ixType().cellCentered() && ba.size() > 0 && ba.isDisjoint() &&
        geom.Domain().contains(ba.minimalBox());
}

} // namespace lbm::boundary

#endif
