#ifndef LBM_BOUNDARY_DOMAIN_BOUNDARYSAMPLINGPOSITION_H
#define LBM_BOUNDARY_DOMAIN_BOUNDARYSAMPLINGPOSITION_H

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/**
 * @brief Return the AMReX physical center of one cell-centered valid node.
 *
 * AMReX's Cartesian CoordSys treats ProbLo as the physical offset of index
 * zero, so the cell center is `ProbLo + (index + 1/2) CellSize` even when the
 * Geometry domain has a non-zero small end. The returned value is device-safe;
 * it carries no boundary scheme or profile interpretation.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> boundary_cell_center_position(
    amrex::GeometryData const& geom,
    int i,
    int j,
    int k) noexcept
{
    const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{};
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        position[axis] =
            geom.ProbLo(axis) +
            (static_cast<amrex::Real>(index[axis]) + 0.5) *
                geom.CellSize(axis);
    }
    return position;
}

/**
 * @brief Return the physical midpoint of a fluid-to-ghost population link.
 *
 * The supplied indices identify the valid fluid cell reached from a physical
 * ghost during pull streaming. Subtracting half of the population velocity
 * from that cell center locates the HBB wall/link intersection. For diagonal
 * populations, tangential coordinates are shifted along the same link; this
 * is the existing halfway-bounce-back geometry rather than a projection onto
 * an independently chosen face point.
 *
 * @pre Descriptor::c(iPop,axis) is defined for every AMREX_SPACEDIM axis and
 *      the link from the fluid cell to its source ghost crosses a physical
 *      domain face.
 */
template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
halfway_boundary_link_position(
    amrex::GeometryData const& geom,
    int fluidI,
    int fluidJ,
    int fluidK,
    int iPop) noexcept
{
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position =
        boundary_cell_center_position(geom, fluidI, fluidJ, fluidK);
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        position[axis] -=
            0.5 * static_cast<amrex::Real>(Descriptor::c(iPop, axis)) *
            geom.CellSize(axis);
    }
    return position;
}

} // namespace lbm::boundary

#endif
