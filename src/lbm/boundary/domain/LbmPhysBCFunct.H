#ifndef LBM_BOUNDARY_DOMAIN_LBMPHYSBCFUNCT_H
#define LBM_BOUNDARY_DOMAIN_LBMPHYSBCFUNCT_H

#include "lbm/boundary/core/BoundaryTypes.H"
#include "lbm/boundary/domain/BoundarySamplingPosition.H"
#include "lbm/boundary/domain/DomainBoundaryPrescription.H"
#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/schemes/HalfwayBounceBackScheme.H"

#include <AMReX_Array4.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>

namespace lbm::boundary {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool domain_contains(amrex::Box const& domain, int i, int j, int k) noexcept
{
    return i >= domain.smallEnd(0) && i <= domain.bigEnd(0) &&
           j >= domain.smallEnd(1) && j <= domain.bigEnd(1) &&
           k >= domain.smallEnd(2) && k <= domain.bigEnd(2);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
DomainSide infer_domain_side(amrex::IntVect const& ghost, amrex::Box const& domain) noexcept
{
    if (ghost[0] < domain.smallEnd(0)) {
        return DomainSide::XLo;
    }
    if (ghost[0] > domain.bigEnd(0)) {
        return DomainSide::XHi;
    }
    if (ghost[1] < domain.smallEnd(1)) {
        return DomainSide::YLo;
    }
    if (ghost[1] > domain.bigEnd(1)) {
        return DomainSide::YHi;
    }
    if (ghost[2] < domain.smallEnd(2)) {
        return DomainSide::ZLo;
    }
    return DomainSide::ZHi;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int collect_outside_domain_sides(
    amrex::IntVect const& ghost,
    amrex::Box const& domain,
    DomainSide* sides) noexcept
{
    int count = 0;
    if (ghost[0] < domain.smallEnd(0)) {
        sides[count++] = DomainSide::XLo;
    } else if (ghost[0] > domain.bigEnd(0)) {
        sides[count++] = DomainSide::XHi;
    }
    if (ghost[1] < domain.smallEnd(1)) {
        sides[count++] = DomainSide::YLo;
    } else if (ghost[1] > domain.bigEnd(1)) {
        sides[count++] = DomainSide::YHi;
    }
    if (ghost[2] < domain.smallEnd(2)) {
        sides[count++] = DomainSide::ZLo;
    } else if (ghost[2] > domain.bigEnd(2)) {
        sides[count++] = DomainSide::ZHi;
    }
    return count;
}

/**
 * @brief Choose the effective face for a physical edge/corner ghost.
 *
 * No-slip walls take precedence over non-wall faces. If no touched face is a
 * no-slip wall, the deterministic x/y/z collection order selects the first
 * face. The compiled post-stream plan shares the no-slip-over-non-wall
 * priority, but has its own tie-breaking and admissibility rules; this
 * PhysBC-specific order must remain stable for ghost-corner behavior.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
DomainSide infer_domain_side_wall_wins(
    amrex::IntVect const& ghost,
    amrex::Box const& domain,
    DomainFaceTableView const& faceTable) noexcept
{
    DomainSide sides[AMREX_SPACEDIM];
    const int count = collect_outside_domain_sides(ghost, domain, sides);
    for (int n = 0; n < count; ++n) {
        const int sideIndex = domain_side_index(sides[n]);
        if (faceTable.enabled[sideIndex] != 0 &&
            faceTable.physical[sideIndex] == BoundaryPhysical::NoSlipWall) {
            return sides[n];
        }
    }
    return count > 0 ? sides[0] : infer_domain_side(ghost, domain);
}

template <class Descriptor>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real local_population_density(
    amrex::Array4<amrex::Real> const& data,
    int i,
    int j,
    int k,
    int dcomp) noexcept
{
    amrex::Real rho = 0.0;
    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
        rho += data(i, j, k, dcomp + iPop);
    }
    return rho;
}

template <class Descriptor>
struct LbmDomainPopulationFill {
    DomainFaceTableView faceTable;

    /**
     * @brief Fill incoming pull-stream populations in one physical ghost cell.
     *
     * AMReX GpuBndryFuncFab calls this functor only for cells outside the
     * valid+periodic grown domain. For each population whose forward neighbor
     * is a valid fluid cell, the physical boundary is the intervening cell
     * face and the opposite post-collision fluid population supplies HBB.
     * Edge/corner ownership follows the fixed wall-wins rule. Components that
     * cannot be pulled by a valid neighbor are intentionally left untouched.
     *
     * The functor captures only DomainFaceTableView and is device-safe. For
     * moving/velocity boundaries it computes the physical halfway-link point
     * from GeometryData, then evaluates the selected profile at the supplied
     * physical time. Current built-ins are steady, but the time value is not
     * discarded by the execution path.
     */
    AMREX_GPU_DEVICE
    void operator() (
        amrex::IntVect const& ghost,
        amrex::Array4<amrex::Real> const& data,
        int dcomp,
        int numcomp,
        amrex::GeometryData const& geom,
        amrex::Real physicalTime,
        amrex::BCRec const*,
        int,
        int) const noexcept
    {
        amrex::ignore_unused(numcomp);
        const amrex::Box& domain = geom.Domain();

        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            const int fi = ghost[0] + Descriptor::cx(iPop);
            const int fj = ghost[1] + Descriptor::cy(iPop);
            const int fk = ghost[2] + Descriptor::cz(iPop);
            // A diagonal from an outer FAB ghost can point beyond this FAB.
            // Such a component has no valid pull consumer in this FAB.
            if (!domain_contains(domain, fi, fj, fk) ||
                !data.contains(fi, fj, fk)) {
                continue;
            }

            const DomainSide side = infer_domain_side_wall_wins(ghost, domain, faceTable);
            const int sideIndex = domain_side_index(side);
            if (faceTable.enabled[sideIndex] == 0) {
                continue;
            }
            if (faceTable.scheme[sideIndex] != BoundaryScheme::HalfwayBounceBack) {
                continue;
            }

            const int opp = Descriptor::opposite(iPop);
            amrex::Real value = data(fi, fj, fk, dcomp + opp);
            if (faceTable.physical[sideIndex] == BoundaryPhysical::MovingWall) {
                const auto position =
                    halfway_boundary_link_position<Descriptor>(
                        geom, fi, fj, fk, iPop);
                const amrex::GpuArray<amrex::Real, 3> boundaryULb =
                    UniformVelocityProfile{
                        faceTable.prescribedULb[sideIndex]}(
                            position, physicalTime);
                value += HalfwayBounceBackScheme<Descriptor>::moving_wall_correction(
                    iPop, boundaryULb);
            } else if (is_velocity_boundary(faceTable.physical[sideIndex])) {
                const auto position =
                    halfway_boundary_link_position<Descriptor>(
                        geom, fi, fj, fk, iPop);
                amrex::Real rho = local_population_density<Descriptor>(
                    data, fi, fj, fk, dcomp);
                amrex::GpuArray<amrex::Real, 3> boundaryULb =
                    UniformVelocityProfile{
                        faceTable.prescribedULb[sideIndex]}(
                            position, physicalTime);
                if (faceTable.profile[sideIndex] == BoundaryProfileKind::Poiseuille) {
                    rho = Descriptor::rho0_lb();
                    boundaryULb = make_poiseuille_velocity_profile(
                        faceTable,
                        sideIndex,
                        side,
                        faceTable.physical[sideIndex])(
                            position, physicalTime);
                }
                value += HalfwayBounceBackScheme<Descriptor>::velocity_correction(
                    iPop, rho, boundaryULb);
            }
            data(ghost[0], ghost[1], ghost[2], dcomp + iPop) = value;
        }
    }
};

} // namespace lbm::boundary

#endif
