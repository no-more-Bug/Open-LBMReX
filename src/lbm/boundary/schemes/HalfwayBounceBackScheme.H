#ifndef LBM_BOUNDARY_SCHEMES_HALFWAYBOUNCEBACKSCHEME_H
#define LBM_BOUNDARY_SCHEMES_HALFWAYBOUNCEBACKSCHEME_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/**
 * @brief Local population correction for halfway bounce-back boundaries.
 *
 * The physical boundary lies halfway between the valid fluid cell and the
 * physical ghost. LbmDomainPopulationFill supplies the opposite post-collision
 * fluid population and adds this correction for the prescribed boundary
 * velocity. The helper is stateless and device-safe.
 */
template <class Descriptor>
struct HalfwayBounceBackScheme {
    /** @brief Moving-wall correction using the Descriptor reference density. */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real moving_wall_correction(
        int iPop,
        amrex::GpuArray<amrex::Real, 3> const& wallULb) noexcept
    {
        return velocity_correction(iPop, Descriptor::rho0_lb(), wallULb);
    }

    /**
     * @brief Velocity correction for one incoming population and supplied density.
     *
     * @pre iPop is the population whose source lies in the physical ghost and
     *      wallULb is expressed in lattice units.
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real velocity_correction(
        int iPop,
        amrex::Real rhoLb,
        amrex::GpuArray<amrex::Real, 3> const& wallULb) noexcept
    {
        const amrex::Real cDotU =
            static_cast<amrex::Real>(Descriptor::cx(iPop)) * wallULb[0] +
            static_cast<amrex::Real>(Descriptor::cy(iPop)) * wallULb[1] +
            static_cast<amrex::Real>(Descriptor::cz(iPop)) * wallULb[2];
        return 2.0 * Descriptor::w(iPop) * rhoLb * Descriptor::invCs2() * cDotU;
    }
};

} // namespace lbm::boundary

#endif
