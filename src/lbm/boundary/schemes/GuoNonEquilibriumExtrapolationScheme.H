#ifndef LBM_BOUNDARY_SCHEMES_GUONONEQUILIBRIUMEXTRAPOLATIONSCHEME_H
#define LBM_BOUNDARY_SCHEMES_GUONONEQUILIBRIUMEXTRAPOLATIONSCHEME_H

#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/** @brief Macroscopic moments recovered from the first interior donor node. */
struct GuoDonorState {
    amrex::Real rho = 0.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

/**
 * @brief Full-population Guo non-equilibrium extrapolation on a valid boundary node.
 *
 * Donor and boundary are cell-centered nodes at the same population time layer.
 * For every iPop the scheme writes
 * f_b^eq(rho_b,u_b) + (f_1 - f_1^eq(rho_1,u_1)); there is no opposite-direction
 * mapping. Callers supply macroscopic boundary data according to the physical
 * condition and must provide a synchronized first-interior donor.
 */
template <class Descriptor>
struct GuoNonEquilibriumExtrapolationScheme {
    /** @brief Recover donor density and velocity once for a full-q reconstruction. */
    template <class DonorCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static GuoDonorState donor_state(DonorCell const& donor) noexcept
    {
        GuoDonorState state;
        state.rho = Moments<Descriptor>::rho(donor);
        Moments<Descriptor>::velocity(
            donor, state.rho, state.ux, state.uy, state.uz);
        return state;
    }

    /**
     * @brief Reconstruct all q boundary populations from a complete boundary state.
     *
     * Existing boundary values are disposable; every component is overwritten.
     */
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void reconstruct(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::Real rhoBoundary,
        amrex::GpuArray<amrex::Real, 3> const& velocityBoundary) noexcept
    {
        GuoDonorState const donorState = donor_state(donor);
        reconstruct_with_donor_state(
            donor, boundary, donorState, rhoBoundary, velocityBoundary);
    }

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void no_slip(
        DonorCell const& donor,
        BoundaryCell& boundary) noexcept
    {
        GuoDonorState const donorState = donor_state(donor);
        reconstruct_with_donor_state(
            donor,
            boundary,
            donorState,
            donorState.rho,
            amrex::GpuArray<amrex::Real, 3>{0.0, 0.0, 0.0});
    }

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void velocity(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, 3> const& velocityBoundary) noexcept
    {
        GuoDonorState const donorState = donor_state(donor);
        reconstruct_with_donor_state(
            donor, boundary, donorState, donorState.rho, velocityBoundary);
    }

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void moving_wall(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, 3> const& wallVelocity) noexcept
    {
        velocity(donor, boundary, wallVelocity);
    }

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void pressure(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::Real rhoBoundary) noexcept
    {
        GuoDonorState const donorState = donor_state(donor);
        reconstruct_with_donor_state(
            donor,
            boundary,
            donorState,
            rhoBoundary,
            amrex::GpuArray<amrex::Real, 3>{
                donorState.ux, donorState.uy, donorState.uz});
    }

    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void prescribed_state(
        DonorCell const& donor,
        BoundaryCell& boundary,
        amrex::Real rhoBoundary,
        amrex::GpuArray<amrex::Real, 3> const& velocityBoundary) noexcept
    {
        reconstruct(donor, boundary, rhoBoundary, velocityBoundary);
    }

    /**
     * @brief Copy all q donor populations to the boundary for exact-copy outflow.
     *
     * This is dispatched under the Guo post-stream execution path but does not
     * evaluate the non-equilibrium extrapolation formula.
     */
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void free_outflow_copy(
        DonorCell const& donor,
        BoundaryCell& boundary) noexcept
    {
        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            boundary[iPop] = donor[iPop];
        }
    }

private:
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void reconstruct_with_donor_state(
        DonorCell const& donor,
        BoundaryCell& boundary,
        GuoDonorState const& donorState,
        amrex::Real rhoBoundary,
        amrex::GpuArray<amrex::Real, 3> const& velocityBoundary) noexcept
    {
        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            const amrex::Real donorEquilibrium = Equilibrium<Descriptor>::compute(
                iPop,
                donorState.rho,
                donorState.ux,
                donorState.uy,
                donorState.uz);
            const amrex::Real boundaryEquilibrium = Equilibrium<Descriptor>::compute(
                iPop,
                rhoBoundary,
                velocityBoundary[0],
                velocityBoundary[1],
                velocityBoundary[2]);
            const amrex::Real donorNonEquilibrium =
                donor[iPop] - donorEquilibrium;
            boundary[iPop] = boundaryEquilibrium + donorNonEquilibrium;
        }
    }
};

} // namespace lbm::boundary

#endif
