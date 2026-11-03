#ifndef LBM_BOUNDARY_SCHEMES_BOUZIDIINTERPOLATEDBOUNCEBACKSCHEME_H
#define LBM_BOUNDARY_SCHEMES_BOUZIDIINTERPOLATEDBOUNCEBACKSCHEME_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/** @brief Static no-slip Bouzidi reconstruction for one cut-link population. */
struct BouzidiInterpolatedBounceBackScheme {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real reconstruct(
        amrex::Real q,
        amrex::Real anchorForward,
        amrex::Real anchorOpposite,
        amrex::Real nextFluidForward) noexcept
    {
        if (q < amrex::Real(0.5)) {
            return amrex::Real(2.0) * q * anchorForward +
                (amrex::Real(1.0) - amrex::Real(2.0) * q) *
                    nextFluidForward;
        }
        const amrex::Real twoQ = amrex::Real(2.0) * q;
        return anchorForward / twoQ +
            (twoQ - amrex::Real(1.0)) * anchorOpposite / twoQ;
    }
};

} // namespace lbm::boundary

#endif
