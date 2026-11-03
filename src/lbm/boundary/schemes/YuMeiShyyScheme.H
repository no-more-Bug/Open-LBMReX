#ifndef LBM_BOUNDARY_SCHEMES_YUMEISHYYSCHEME_H
#define LBM_BOUNDARY_SCHEMES_YUMEISHYYSCHEME_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/** @brief Static no-slip Yu-Mei-Shyy reconstruction for one cut-link population. */
struct YuMeiShyyScheme {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real reconstruct(
        amrex::Real q,
        amrex::Real anchorForward,
        amrex::Real anchorOpposite,
        amrex::Real nextFluidForward) noexcept
    {
        return (q * anchorOpposite + q * anchorForward +
                (amrex::Real(1.0) - q) * nextFluidForward) /
            (amrex::Real(1.0) + q);
    }
};

} // namespace lbm::boundary

#endif
