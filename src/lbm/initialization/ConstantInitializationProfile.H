#ifndef LBM_INITIALIZATION_CONSTANTINITIALIZATIONPROFILE_H
#define LBM_INITIALIZATION_CONSTANTINITIALIZATIONPROFILE_H

#include "lbm/initialization/MacroscopicState.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::initialization {

/**
 * @brief Device-safe constant bulk-fluid initialization profile.
 *
 * The profile describes macroscopic values on every valid cell. It owns no
 * field storage and has no boundary semantics; population reconstruction and
 * ghost synchronization remain responsibilities of the level initialization
 * Operator.
 */
struct ConstantInitializationProfile {
    MacroscopicState state;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        return state;
    }
};

} // namespace lbm::initialization

#endif
