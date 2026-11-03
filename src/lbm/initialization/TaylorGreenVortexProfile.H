#ifndef LBM_INITIALIZATION_TAYLORGREENVORTEXPROFILE_H
#define LBM_INITIALIZATION_TAYLORGREENVORTEXPROFILE_H

#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/TaylorGreenVortexMath.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::initialization {

/**
 * @brief Built-in Taylor--Green macroscopic initializer on a fixed
 *        mathematical domain.
 *
 * The profile consumes AMReX physical cell-center positions supplied by the
 * initialization operator and maps their x/y coordinates to the default
 * periodic domain [0,1] x [0,1].  It deliberately carries no Geometry,
 * integer-index, boundary-plan, or benchmark-layout knowledge.
 *
 * TaylorGreenVortexParams currently stores decay and pressureDecay already
 * evaluated for the initialization time.  Consequently, the uniform time
 * argument accepted by the profile contract must equal the time used to build
 * params; the single-level initializer currently resolves both at t0 = 0.
 *
 * @pre xLength and yLength are positive, and params was resolved for the time
 *      passed by EquilibriumPopulationInitializationOperator.
 * @note The z coordinate is intentionally ignored by this quasi-two-dimensional
 *       built-in profile.
 */
struct TaylorGreenVortexProfile {
    TaylorGreenVortexParams params;
    amrex::Real xOrigin = 0.0;
    amrex::Real xLength = 1.0;
    amrex::Real yOrigin = 0.0;
    amrex::Real yLength = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real) const noexcept
    {
        amrex::Real ux = 0.0;
        amrex::Real uy = 0.0;
        tgv_velocity(
            params,
            position,
            xOrigin,
            xLength,
            yOrigin,
            yLength,
            ux,
            uy);
        return MacroscopicState{
            tgv_density(
                params,
                position,
                xOrigin,
                xLength,
                yOrigin,
                yLength),
            ux,
            uy,
            0.0};
    }
};

} // namespace lbm::initialization

#endif
