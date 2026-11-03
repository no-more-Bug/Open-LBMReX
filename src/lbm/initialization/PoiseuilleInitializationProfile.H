#ifndef LBM_INITIALIZATION_POISEUILLEINITIALIZATIONPROFILE_H
#define LBM_INITIALIZATION_POISEUILLEINITIALIZATIONPROFILE_H

#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/PoiseuilleInitializationParams.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::initialization {

/**
 * @brief Device-safe macroscopic profile for Poiseuille bulk initialization.
 *
 * This built-in is the standard x-directed profile on the mathematical domain
 * `[0,4] x [0,1]`. Those constants are profile data, not inferred from AMReX
 * Geometry or a boundary plan. The user/application validator is responsible
 * for making the AMReX physical envelope consistent with this selected
 * built-in.
 *
 * This type is independent of boundary profiles. It initializes interior-fluid
 * valid cells (including a temporary value on Guo boundary nodes) and returns
 * only rho/u; the equilibrium population Operator performs all field writes.
 */
struct PoiseuilleInitializationProfile {
    PoiseuilleInitializationParams params;
    amrex::Real streamwiseOrigin = 0.0;
    amrex::Real streamwiseLength = 4.0;
    amrex::Real transverseOrigin = 0.0;
    amrex::Real channelHeight = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real) const noexcept
    {
        const amrex::Real xi =
            (position[0] - streamwiseOrigin) / streamwiseLength;
        const amrex::Real eta =
            (position[1] - transverseOrigin) / channelHeight;
        const amrex::Real rho = params.linearDensity != 0 ?
            params.rhoXLoLb +
                xi * (params.rhoXHiLb - params.rhoXLoLb) :
            params.rhoLb;
        return MacroscopicState{
            rho,
            4.0 * params.uMaxLb * eta * (1.0 - eta),
            0.0,
            0.0};
    }
};

} // namespace lbm::initialization

#endif
