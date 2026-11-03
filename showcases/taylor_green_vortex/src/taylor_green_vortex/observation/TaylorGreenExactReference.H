#ifndef LBM_CASES_TAYLORGREENVORTEX_EXACTREFERENCE_H
#define LBM_CASES_TAYLORGREENVORTEX_EXACTREFERENCE_H

#include "lbm/initialization/TaylorGreenVortexMath.H"
#include "lbm/initialization/TaylorGreenVortexProfile.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::cases::taylor_green_vortex {

/** Device-safe, case-owned analytical state for one observed time layer. */
struct TaylorGreenExactReference {
    initialization::TaylorGreenVortexProfile profile;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        int,
        int,
        int) const noexcept
    {
        return profile(position, 0.0);
    }
};

inline TaylorGreenExactReference make_exact_reference(
    initialization::TaylorGreenVortexParams const& initial,
    int step)
{
    initialization::TaylorGreenVortexParams const evolved =
        initialization::tgv_params_at_time(
            initial, static_cast<amrex::Real>(step));
    return {
        initialization::TaylorGreenVortexProfile{evolved},
        initial.rho0};
}

} // namespace lbm::cases::taylor_green_vortex

#endif
