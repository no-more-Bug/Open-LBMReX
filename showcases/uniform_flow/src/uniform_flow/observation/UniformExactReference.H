#ifndef LBM_CASES_UNIFORMFLOW_EXACTREFERENCE_H
#define LBM_CASES_UNIFORMFLOW_EXACTREFERENCE_H

#include "lbm/initialization/MacroscopicState.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::cases::uniform_flow {

struct ExactMacroscopicReference {
    amrex::Real rho = 1.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

struct UniformExactReference {
    ExactMacroscopicReference state;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ExactMacroscopicReference operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return state;
    }
};

inline UniformExactReference make_exact_reference(
    initialization::MacroscopicState const& state) noexcept
{
    return {{state.rho, state.ux, state.uy, state.uz}, state.rho};
}

} // namespace lbm::cases::uniform_flow

#endif
