#ifndef LBM_CASES_POISEUILLEFLOW_EXACTREFERENCE_H
#define LBM_CASES_POISEUILLEFLOW_EXACTREFERENCE_H

#include "poiseuille_flow/observation/PoiseuilleDiagnostics.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::cases::poiseuille_flow {

struct ExactMacroscopicReference {
    amrex::Real rho = 1.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

struct PoiseuilleExactReference {
    PoiseuilleDiagnosticGeometry geometry;
    amrex::Real rhoReference = 1.0;
    amrex::Real velocityReference = 0.0;
    amrex::Real massReferenceDensity = 1.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    ExactMacroscopicReference operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int j,
        int) const noexcept
    {
        return {
            rhoReference,
            poiseuille_ux_exact(geometry, j, velocityReference),
            0.0,
            0.0};
    }
};

} // namespace lbm::cases::poiseuille_flow

#endif
