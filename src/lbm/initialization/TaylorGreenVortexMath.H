#ifndef LBM_INITIALIZATION_TAYLORGREENVORTEXMATH_H
#define LBM_INITIALIZATION_TAYLORGREENVORTEXMATH_H

#include "lbm/initialization/TaylorGreenVortexParams.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace lbm::initialization {

/**
 * @brief Return the same Taylor--Green state parameters at one lattice time.
 *
 * The resolved density, velocity amplitude, viscosity, sound speed, and wave
 * numbers remain unchanged. Only the absolute analytical decay factors are
 * recomputed, so callers may derive diagnostics at time n from the immutable
 * t0 initialization setup without falling back to unrelated LBM reference
 * values.
 */
inline TaylorGreenVortexParams tgv_params_at_time(
    TaylorGreenVortexParams params,
    amrex::Real time)
{
    params.decay = std::exp(
        -params.nu *
        (params.kx * params.kx + params.ky * params.ky) *
        time);
    params.pressureDecay = params.decay * params.decay;
    return params;
}

inline TaylorGreenVortexParams make_tgv_params(
    int nx,
    int ny,
    amrex::Real rho0,
    amrex::Real u0,
    amrex::Real nu,
    amrex::Real cs2,
    amrex::Real time)
{
    TaylorGreenVortexParams p;
    p.rho0 = rho0;
    p.u0 = u0;
    p.nu = nu;
    p.cs2 = cs2;
    p.length = static_cast<amrex::Real>(nx);
    p.kx = 2.0 * amrex::Math::pi<amrex::Real>() / p.length;
    p.ky = 2.0 * amrex::Math::pi<amrex::Real>() / static_cast<amrex::Real>(ny);
    return tgv_params_at_time(p, time);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void tgv_velocity(
    TaylorGreenVortexParams const& p,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
    amrex::Real xOrigin,
    amrex::Real xLength,
    amrex::Real yOrigin,
    amrex::Real yLength,
    amrex::Real& ux,
    amrex::Real& uy) noexcept
{
    const amrex::Real x =
        2.0 * amrex::Math::pi<amrex::Real>() *
        (position[0] - xOrigin) / xLength;
    const amrex::Real y =
        2.0 * amrex::Math::pi<amrex::Real>() *
        (position[1] - yOrigin) / yLength;
    const auto sx_cx = amrex::Math::sincos(x);
    const auto sy_cy = amrex::Math::sincos(y);

    ux = -p.u0 * sx_cx.second * sy_cy.first * p.decay;
    uy =  p.u0 * sx_cx.first  * sy_cy.second * p.decay;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real tgv_density(
    TaylorGreenVortexParams const& p,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
    amrex::Real xOrigin,
    amrex::Real xLength,
    amrex::Real yOrigin,
    amrex::Real yLength) noexcept
{
    const amrex::Real x2 =
        4.0 * amrex::Math::pi<amrex::Real>() *
        (position[0] - xOrigin) / xLength;
    const amrex::Real y2 =
        4.0 * amrex::Math::pi<amrex::Real>() *
        (position[1] - yOrigin) / yLength;
    const auto sx2_cx2 = amrex::Math::sincos(x2);
    const auto sy2_cy2 = amrex::Math::sincos(y2);

    return p.rho0 -
        0.25 * p.rho0 * p.u0 * p.u0 / p.cs2 *
        (sx2_cx2.second + sy2_cy2.second) * p.pressureDecay;
}

} // namespace lbm::initialization

#endif
