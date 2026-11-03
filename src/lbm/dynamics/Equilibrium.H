#ifndef LBM_DYNAMICS_EQUILIBRIUM_H
#define LBM_DYNAMICS_EQUILIBRIUM_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm {

template <class Descriptor>
struct Equilibrium {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real compute(
        int iPop,
        amrex::Real rho,
        amrex::Real ux,
        amrex::Real uy,
        amrex::Real uz) noexcept
    {
        const amrex::Real cu =
            Descriptor::cx(iPop) * ux +
            Descriptor::cy(iPop) * uy +
            Descriptor::cz(iPop) * uz;
        const amrex::Real uSqr = ux * ux + uy * uy + uz * uz;
        const amrex::Real invCs2 = Descriptor::invCs2();

        return Descriptor::w(iPop) * rho *
            (1.0 + invCs2 * cu +
             0.5 * invCs2 * invCs2 * cu * cu -
             0.5 * invCs2 * uSqr);
    }
};

} // namespace lbm

#endif
