#ifndef LBM_DYNAMICS_MOMENTS_H
#define LBM_DYNAMICS_MOMENTS_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm {

template <class Descriptor>
struct Moments {
    template <class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real rho(Cell const& f) noexcept
    {
        amrex::Real rhoValue = 0.0;
        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            rhoValue += f[iPop];
        }
        return rhoValue;
    }

    template <class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void velocity(
        Cell const& f,
        amrex::Real rhoValue,
        amrex::Real& ux,
        amrex::Real& uy,
        amrex::Real& uz) noexcept
    {
        amrex::Real jx = 0.0;
        amrex::Real jy = 0.0;
        amrex::Real jz = 0.0;

        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            jx += Descriptor::cx(iPop) * f[iPop];
            jy += Descriptor::cy(iPop) * f[iPop];
            jz += Descriptor::cz(iPop) * f[iPop];
        }

        const amrex::Real invRho = rhoValue != 0.0 ? 1.0 / rhoValue : 0.0;
        ux = jx * invRho;
        uy = jy * invRho;
        uz = jz * invRho;
    }
};

} // namespace lbm

#endif
