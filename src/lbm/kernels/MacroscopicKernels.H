#ifndef LBM_KERNELS_MACROSCOPICKERNELS_H
#define LBM_KERNELS_MACROSCOPICKERNELS_H

#include "lbm/dynamics/Moments.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::kernels {

template <class Descriptor>
struct MacroscopicKernel {
    template <class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void compute(
        Cell const& cell,
        amrex::Real& rho,
        amrex::Real& ux,
        amrex::Real& uy,
        amrex::Real& uz) noexcept
    {
        rho = Moments<Descriptor>::rho(cell);
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        Moments<Descriptor>::velocity(cell, rho, ux, uy, uz);
    }
};

} // namespace lbm::kernels

#endif
