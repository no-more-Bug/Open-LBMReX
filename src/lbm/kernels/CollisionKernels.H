#ifndef LBM_KERNELS_COLLISIONKERNELS_H
#define LBM_KERNELS_COLLISIONKERNELS_H

#include <AMReX_GpuQualifiers.H>

namespace lbm::kernels {

template <class Descriptor>
struct CollisionKernel {
    template <class Cell, class Dynamics>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void apply(Cell&& cell, Dynamics const& dynamics) noexcept
    {
        dynamics.collide(cell);
    }
};

} // namespace lbm::kernels

#endif
