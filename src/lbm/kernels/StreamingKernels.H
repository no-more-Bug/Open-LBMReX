#ifndef LBM_KERNELS_STREAMINGKERNELS_H
#define LBM_KERNELS_STREAMINGKERNELS_H

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::kernels {

template <class Descriptor>
struct PullStreamingKernel {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real population(
        amrex::Array4<amrex::Real const> const& src,
        int i,
        int j,
        int k,
        int iPop) noexcept
    {
        return src(i - Descriptor::cx(iPop),
                   j - Descriptor::cy(iPop),
                   k - Descriptor::cz(iPop),
                   iPop);
    }
};

} // namespace lbm::kernels

#endif
