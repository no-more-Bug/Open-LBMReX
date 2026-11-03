#ifndef LBM_DESCRIPTORS_D2Q9DESCRIPTOR_H
#define LBM_DESCRIPTORS_D2Q9DESCRIPTOR_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm {

struct D2Q9Descriptor {
    using real = amrex::Real;

    static constexpr int d = 2;
    static constexpr int q = 9;
    static constexpr int vicinity = 1;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int iPop) noexcept
    {
        return (iPop == 1 || iPop == 2 || iPop == 3) ? -1 :
               (iPop == 5 || iPop == 6 || iPop == 7) ? 1 : 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int iPop) noexcept
    {
        return (iPop == 1 || iPop == 7 || iPop == 8) ? 1 :
               (iPop == 3 || iPop == 4 || iPop == 5) ? -1 : 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int) noexcept
    {
        return 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int c(int iPop, int iDim) noexcept
    {
        return iDim == 0 ? cx(iPop) : (iDim == 1 ? cy(iPop) : cz(iPop));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cNormSqr(int iPop) noexcept
    {
        return cx(iPop) * cx(iPop) + cy(iPop) * cy(iPop);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr real w(int iPop) noexcept
    {
        return iPop == 0 ? real(4.0 / 9.0) :
               cNormSqr(iPop) == 1 ? real(1.0 / 9.0) : real(1.0 / 36.0);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr real cs2() noexcept
    {
        return real(1.0 / 3.0);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr real invCs2() noexcept
    {
        return real(3.0);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr real rho0_lb() noexcept
    {
        return real(1.0);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int opposite(int iPop) noexcept
    {
        return iPop == 0 ? 0 :
               iPop == 1 ? 5 :
               iPop == 2 ? 6 :
               iPop == 3 ? 7 :
               iPop == 4 ? 8 :
               iPop == 5 ? 1 :
               iPop == 6 ? 2 :
               iPop == 7 ? 3 : 4;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr char const* name() noexcept
    {
        return "D2Q9";
    }
};

} // namespace lbm

#endif
