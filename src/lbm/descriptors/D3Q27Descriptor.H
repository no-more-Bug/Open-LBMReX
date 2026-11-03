#ifndef LBM_DESCRIPTORS_D3Q27DESCRIPTOR_H
#define LBM_DESCRIPTORS_D3Q27DESCRIPTOR_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm {

struct D3Q27Descriptor {
    using real = amrex::Real;

    static constexpr int d = 3;
    static constexpr int q = 27;
    static constexpr int vicinity = 1;

    // Palabos-compatible ordering: rest population first, then 13 negative
    // directions followed by their opposite partners.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int iPop) noexcept
    {
        return iPop == 0 ? 0 :
               (iPop == 1 || iPop == 4 || iPop == 5 || iPop == 6 ||
                iPop == 7 || iPop == 10 || iPop == 11 || iPop == 12 ||
                iPop == 13) ? -1 :
               (iPop == 14 || iPop == 17 || iPop == 18 || iPop == 19 ||
                iPop == 20 || iPop == 23 || iPop == 24 || iPop == 25 ||
                iPop == 26) ? 1 : 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int iPop) noexcept
    {
        return iPop == 0 ? 0 :
               (iPop == 2 || iPop == 4 || iPop == 8 || iPop == 9 ||
                iPop == 10 || iPop == 11 || iPop == 18 || iPop == 25 ||
                iPop == 26) ? -1 :
               (iPop == 5 || iPop == 12 || iPop == 13 || iPop == 15 ||
                iPop == 17 || iPop == 21 || iPop == 22 || iPop == 23 ||
                iPop == 24) ? 1 : 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int iPop) noexcept
    {
        return iPop == 0 ? 0 :
               (iPop == 3 || iPop == 6 || iPop == 8 || iPop == 10 ||
                iPop == 12 || iPop == 20 || iPop == 22 || iPop == 24 ||
                iPop == 26) ? -1 :
               (iPop == 7 || iPop == 9 || iPop == 11 || iPop == 13 ||
                iPop == 16 || iPop == 19 || iPop == 21 || iPop == 23 ||
                iPop == 25) ? 1 : 0;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int c(int iPop, int iDim) noexcept
    {
        return iDim == 0 ? cx(iPop) : (iDim == 1 ? cy(iPop) : cz(iPop));
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cNormSqr(int iPop) noexcept
    {
        return cx(iPop) * cx(iPop) + cy(iPop) * cy(iPop) + cz(iPop) * cz(iPop);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr real w(int iPop) noexcept
    {
        return cNormSqr(iPop) == 0 ? real(8.0 / 27.0) :
               cNormSqr(iPop) == 1 ? real(2.0 / 27.0) :
               cNormSqr(iPop) == 2 ? real(1.0 / 54.0) : real(1.0 / 216.0);
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
        return iPop == 0 ? 0 : (iPop <= 13 ? iPop + 13 : iPop - 13);
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr char const* name() noexcept
    {
        return "D3Q27";
    }
};

} // namespace lbm

#endif
