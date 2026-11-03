#ifndef LBM_DESCRIPTORS_STENCILEXTENT_H
#define LBM_DESCRIPTORS_STENCILEXTENT_H

#include <AMReX_IntVect.H>

namespace lbm {

// Directional reach, including zero extent along inactive stencil axes.
template<class Descriptor>
constexpr amrex::IntVect stencil_extent() noexcept
{
    amrex::IntVect extent(0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int q = 0; q < Descriptor::q; ++q) {
            int const c = Descriptor::c(q, d);
            int const magnitude = c < 0 ? -c : c;
            if (magnitude > extent[d]) extent[d] = magnitude;
        }
    }
    return extent;
}

} // namespace lbm

#endif
