#ifndef LBM_BOUNDARY_METADATA_INTERNALBOUNDARYEXECUTION_H
#define LBM_BOUNDARY_METADATA_INTERNALBOUNDARYEXECUTION_H

#include "lbm/boundary/internal/InternalBoundarySchemeTraits.H"
#include "lbm/boundary/metadata/CanonicalCutLink.H"

#include <AMReX_GpuQualifiers.H>

#include <limits>
#include <type_traits>

namespace lbm::boundary {

struct InternalBoundaryExecutionRecord {
    int canonicalIndex = -1;
    amrex::GpuArray<int, 3> anchorCell{-1, -1, -1};
    int direction = -1;
    int oppositeDirection = -1;
    amrex::Real qFromAnchor =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    BoundaryScheme scheme = BoundaryScheme::Bouzidi;
    InternalBoundaryHalfLinkPolicy halfLinkPolicy =
        InternalBoundaryHalfLinkPolicy::PiecewiseAtHalf;
};

struct InternalBoundaryMetadataView {
    CanonicalCutLinkRecord const* canonical = nullptr;
    InternalBoundaryExecutionRecord const* execution = nullptr;
    amrex::Long const* gridOffsets = nullptr;
    amrex::Long count = 0;
    int gridCount = 0;

    AMREX_GPU_HOST_DEVICE amrex::Long begin(int grid) const noexcept
    {
        return gridOffsets[grid];
    }
    AMREX_GPU_HOST_DEVICE amrex::Long end(int grid) const noexcept
    {
        return gridOffsets[grid + 1];
    }
};

static_assert(std::is_trivially_copyable_v<InternalBoundaryExecutionRecord>);
static_assert(std::is_trivially_copyable_v<InternalBoundaryMetadataView>);

} // namespace lbm::boundary

#endif
