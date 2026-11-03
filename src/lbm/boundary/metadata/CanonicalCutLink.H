#ifndef LBM_BOUNDARY_METADATA_CANONICALCUTLINK_H
#define LBM_BOUNDARY_METADATA_CANONICALCUTLINK_H

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>

#include <limits>
#include <type_traits>

namespace lbm::boundary {

/** @brief Rank-local, scheme-neutral fluid-to-solid link metadata. */
struct CanonicalCutLinkRecord {
    int ownerGrid = -1;
    amrex::GpuArray<int, 3> fluidCell{-1, -1, -1};
    amrex::GpuArray<int, 3> solidCell{-1, -1, -1};
    int population = -1;
    int oppositePopulation = -1;
    int boundaryId = -1;
    int objectId = -1;
    amrex::Real qFromFluid =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::GpuArray<amrex::Real, 3> intersection{
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN()};
    amrex::GpuArray<amrex::Real, 3> normalSolidToFluid{
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN()};
    int surfaceCoincidentFluid = 0;
};

AMREX_GPU_HOST_DEVICE inline bool canonical_cut_link_key_less(
    CanonicalCutLinkRecord const& lhs,
    CanonicalCutLinkRecord const& rhs) noexcept
{
    if (lhs.ownerGrid != rhs.ownerGrid) {
        return lhs.ownerGrid < rhs.ownerGrid;
    }
    for (int d = 0; d < 3; ++d) {
        if (lhs.fluidCell[d] != rhs.fluidCell[d]) {
            return lhs.fluidCell[d] < rhs.fluidCell[d];
        }
    }
    if (lhs.population != rhs.population) {
        return lhs.population < rhs.population;
    }
    if (lhs.boundaryId != rhs.boundaryId) {
        return lhs.boundaryId < rhs.boundaryId;
    }
    return lhs.objectId < rhs.objectId;
}

static_assert(std::is_trivially_copyable_v<CanonicalCutLinkRecord>);

} // namespace lbm::boundary

#endif
