#ifndef LBM_BOUNDARY_INTERNAL_INTERNALPOPULATIONRECONSTRUCTION_H
#define LBM_BOUNDARY_INTERNAL_INTERNALPOPULATIONRECONSTRUCTION_H

#include "lbm/boundary/metadata/InternalBoundaryExecution.H"
#include "lbm/boundary/schemes/BouzidiInterpolatedBounceBackScheme.H"
#include "lbm/boundary/schemes/YuMeiShyyScheme.H"

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

struct InternalPopulationDonors {
    amrex::Real anchorForward;
    amrex::Real anchorOpposite;
    amrex::Real nextFluidForward;
};

/** @brief Reconstruct one validated static no-slip internal population. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real reconstruct_internal_population(
    InternalBoundaryExecutionRecord const& record,
    InternalPopulationDonors const& donors) noexcept
{
    AMREX_ASSUME(
        record.scheme == BoundaryScheme::Bouzidi ||
        record.scheme == BoundaryScheme::Yms);
    if (record.scheme == BoundaryScheme::Bouzidi) {
        return BouzidiInterpolatedBounceBackScheme::reconstruct(
            record.qFromAnchor,
            donors.anchorForward,
            donors.anchorOpposite,
            donors.nextFluidForward);
    }
    return YuMeiShyyScheme::reconstruct(
        record.qFromAnchor,
        donors.anchorForward,
        donors.anchorOpposite,
        donors.nextFluidForward);
}

} // namespace lbm::boundary

#endif
