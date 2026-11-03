#ifndef LBM_BOUNDARY_INTERNAL_INTERNALSOLIDPOPULATIONEXECUTOR_H
#define LBM_BOUNDARY_INTERNAL_INTERNALSOLIDPOPULATIONEXECUTOR_H

#include "lbm/boundary/internal/InternalPopulationReconstruction.H"
#include "lbm/boundary/metadata/InternalBoundaryMetadataStorage.H"
#include "lbm/fields/LatticeField.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

namespace lbm::boundary {

/**
 * @brief Execute rank-local internal cut-link reconstruction.
 *
 * collidedSource is the collision-complete field whose ghosts were synchronized
 * before pull streaming. streamedDestination contains the provisional full-valid
 * pull result; each execution record overwrites its missing incoming population.
 */
template <class Descriptor>
class InternalSolidPopulationExecutor {
public:
    void apply(
        InternalBoundaryMetadataStorage const& storage,
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor>& streamedDestination) const
    {
        // Grid ranges are host scheduling data. Only the device view is
        // dereferenced inside ParallelFor, where it is valid for CPU and GPU.
        const InternalBoundaryMetadataView host = storage.hostView();
        const InternalBoundaryMetadataView device = storage.deviceView();
        amrex::MultiFab const& srcMf = collidedSource.multifab();
        amrex::MultiFab& dstMf = streamedDestination.multifab();

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        // Default non-tiled MFIter visits each rank-local FAB once. Its global
        // BoxArray index selects that FAB's contiguous execution-record range.
        for (amrex::MFIter mfi(dstMf); mfi.isValid(); ++mfi) {
            const int grid = mfi.index();
            const amrex::Long begin = host.begin(grid);
            const amrex::Long count = host.end(grid) - begin;
            if (count == 0) {
                continue;
            }
            auto const src = srcMf.const_array(mfi);
            auto const dst = dstMf.array(mfi);

            // One logical iteration owns one cut link and its single destination
            // component, so no atomics or ordering between records is required.
            amrex::ParallelFor(
                count,
                [=] AMREX_GPU_DEVICE(amrex::Long n) noexcept {
                    auto const& record = device.execution[begin + n];
                    const int i = record.anchorCell[0];
                    const int j = record.anchorCell[1];
                    const int k = record.anchorCell[2];
                    const int direction = record.direction;
                    const int opposite = record.oppositeDirection;
                    // direction points from fluid anchor x_f into the solid;
                    // the third donor is the next fluid cell x_f-c_direction.
                    InternalPopulationDonors const donors{
                        src(i, j, k, direction),
                        src(i, j, k, opposite),
                        src(
                            i - Descriptor::cx(direction),
                            j - Descriptor::cy(direction),
                            k - Descriptor::cz(direction),
                            direction)};
                    dst(i, j, k, opposite) =
                        reconstruct_internal_population(record, donors);
                });
        }
    }
};

} // namespace lbm::boundary

#endif
