#ifndef LBM_OPERATORS_LEVEL_PULLSTREAMINGOPERATOR_H
#define LBM_OPERATORS_LEVEL_PULLSTREAMINGOPERATOR_H

#include "lbm/fields/LatticeField.H"
#include "lbm/kernels/StreamingKernels.H"
#include "lbm/operators/core/LevelOperator.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

#include <string_view>

namespace lbm::operators {

/**
 * @brief Pull all populations from a collided source field into a destination field.
 *
 * The operator traverses the complete AMReX valid domain and deliberately does
 * not skip valid boundary nodes. HBB has already prepared physical source
 * ghosts, while disposable streamed values on Guo boundary nodes are replaced
 * by the post-stream boundary stage.
 */
template <class Descriptor>
class PullStreamingOperator : public LevelOperator {
public:
    PullStreamingOperator() noexcept = default;

    std::string_view name() const noexcept override
    {
        return "pull_streaming";
    }

    /**
     * @brief Produce the next population layer over every valid cell and iPop.
     *
     * @pre src and dst have identical BoxArray/DistributionMapping layouts;
     *      collision has updated src valid cells and the pre-stream stage has
     *      synchronized/finalized every source ghost that a pull may read.
     * @post Every population in valid + writeGrow is written once per FAB;
     *       other destination cells and all source populations remain unchanged.
     *       The default zero width preserves the valid-only single-level pull.
     *
     * MFIter distributes local tiles across OpenMP threads on CPU; Array4 views
     * are captured by value into the device-safe ParallelFor kernel.
     */
    void apply(
        LatticeField<Descriptor>& src,
        LatticeField<Descriptor>& dst,
        amrex::IntVect const& writeGrow = amrex::IntVect(0)) const
    {
        AMREX_ALWAYS_ASSERT(writeGrow.allGE(0) && dst.nGrowVect().allGE(writeGrow) &&
            src.nGrowVect().allGE(writeGrow + stencil_extent<Descriptor>()));
        amrex::MultiFab const& srcMf = src.multifab();
        amrex::MultiFab& dstMf = dst.multifab();

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(dstMf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            // Grow only tiles touching the FAB edge; interior tiles stay disjoint.
            amrex::Box const bx = mfi.growntilebox(writeGrow);
            auto const srcArr = srcMf.const_array(mfi);
            auto const dstArr = dstMf.array(mfi);

            amrex::ParallelFor(bx, Descriptor::q,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) {
                    dstArr(i, j, k, iPop) =
                        kernels::PullStreamingKernel<Descriptor>::population(
                            srcArr, i, j, k, iPop);
                });
        }
    }
};

} // namespace lbm::operators

#endif
