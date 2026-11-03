#ifndef LBM_BOUNDARY_DOMAIN_LBMPOSTSTREAMINGBOUNDARYEXECUTOR_H
#define LBM_BOUNDARY_DOMAIN_LBMPOSTSTREAMINGBOUNDARYEXECUTOR_H

#include "lbm/boundary/domain/BoundarySamplingPosition.H"
#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/domain/post_stream/GuoBoundaryDispatch.H"
#include "lbm/fields/LatticeField.H"

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

namespace lbm::boundary {

/**
 * @brief Executes compiled post-stream boundary regions on valid nodes.
 *
 * This class owns scheme-agnostic region traversal and AMReX kernel launch.
 * Scheme-specific physical/profile selection is delegated to a dispatcher in
 * `domain/post_stream`, keeping the device kernel free of per-cell boundary
 * membership and scheme branches.
 */
template <class Descriptor>
class LbmPostStreamingBoundaryExecutor {
public:
    void apply(
        LatticeField<Descriptor>& populations,
        DomainFaceTableView const& faces,
        PostStreamingBoundaryPlan const& plan,
        amrex::Geometry const& geom,
        amrex::Real physicalTime) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            plan.hasPostStreamingFaces,
            "post-stream boundary executor requires at least one compiled region");
        const amrex::GeometryData geometry = geom.data();

        for (PostStreamBoundaryRegion const& region : plan.regions) {
            const int sideIndex = domain_side_index(region.effectiveSide);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                faces.enabled[sideIndex] != 0 &&
                    faces.stage[sideIndex] ==
                        BoundaryExecutionStage::PostStreamingValidNode,
                "post-stream region does not reference executable face metadata");

            switch (faces.scheme[sideIndex]) {
            case BoundaryScheme::GuoExtrapolation:
                post_stream::GuoBoundaryDispatch<Descriptor>::dispatch(
                    faces,
                    region,
                    sideIndex,
                    [&populations, &region, geometry, physicalTime](
                        auto reconstruction) {
                        applyRegion(
                            populations,
                            region,
                            geometry,
                            physicalTime,
                            reconstruction);
                    });
                break;

            default:
                amrex::Abort(
                    "post-stream executor received unsupported boundary scheme");
            }
        }
    }

    /**
     * @brief Apply one reconstruction functor over a disjoint valid-node region.
     *
     * Donor populations must already be synchronized at the post-stream time
     * layer. The executor computes each boundary node's AMReX physical cell
     * center and forwards it with the target physical time. The functor is
     * copied into the CPU/GPU kernel and must therefore contain only
     * device-safe state callable as
     * `(donor,boundary,physical_position,physicalTime)`.
     *
     * This public kernel-level extension point supports focused contract tests;
     * production physical/profile selection remains in the scheme dispatcher.
     */
    template <class Reconstruction>
    static void applyRegion(
        LatticeField<Descriptor>& populations,
        PostStreamBoundaryRegion const& region,
        amrex::GeometryData geometry,
        amrex::Real physicalTime,
        Reconstruction reconstruction)
    {
        amrex::MultiFab& mf = populations.multifab();
        const int rx = region.inward[0];
        const int ry = region.inward[1];
        const int rz = region.inward[2];

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox() & region.box;
            if (!bx.ok()) {
                continue;
            }
            auto const donorData = mf.const_array(mfi);
            auto const boundaryData = mf.array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    ConstPopulationCell donor{
                        donorData, i + rx, j + ry, k + rz};
                    PopulationCell boundary{boundaryData, i, j, k};
                    const auto position =
                        boundary_cell_center_position(
                            geometry, i, j, k);
                    reconstruction(
                        donor, boundary, position, physicalTime);
                });
        }
    }
};

} // namespace lbm::boundary

#endif
