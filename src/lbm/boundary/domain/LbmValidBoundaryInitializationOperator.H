#ifndef LBM_BOUNDARY_DOMAIN_LBMVALIDBOUNDARYINITIALIZATIONOPERATOR_H
#define LBM_BOUNDARY_DOMAIN_LBMVALIDBOUNDARYINITIALIZATIONOPERATOR_H

#include "lbm/boundary/domain/BoundarySamplingPosition.H"
#include "lbm/boundary/domain/DomainFaceMetadata.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/domain/initialization/GuoBoundaryInitializationDispatch.H"
#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/operators/core/LevelOperator.H"

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <string_view>

namespace lbm::boundary {

/**
 * @brief Initialize compiled valid-node boundary regions with equilibrium data.
 *
 * This level Operator is used only after bulk valid populations have been
 * initialized. For each disjoint region it recovers the first strictly
 * interior donor's macroscopic state, lets the scheme-specific initialization
 * dispatcher complete the prescribed boundary state at the supplied current
 * time, and overwrites all q boundary populations with that state's
 * equilibrium. It never evaluates runtime donor non-equilibrium terms.
 *
 * Donor ghosts, when needed by the BoxArray decomposition, must be synchronized
 * before apply(). The compiled plan guarantees that region donors are strictly
 * interior to the physical shell, so in-place boundary writes cannot become a
 * later region's donor.
 */
template <class Descriptor>
class LbmValidBoundaryInitializationOperator : public operators::LevelOperator {
public:
    std::string_view name() const noexcept override
    {
        return "valid_boundary_initialization";
    }

    /**
     * @brief Initialize every post-stream region in a compiled boundary plan.
     *
     * A pure HBB plan has no post-stream regions and is an intentional no-op.
     * Physical/profile selection remains outside the kernel in the narrow
     * scheme dispatcher.
     */
    void apply(
        LatticeField<Descriptor>& populations,
        DomainFaceTableView const& faces,
        PostStreamingBoundaryPlan const& plan,
        amrex::Geometry const& geom,
        amrex::Real physicalTime) const
    {
        if (!plan.hasPostStreamingFaces) {
            return;
        }
        const amrex::GeometryData geometry = geom.data();

        for (PostStreamBoundaryRegion const& region : plan.regions) {
            const int sideIndex = domain_side_index(region.effectiveSide);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                faces.enabled[sideIndex] != 0 &&
                    faces.stage[sideIndex] ==
                        BoundaryExecutionStage::PostStreamingValidNode,
                "valid-boundary initialization region references non-executable face metadata");

            switch (faces.scheme[sideIndex]) {
            case BoundaryScheme::GuoExtrapolation:
                initialization::GuoBoundaryInitializationDispatch::dispatch(
                    faces,
                    region,
                    sideIndex,
                    [this, &populations, &region, geometry, physicalTime](
                        auto stateResolver) {
                        applyRegion(
                            populations,
                            region,
                            geometry,
                            physicalTime,
                            stateResolver);
                    });
                break;

            default:
                amrex::Abort(
                    "valid-boundary initialization received unsupported scheme");
            }
        }
    }

    /**
     * @brief Apply one already-dispatched state resolver to a compiled region.
     *
     * This is the kernel-level extension point shared by the built-in Guo
     * dispatcher and focused diagnostics. The resolver must be device-safe and
     * callable as `(donor_state,physical_position,physicalTime)`.
     */
    template <class StateResolver>
    void applyRegion(
        LatticeField<Descriptor>& populations,
        PostStreamBoundaryRegion const& region,
        amrex::GeometryData geometry,
        amrex::Real physicalTime,
        StateResolver stateResolver) const
    {
        amrex::MultiFab& mf = populations.multifab();
        const int rx = region.inward[0];
        const int ry = region.inward[1];
        const int rz = region.inward[2];

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(mf, amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
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
                    initialization::ValidBoundaryMacroscopicState donorState;
                    donorState.rho = Moments<Descriptor>::rho(donor);
                    Moments<Descriptor>::velocity(
                        donor,
                        donorState.rho,
                        donorState.ux,
                        donorState.uy,
                        donorState.uz);
                    const auto position =
                        boundary_cell_center_position(
                            geometry, i, j, k);
                    const initialization::ValidBoundaryMacroscopicState state =
                        stateResolver(donorState, position, physicalTime);
                    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                        boundaryData(i, j, k, iPop) =
                            Equilibrium<Descriptor>::compute(
                                iPop,
                                state.rho,
                                state.ux,
                                state.uy,
                                state.uz);
                    }
                });
        }
    }
};

} // namespace lbm::boundary

#endif
