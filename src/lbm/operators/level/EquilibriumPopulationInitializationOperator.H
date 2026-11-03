#ifndef LBM_OPERATORS_LEVEL_EQUILIBRIUMPOPULATIONINITIALIZATIONOPERATOR_H
#define LBM_OPERATORS_LEVEL_EQUILIBRIUMPOPULATIONINITIALIZATIONOPERATOR_H

#include "lbm/dynamics/Equilibrium.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/operators/core/LevelOperator.H"

#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_REAL.H>

#include <string_view>

namespace lbm::operators {

/**
 * @brief Initialize every valid cell from a device-safe macroscopic profile.
 *
 * The Operator owns the index-space traversal and the conversion from AMReX
 * cell indices to physical cell-center positions. The profile therefore sees
 * only `(position,time)` and returns a macroscopic state; it does not depend on
 * Geometry, MFIter, boundary geometry, or population storage.
 *
 * @tparam Descriptor Lattice descriptor supplying q and equilibrium weights.
 * @tparam Profile Trivially copyable device functor callable as
 *         `profile(GpuArray<Real,AMREX_SPACEDIM>, Real)`.
 *
 * @pre geom describes the same cell-centered valid domain and periodicity as
 *      populations. The supplied time is the initialization time layer.
 * @post Every valid population is reconstructed from equilibrium and
 *       inter-FAB/periodic ghosts are synchronized. Non-periodic physical
 *       ghosts have no boundary-condition meaning after this operation.
 *
 * All ranks must call apply in the same application phase because the final
 * FillBoundary may communicate across MPI ranks. Each ParallelFor writes only
 * its owning valid tile and is safe under the existing MFIter/OpenMP policy.
 */
template <class Descriptor, class Profile>
class EquilibriumPopulationInitializationOperator : public LevelOperator {
public:
    EquilibriumPopulationInitializationOperator(
        Profile profile,
        amrex::Geometry const& geom) noexcept
        : profile_(profile)
        , geom_(geom)
    {}

    std::string_view name() const noexcept override
    {
        return "equilibrium_population_initialization";
    }

    void apply(
        LatticeField<Descriptor>& populations,
        amrex::Real initializationTime) const
    {
        amrex::MultiFab& f = populations.multifab();
        const amrex::GeometryData geometry = geom_.data();
        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.tilebox();
            auto const fArr = f.array(mfi);
            auto const localProfile = profile_;

            amrex::ParallelFor(bx, Descriptor::q,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) {
                    const amrex::GpuArray<
                        amrex::Real,
                        AMREX_SPACEDIM> position{
                        AMREX_D_DECL(
                            geometry.ProbLo(0) +
                                (static_cast<amrex::Real>(i) + 0.5) *
                                    geometry.CellSize(0),
                            geometry.ProbLo(1) +
                                (static_cast<amrex::Real>(j) + 0.5) *
                                    geometry.CellSize(1),
                            geometry.ProbLo(2) +
                                (static_cast<amrex::Real>(k) + 0.5) *
                                    geometry.CellSize(2))};
                    const auto state =
                        localProfile(position, initializationTime);
                    fArr(i, j, k, iPop) = Equilibrium<Descriptor>::compute(
                        iPop, state.rho, state.ux, state.uy, state.uz);
                });
        }
        populations.fillBoundary(geom_);
    }

private:
    Profile profile_;
    amrex::Geometry const& geom_;
};

} // namespace lbm::operators

#endif
