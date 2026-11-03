#ifndef LBM_AMR_SYNCHRONIZEDVELOCITY_HH
#define LBM_AMR_SYNCHRONIZEDVELOCITY_HH

#include "lbm/kernels/MacroscopicKernels.H"

#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>

namespace lbm::amr {
namespace detail {
template<class Descriptor>
void fill_synchronized_velocity(amrex::MultiFab const& populations, amrex::MultiFab& velocity)
{
        for (amrex::MFIter mfi(populations, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const f = populations.const_array(mfi);
            auto const u = velocity.array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                amrex::Real rho, ux, uy, uz;
                kernels::MacroscopicKernel<Descriptor>::compute(
                    ConstPopulationCell{f,i,j,k}, rho, ux, uy, uz);
                u(i,j,k,0) = ux; u(i,j,k,1) = uy; u(i,j,k,2) = uz;
            });
        }
}
} // namespace detail


template<class Descriptor>
SynchronizedVelocity<Descriptor>::SynchronizedVelocity(
    std::vector<VelocitySourceLevel> const& levels, amrex::IntVect const& ratio)
{
    auto const grow = stencil_extent<Descriptor>();
    velocity_.reserve(levels.size());
    for (auto const& source : levels) {
        auto const& populations = source.populations;
        auto velocity = std::make_unique<amrex::MultiFab>(
            populations.boxArray(), populations.DistributionMap(), 3, grow);
        detail::fill_synchronized_velocity<Descriptor>(populations,*velocity);
        velocity_.push_back(std::move(velocity));
    }
    for (int lev = static_cast<int>(levels.size())-1; lev > 0; --lev)
        amrex::average_down(*velocity_[lev], *velocity_[lev-1],
                           levels[lev].geometry, levels[lev-1].geometry, 0, 3, ratio);

    amrex::Vector<amrex::BCRec> boundaries(3);
    for (auto& boundary : boundaries) {
        for (int d = 0; d < 3; ++d) {
            boundary.setLo(d, amrex::BCType::int_dir);
            boundary.setHi(d, amrex::BCType::int_dir);
        }
    }
    amrex::PhysBCFunctNoOp physicalBoundary;
    for (int lev = 0; lev < static_cast<int>(levels.size()); ++lev) {
        if (lev == 0) {
            velocity_[lev]->FillBoundary(levels[lev].geometry.periodicity());
        } else {
            amrex::FillPatchTwoLevels(*velocity_[lev], grow, 0.0,
                amrex::Vector<amrex::MultiFab*>{velocity_[lev-1].get()}, amrex::Vector<amrex::Real>{0.0},
                amrex::Vector<amrex::MultiFab*>{velocity_[lev].get()}, amrex::Vector<amrex::Real>{0.0},
                0, 0, 3, levels[lev-1].geometry, levels[lev].geometry,
                physicalBoundary, 0, physicalBoundary, 0, ratio,
                &amrex::cell_cons_interp, boundaries, 0);
        }
    }
}

template<class Descriptor>
SynchronizedVelocity<Descriptor>::~SynchronizedVelocity()
{
    amrex::Gpu::streamSynchronize();
}

} // namespace lbm::amr
#endif
