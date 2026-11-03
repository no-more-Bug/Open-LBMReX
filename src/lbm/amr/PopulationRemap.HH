#ifndef LBM_AMR_POPULATIONREMAP_HH
#define LBM_AMR_POPULATIONREMAP_HH

#include <AMReX_Gpu.H>
#include <AMReX_MultiFabUtil.H>

namespace lbm::amr {

template<class Descriptor>
PopulationRemap<Descriptor>::PopulationRemap(
    std::vector<amrex::MultiFab const*> oldLevels, amrex::IntVect ratio)
    : oldLevels_(std::move(oldLevels)), ratio_(ratio)
{
    AMREX_ALWAYS_ASSERT(!oldLevels_.empty());
    restrictions_.resize(oldLevels_.size()-1);
    for (int lev = static_cast<int>(oldLevels_.size())-2; lev >= 0; --lev) {
        auto const& fine = *oldLevels_[lev+1];
        std::unique_ptr<amrex::MultiFab> composite;
        auto const* source = &fine;
        if (lev+1 < static_cast<int>(restrictions_.size())) {
            composite = std::make_unique<amrex::MultiFab>(
                fine.boxArray(),fine.DistributionMap(),Descriptor::q,0);
            amrex::MultiFab::Copy(*composite,fine,0,0,Descriptor::q,0);
            composite->ParallelCopy(*restrictions_[lev+1],0,0,Descriptor::q,0,0);
            source = composite.get();
        }
        auto coarseBoxes = fine.boxArray();
        coarseBoxes.coarsen(ratio_);
        auto restricted = std::make_unique<amrex::MultiFab>(
            coarseBoxes,fine.DistributionMap(),Descriptor::q,0);
        amrex::average_down(*source,*restricted,0,Descriptor::q,ratio_);
        amrex::Gpu::streamSynchronize();
        restrictions_[lev] = std::move(restricted);
    }
}

template<class Descriptor>
void PopulationRemap<Descriptor>::fillNewLevel(
    int lev, amrex::MultiFab& destination, amrex::MultiFab const* coarseNew,
    amrex::BoxArray const& newFineBoxes) const
{
    if (lev > 0) {
        AMREX_ALWAYS_ASSERT(coarseNew != nullptr);
        auto donorBoxes = destination.boxArray();
        donorBoxes.coarsen(ratio_);
        amrex::MultiFab donor(donorBoxes,destination.DistributionMap(),Descriptor::q,0);
        donor.ParallelCopy(*coarseNew,0,0,Descriptor::q,0,0);
        auto const ratio = ratio_;
        for (amrex::MFIter mfi(destination,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const out = destination.array(mfi);
            auto const in = donor.const_array(mfi);
            // HierarchyGeometry defines zero-based domains, so valid indices are nonnegative.
            amrex::ParallelFor(mfi.tilebox(),Descriptor::q,
                [=] AMREX_GPU_DEVICE(int i,int j,int k,int q) {
                    out(i,j,k,q) = in(i/ratio[0],j/ratio[1],k/ratio[2],q);
                });
        }
        amrex::Gpu::streamSynchronize();
    }
    if (lev < static_cast<int>(oldLevels_.size()))
        destination.ParallelCopy(*oldLevels_[lev],0,0,Descriptor::q,0,0);
    backfillUncovered(lev,destination,newFineBoxes);
}

template<class Descriptor>
void PopulationRemap<Descriptor>::backfillUncovered(
    int lev, amrex::MultiFab& destination, amrex::BoxArray const& newFineBoxes) const
{
    if (lev >= static_cast<int>(restrictions_.size())) return;
    auto const oldCovered = amrex::makeFineMask(destination.boxArray(),destination.DistributionMap(),
        oldLevels_[lev+1]->boxArray(),ratio_,0,1);
    auto const newCovered = amrex::makeFineMask(destination.boxArray(),destination.DistributionMap(),
        newFineBoxes,ratio_,0,1);
    amrex::MultiFab averaged(destination.boxArray(),destination.DistributionMap(),Descriptor::q,0);
    averaged.ParallelCopy(*restrictions_[lev],0,0,Descriptor::q,0,0);
    for (amrex::MFIter mfi(destination,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const out = destination.array(mfi);
        auto const in = averaged.const_array(mfi);
        auto const before = oldCovered.const_array(mfi);
        auto const after = newCovered.const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i,int j,int k,int q) {
                if (before(i,j,k) && !after(i,j,k)) out(i,j,k,q) = in(i,j,k,q);
            });
    }
    amrex::Gpu::streamSynchronize();
}

} // namespace lbm::amr
#endif
