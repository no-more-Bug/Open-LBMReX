#ifndef LBM_AMR_COARSEFINECOUPLING_HH
#define LBM_AMR_COARSEFINECOUPLING_HH

#include "lbm/profiling/TimestepProfiler.H"

#include "lbm/amr/StaticRegionTopology.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFabUtil.H>

namespace lbm::amr {

template<class Descriptor>
CoarseFineCoupling<Descriptor>::CoarseFineCoupling(
    amrex::Geometry const& coarseGeometry, Field const& coarse,
    amrex::Geometry const& fineGeometry, Field const& fine,
    amrex::IntVect const& ratio)
    : ratio_(ratio), reach_(stencil_extent<Descriptor>()),
      coarsePeriodicity_(coarseGeometry.periodicity()),
      finePeriodicity_(fineGeometry.periodicity())
{
    AMREX_ALWAYS_ASSERT(fine.nGrowVect().allGE(2 * reach_));
    auto const& fba = fine.multifab().boxArray();
    auto const& cba = coarse.multifab().boxArray();
    auto const& fdm = fine.multifab().DistributionMap();
    auto const& cdm = coarse.multifab().DistributionMap();

    amrex::BoxList coarseSources;
    amrex::Vector<int> fineOwners;
    for (int b = 0; b < fba.size(); ++b) {
        for (auto const& piece : fba.complementIn(amrex::grow(fba[b], 2 * reach_), finePeriodicity_)) {
            fineTargets_.push_back({piece, b});
            coarseSources.push_back(amrex::coarsen(piece, ratio_));
            fineOwners.push_back(fdm[b]);
        }
    }
    // Different target FABs need their own copies, even when these overlap.
    if (!fineTargets_.empty()) {
        coarseDonor_.define(amrex::BoxArray(std::move(coarseSources)),
                            amrex::DistributionMapping(std::move(fineOwners)), Descriptor::q, 0);
    }

    auto const covered = detail::coarse_cells(fba, ratio_);
    auto const grown = detail::periodic_grow(covered, coarseGeometry.Domain(), coarseGeometry.periodicity(), Descriptor::d, 1);
    amrex::BoxList targets;
    amrex::Vector<int> coarseOwners;
    for (int b = 0; b < grown.size(); ++b) {
        for (auto const& piece : covered.complementIn(grown[b])) {
            for (auto const& hit : cba.intersections(piece)) {
                targets.push_back(hit.second);
                coarseFabs_.push_back(hit.first);
                coarseOwners.push_back(cdm[hit.first]);
            }
        }
    }
    if (!coarseFabs_.empty()) {
        amrex::BoxArray targetBoxes(std::move(targets));
        amrex::DistributionMapping owners(std::move(coarseOwners));
        coverageMask_ = amrex::makeFineMask(targetBoxes, owners, reach_, fba, ratio_,
                                           coarsePeriodicity_, 0, 1);
        targetBoxes.refine(ratio_);
        fineDonor_.define(targetBoxes, owners, Descriptor::q, reach_);
    }
}

template<class Descriptor>
void CoarseFineCoupling<Descriptor>::explode(Field const& coarseStar, Field& fineFirstSource,
                                            CoarseFineTimingSample* timing)
{
    if (timing) *timing = {};
    if (fineTargets_.empty()) return; // Global geometry, never local rank occupancy.
    auto const communication = profiling::TimestepProfiler::measureOperation(timing != nullptr, [&] {
        coarseDonor_.ParallelCopy(coarseStar.multifab(), 0, 0, Descriptor::q,
                                 amrex::IntVect(0), amrex::IntVect(0),
                                 coarsePeriodicity_, amrex::FabArrayBase::COPY);
    });
    auto const computation = profiling::TimestepProfiler::measureOperation(timing != nullptr, [&] {
    auto const ratio = ratio_;
    for (amrex::MFIter mfi(coarseDonor_); mfi.isValid(); ++mfi) {
        auto const& target = fineTargets_[mfi.index()];
        auto const src = coarseDonor_.const_array(mfi);
        auto const dst = fineFirstSource.multifab().array(target.fab);
        amrex::ParallelFor(target.cells, Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int q) {
                dst(i,j,k,q) = src(amrex::coarsen(i,ratio[0]), amrex::coarsen(j,ratio[1]),
                                   amrex::coarsen(k,ratio[2]), q);
            });
    }
    });
    if (timing) *timing = {communication, computation, true};
}

template<class Descriptor>
void CoarseFineCoupling<Descriptor>::coalesce(Field const& fineSecondSource, Field& coarseNext,
                                             CoarseFineTimingSample* timing)
{
    if (timing) *timing = {};
    if (coarseFabs_.empty()) return;
    auto const communication = profiling::TimestepProfiler::measureOperation(timing != nullptr, [&] {
        fineDonor_.ParallelCopy(fineSecondSource.multifab(), 0, 0, Descriptor::q,
                               reach_, reach_, finePeriodicity_, amrex::FabArrayBase::COPY);
    });
    auto const computation = profiling::TimestepProfiler::measureOperation(timing != nullptr, [&] {
    auto const ratio = ratio_;
    int const volumes = ratio[0] * ratio[1] * ratio[2];
    for (amrex::MFIter mfi(coverageMask_, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const covered = coverageMask_.const_array(mfi);
        auto const src = fineDonor_.const_array(mfi.index());
        auto const dst = coarseNext.multifab().array(coarseFabs_[mfi.index()]);
        amrex::ParallelFor(mfi.tilebox(), Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int q) {
                amrex::Real incoming = 0;
                int ordinary = 0;
                for (int z = 0; z < ratio[2]; ++z)
                for (int y = 0; y < ratio[1]; ++y)
                for (int x = 0; x < ratio[0]; ++x) {
                    int const sx = ratio[0]*i + x - Descriptor::cx(q);
                    int const sy = ratio[1]*j + y - Descriptor::cy(q);
                    int const sz = ratio[2]*k + z - Descriptor::cz(q);
                    int const rx = sx - Descriptor::cx(q);
                    int const ry = sy - Descriptor::cy(q);
                    int const rz = sz - Descriptor::cz(q);
                    if (covered(amrex::coarsen(sx,ratio[0]), amrex::coarsen(sy,ratio[1]),
                                amrex::coarsen(sz,ratio[2])) ||
                        covered(amrex::coarsen(rx,ratio[0]), amrex::coarsen(ry,ratio[1]),
                                amrex::coarsen(rz,ratio[2]))) {
                        incoming += src(sx,sy,sz,q);
                    } else {
                        ++ordinary;
                    }
                }
                if (ordinary == volumes) return;
                if (ordinary != 0) incoming += ordinary * dst(i,j,k,q);
                dst(i,j,k,q) = incoming / volumes;
            });
    }
    });
    if (timing) *timing = {communication, computation, true};
}

} // namespace lbm::amr
#endif
