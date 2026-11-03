#ifndef LBM_AMR_STATICREGIONTOPOLOGY_H
#define LBM_AMR_STATICREGIONTOPOLOGY_H

#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_Periodicity.H>
#include <vector>

namespace lbm::amr::detail {

enum RepairReason : unsigned { GapFill = 1, CoarseNesting = 2, AmrexCoverage = 4 };

amrex::BoxArray cell_union(amrex::BoxList boxes);
amrex::BoxArray cell_union(amrex::BoxArray const& a, amrex::BoxArray const& b);
amrex::BoxArray coarse_cells(amrex::BoxArray const& cells, amrex::IntVect const& ratio);
bool covers(amrex::BoxArray const& owners, amrex::BoxArray const& needed);

// Normalize periodic directions and clip nonperiodic growth to the domain.
amrex::BoxArray periodic_grow(amrex::BoxArray const& cells, amrex::Box const& domain,
                              amrex::Periodicity const& periodicity, int activeDimensions, long long radius);

void repair_regions(std::vector<amrex::BoxArray>& cells, amrex::Box const& coarseDomain,
                    amrex::IntVect const& ratio, amrex::Periodicity const& periodicity, int activeDimensions,
                    int gapRadius, int nestingMargin, std::vector<unsigned>& reasons);

} // namespace lbm::amr::detail
#endif
