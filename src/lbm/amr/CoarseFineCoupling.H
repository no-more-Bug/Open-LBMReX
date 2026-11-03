#ifndef LBM_AMR_COARSEFINECOUPLING_H
#define LBM_AMR_COARSEFINECOUPLING_H

#include "lbm/fields/LatticeField.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_Periodicity.H>
#include <vector>

namespace lbm::amr {

struct CoarseFineTimingSample {
    amrex::Real communication = 0;
    amrex::Real computation = 0;
    bool performed = false;
};

/** Scheme-B exchange for one adjacent pair during a fixed-layout interval.
 * Reconstruct this instance when either level changes BoxArray (including FAB
 * order) or DistributionMapping. Donor values are refreshed by each exchange.
 * Layouts obey the prepared 2:1 alignment/nesting contract. Every call borrows
 * fields with those layouts; no population pointer survives a call or swap.
 * All ranks participate, including ranks without fine valid cells.
 */
template<class Descriptor>
class CoarseFineCoupling {
public:
    using Field = LatticeField<Descriptor>;

    CoarseFineCoupling(amrex::Geometry const& coarseGeometry, Field const& coarse,
                      amrex::Geometry const& fineGeometry, Field const& fine,
                      amrex::IntVect const& ratio);

    // C* -> every required copy of G0. The caller fills same-level ghosts.
    void explode(Field const& coarseStar, Field& fineFirstSource, CoarseFineTimingSample* timing = nullptr);

    // Read F1*/preserved G1 after the second same-level fill. Update only
    // uncovered coarse interface targets, using their existing Ctmp shares.
    void coalesce(Field const& fineSecondSource, Field& coarseNext, CoarseFineTimingSample* timing = nullptr);

private:
    struct FineTarget { amrex::Box cells; int fab; };
    amrex::IntVect ratio_, reach_;
    amrex::Periodicity coarsePeriodicity_, finePeriodicity_;
    std::vector<FineTarget> fineTargets_;
    std::vector<int> coarseFabs_;
    amrex::MultiFab coarseDonor_, fineDonor_;
    amrex::iMultiFab coverageMask_;
};

} // namespace lbm::amr

#include "lbm/amr/CoarseFineCoupling.HH"
#endif
