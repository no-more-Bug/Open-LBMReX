#ifndef LBM_AMR_POPULATIONREMAP_H
#define LBM_AMR_POPULATIONREMAP_H

#include <AMReX_MultiFab.H>

#include <memory>
#include <vector>

namespace lbm::amr {

/** Full-population migration at a single synchronized time.
 * Caller retains the borrowed old valid-population fields through all calls.
 * Stores only covered-coarse restrictions. One temporary intermediate fine field
 * is used while cascading finer information; the old finest field is never copied.
 */
template<class Descriptor>
class PopulationRemap {
public:
    PopulationRemap(std::vector<amrex::MultiFab const*> oldLevels, amrex::IntVect ratio);

    // Destination is new storage. coarseNew is required above L0 and already remapped.
    void fillNewLevel(int lev, amrex::MultiFab& destination, amrex::MultiFab const* coarseNew,
                      amrex::BoxArray const& newFineBoxes) const;
    // Also called for unchanged retained layouts, especially L0.
    void backfillUncovered(int lev, amrex::MultiFab& destination,
                           amrex::BoxArray const& newFineBoxes) const;

private:
    std::vector<amrex::MultiFab const*> oldLevels_;
    amrex::IntVect ratio_;
    std::vector<std::unique_ptr<amrex::MultiFab>> restrictions_;
};

} // namespace lbm::amr
#include "lbm/amr/PopulationRemap.HH"
#endif
