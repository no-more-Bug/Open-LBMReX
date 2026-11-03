#ifndef LBM_TESTS_STATICHIERARCHYTOPOLOGYFIXTURES_H
#define LBM_TESTS_STATICHIERARCHYTOPOLOGYFIXTURES_H

#include "lbm/amr/StaticRefinement.H"
#include <AMReX_Box.H>
#include <string>
#include <vector>

namespace lbm::tests::static_hierarchy::topology {

struct ExpectedLevelCoverage {
    std::vector<amrex::Box> boxes; // hand-derived valid-cell coordinates; periodic images allowed
    int components = 1;
};

struct Fixture {
    std::string name;
    amr::StaticHierarchySpec spec;
    bool requireRemoteDonor = false;
    bool requireFeedback = false;
    long expectedFinestTagCells = -1; // only the hand-counted discrete examples
    std::vector<ExpectedLevelCoverage> expectedCoverage;
};

std::vector<Fixture> fixtures(int dimensions, bool mpi = false);
std::vector<Fixture> output_fixtures(int dimensions, bool mpi = false);
// Kept separate so the original validation matrix retains its sample identity.
std::vector<Fixture> supplement_fixtures(int dimensions, bool mpi = false);
std::vector<Fixture> supplement_output_fixtures(bool mpi = false);
// A tests-only adapter for the same one-source Spec used by direct composition.
amr::StaticHierarchySpec read_spec_inputs();

} // namespace lbm::tests::static_hierarchy::topology
#endif
