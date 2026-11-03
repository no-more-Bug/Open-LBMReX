#ifndef LBM_TESTS_AMRDONOREVIDENCE_H
#define LBM_TESTS_AMRDONOREVIDENCE_H

#include "integration/runtime/AmrEvolutionTestCase.H"

namespace lbm::tests::amr_evolution {
// Enumerate global valid-source dependencies from the actual BA/DM once on IO rank.
// Counts describe copies/subvolumes, not MPI messages or ParallelCopy sends.
void write_donor_evidence(EvolutionView const& view, std::string const& sample,
                          std::string const& directory, bool requireRemote);
}
#endif
