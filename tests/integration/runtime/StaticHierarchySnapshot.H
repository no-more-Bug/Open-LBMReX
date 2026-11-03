#ifndef LBM_TESTS_STATICHIERARCHYSNAPSHOT_H
#define LBM_TESTS_STATICHIERARCHYSNAPSHOT_H

#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "lbm/amr/StaticRefinement.H"

namespace lbm::tests::static_hierarchy {

// Shared geometry identity for population snapshots and native plotfile oracles.
// Actual boxes describe every level, including automatically generated levels.
void write_static_identity(std::string const& directory, std::string const& sample,
    amr::StaticHierarchySpec const& spec, std::vector<Frame> const& frames,
    std::string const& profile, std::array<Real,2> references = {1.0,0.02});

void write_static_snapshot(std::string const& directory, std::string const& sample,
    amr::StaticHierarchySpec const& spec, std::vector<Frame> const& frames,
    std::string const& profile);

} // namespace lbm::tests::static_hierarchy

#endif
