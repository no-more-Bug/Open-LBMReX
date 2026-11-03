#ifndef LBM_TESTS_STATICHIERARCHYTOPOLOGYORACLE_H
#define LBM_TESTS_STATICHIERARCHYTOPOLOGYORACLE_H

#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "lbm/amr/StaticRefinement.H"

namespace lbm::tests::static_hierarchy::topology {

struct Fixture;
// Tests-owned periodic Boolean-cell oracle, independent of production Box algebra.
void validate_coverage(amr::StaticHierarchySpec const& spec,
    std::vector<Frame> const& frames, int dimensions);
// Supplement fixtures use complete hand-derived sets instead of Boolean repair.
void validate_coverage(Fixture const& fixture, std::vector<Frame> const& frames, int dimensions);

} // namespace lbm::tests::static_hierarchy::topology
#endif
