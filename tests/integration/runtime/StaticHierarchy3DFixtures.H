#ifndef LBM_TESTS_STATICHIERARCHY3DFIXTURES_H
#define LBM_TESTS_STATICHIERARCHY3DFIXTURES_H

#include "integration/runtime/StaticHierarchy3DTestSupport.H"

namespace lbm::tests::static_hierarchy::d3 {

struct Fixture {
    std::string name;
    amr::StaticHierarchySpec spec;
    int finestLevel = 0;
    bool requireCrossRank = false;
    bool requireEmptyFineRank = false;
};

std::vector<Fixture> layout_fixtures(bool mpi);
Fixture central_fixture(bool perLevel, int levels = 3, int boxSize = 16);

} // namespace lbm::tests::static_hierarchy::d3
#endif
