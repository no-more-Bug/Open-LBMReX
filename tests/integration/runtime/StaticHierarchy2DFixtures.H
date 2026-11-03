#ifndef LBM_TESTS_STATICHIERARCHY2DFIXTURES_H
#define LBM_TESTS_STATICHIERARCHY2DFIXTURES_H

#include "lbm/amr/StaticRefinement.H"
#include <string>
#include <vector>

namespace lbm::tests::static_hierarchy::d2_layout {

struct Sample {
    std::string name;
    amr::StaticHierarchySpec spec;
    int finestLevel = 0;
    bool checkThroughX = false;
};

amr::RefinementBox rectangle(amrex::Real xlo, amrex::Real xhi, amrex::Real ylo, amrex::Real yhi);
Sample sample(std::string name, bool perLevel, std::vector<amr::RefinementBox> const& regions,
              int maxGridSize = 32, int finestLevel = -1);
Sample g2(bool perLevel);
Sample g3(bool perLevel);
Sample corner(bool perLevel, int maxGridSize = 8);
Sample p3(bool perLevel);
Sample through_x(bool perLevel, int levels);
void translate_regions(amr::StaticHierarchySpec& spec, amrex::Real x, amrex::Real y, amrex::Real z = 0);

} // namespace lbm::tests::static_hierarchy::d2_layout
#endif
