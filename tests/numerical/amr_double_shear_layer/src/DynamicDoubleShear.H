#ifndef LBM_TESTS_NUMERICAL_DYNAMICDOUBLESHEAR_H
#define LBM_TESTS_NUMERICAL_DYNAMICDOUBLESHEAR_H

#include "DoubleShear.H"
#include "lbm/amr/DynamicRefinement.H"

namespace double_shear {
inline lbm::amr::DynamicHierarchySpec dynamic_geometry(int dimensions, int maxLevel, int maxBox)
{
    lbm::amr::DynamicHierarchySpec spec;
    static_cast<lbm::amr::HierarchyGeometrySpec&>(spec) = base_geometry(dimensions,128,maxBox);
    spec.maxLevel = maxLevel;
    return spec;
}
inline lbm::amr::DynamicHierarchySpec fixed_band_geometry(int dimensions, int maxBox)
{
    auto spec = dynamic_geometry(dimensions,2,maxBox);
    auto band = lbm::amr::RefinementBox{spec.probLo,spec.probHi};
    band.lo[1] = 0.65;
    band.hi[1] = 0.85;
    spec.constraints = {{band,1,2}};
    return spec;
}
} // namespace double_shear
#endif
