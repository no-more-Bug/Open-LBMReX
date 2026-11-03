#ifndef LBM_TESTS_NUMERICAL_TANDEMCYLINDER_H
#define LBM_TESTS_NUMERICAL_TANDEMCYLINDER_H

#include "Re20Case.H"
#include <utility>

namespace amr_tandem {

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, bool crossLevel, int maxGridSize)
{
    auto spec = amr_re20::geometry(dimensions, "amr", maxGridSize);
    if (crossLevel) spec.regions.back().box.hi[0] = 0.3;
    return spec;
}

inline lbm::amr::AmrBoundarySpec boundaries(std::string const& scheme, bool shared)
{
    auto spec = amr_re20::boundaries(scheme);
    auto& first = spec.internal.front();
    first.name = "cylinder_a";
    first.objectName = shared ? "tandem" : "object_a";
    auto second = first;
    second.name = "cylinder_b";
    second.objectName = shared ? "tandem" : "object_b";
    second.centerPhys[0] = 0.4;
    spec.internal.push_back(std::move(second));
    return spec;
}

} // namespace amr_tandem
#endif
