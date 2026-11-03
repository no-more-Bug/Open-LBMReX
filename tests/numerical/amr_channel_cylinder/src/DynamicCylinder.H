#ifndef LBM_TESTS_NUMERICAL_DYNAMICCYLINDER_H
#define LBM_TESTS_NUMERICAL_DYNAMICCYLINDER_H

#include "TandemCylinder.H"
#include "lbm/amr/DynamicRefinement.H"
#include <set>

namespace dynamic_cylinder {
using amr_re20::Real;
using amr_re20::Initial;
using amr_re20::meanSpeed;
constexpr int fullSteps = 200000;

inline lbm::amr::DynamicHierarchySpec geometry(int dimensions, std::string const& variant, int maxBox)
{
    lbm::amr::DynamicHierarchySpec spec;
    static_cast<lbm::amr::HierarchyGeometrySpec&>(spec) = amr_re20::geometry(dimensions,"amr",maxBox);
    spec.maxLevel = 2;
    auto region = [](Real x0, Real x1, Real y0, Real y1) {
        return lbm::amr::RefinementBox{{x0,y0,0},{x1,y1,Real(.01)}};
    };
    if (variant == "tandem-a")
        spec.constraints.push_back({region(.13,.47,.13,.27),2,2});
    else {
        spec.constraints.push_back({region(.13,.27,.13,.27),2,2});
        if (variant == "tandem-b")
            spec.constraints.push_back({region(.33,.47,.13,.27),2,2});
    }
    // Four outer L0 valid cells: Guo physical faces lie at their outer centers.
    Real const guard = 4*Real(.005);
    auto const& lo = spec.probLo;
    auto const& hi = spec.probHi;
    spec.constraints.push_back({region(lo[0],lo[0]+guard,lo[1],hi[1]),0,0});
    spec.constraints.push_back({region(hi[0]-guard,hi[0],lo[1],hi[1]),0,0});
    spec.constraints.push_back({region(lo[0],hi[0],lo[1],lo[1]+guard),0,0});
    spec.constraints.push_back({region(lo[0],hi[0],hi[1]-guard,hi[1]),0,0});
    return spec;
}

inline lbm::amr::AmrBoundarySpec boundaries(std::string const& scheme, std::string const& variant)
{
    auto result = amr_tandem::boundaries(scheme,false);
    if (variant == "single") result.internal.resize(1);
    return result;
}

inline std::set<int> sample_steps(int last)
{
    std::set<int> result{0,last};
    for (int step = 0; step <= last; step += 4000) result.insert(step);
    for (int step = 160000; step <= last; step += 400) result.insert(step);
    return result;
}
} // namespace dynamic_cylinder
#endif
