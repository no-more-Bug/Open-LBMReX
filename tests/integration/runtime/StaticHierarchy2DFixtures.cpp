#include "integration/runtime/StaticHierarchy2DFixtures.H"
#include <utility>

namespace lbm::tests::static_hierarchy::d2_layout {
using Real = amrex::Real;
using Spec = amr::StaticHierarchySpec;
using Rectangle = amr::RefinementBox;

Rectangle rectangle(Real xlo, Real xhi, Real ylo, Real yhi)
{ return {{xlo,ylo,0.0},{xhi,yhi,1.0/16.0}}; }

Sample sample(std::string name, bool perLevel, std::vector<Rectangle> const& regions,
              int maxGridSize, int finestLevel)
{
    if (finestLevel < 0) finestLevel = int(regions.size());
    Sample result{std::move(name), {}, finestLevel};
    result.spec.maxGridSize = {amrex::IntVect(maxGridSize,maxGridSize,1)};
    result.spec.gapFillRadiusCells = 1;
    result.spec.coarseNestingMarginCells = 1;
    if (perLevel) {
        for (std::size_t lev = 0; lev < regions.size(); ++lev)
            result.spec.regions.push_back({int(lev)+1,regions[lev]});
    } else result.spec.regions.push_back({finestLevel,regions.back()});
    return result;
}

Sample g2(bool perLevel)
{
    return sample(perLevel ? "G2-P" : "G2-A", perLevel,
        {rectangle(0.25,0.75,0.25,0.75)});
}

Sample g3(bool perLevel)
{
    return sample(perLevel ? "G3-P" : "G3-A", perLevel,
        {rectangle(0.25,0.75,0.25,0.75), rectangle(0.375,0.625,0.375,0.625)});
}

Sample corner(bool perLevel, int maxGridSize)
{
    return sample(perLevel ? "P-corner-P" : "P-corner-A", perLevel,
        {rectangle(7.0/8.0,5.0/4.0,15.0/16.0,9.0/8.0)}, maxGridSize);
}

Sample p3(bool perLevel)
{
    return sample(perLevel ? "P3-P" : "P3-A", perLevel,
        {rectangle(0.75,11.0/8.0,0.25,0.75), rectangle(7.0/8.0,1.25,0.375,0.625)}, 8);
}

Sample through_x(bool perLevel, int levels)
{
    std::vector<Rectangle> regions{rectangle(0,1,0.25,0.75)};
    if (levels == 3) {
        regions.push_back(rectangle(0,1,0.375,0.625));
    }
    auto result = sample(std::string(levels == 2 ? "P-through-x-" : "P-through-3-x-") +
        (perLevel ? "P" : "A"), perLevel, std::move(regions));
    result.checkThroughX = true;
    return result;
}

void translate_regions(Spec& spec, Real x, Real y, Real z)
{
    auto translate = [=](Rectangle& r) {
        r.lo[0] += x; r.hi[0] += x; r.lo[1] += y; r.hi[1] += y;
        r.lo[2] += z; r.hi[2] += z;
    };
    for (auto& request : spec.regions) translate(request.box);
}

} // namespace lbm::tests::static_hierarchy::d2_layout
