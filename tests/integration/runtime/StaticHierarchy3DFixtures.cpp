#include "integration/runtime/StaticHierarchy3DFixtures.H"

namespace lbm::tests::static_hierarchy::d3 {
namespace {

using Interval = std::array<Real,2>;
constexpr Interval U{0,1}, C{0.25,0.75}, D{0.375,0.625}, W{0.875,1.25}, B{0,0.5};

amr::RefinementBox region(Interval x, Interval y, Interval z)
{ return {{x[0],y[0],z[0]},{x[1],y[1],z[1]}}; }

Fixture make(std::string name, bool perLevel, std::vector<amr::RefinementBox> const& regions,
             int boxSize = 16, int finestLevel = -1)
{
    if (finestLevel < 0) finestLevel = int(regions.size());
    Fixture result{std::move(name),base_spec(boxSize),finestLevel};
    if (perLevel) {
        for (std::size_t lev = 0; lev < regions.size(); ++lev)
            result.spec.regions.push_back({int(lev)+1,regions[lev]});
    } else result.spec.regions.push_back({finestLevel,regions.back()});
    return result;
}

void translate(Fixture& fixture, amrex::Array<Real,3> const& delta, bool moveDomain)
{
    for (auto& request : fixture.spec.regions) for (int d = 0; d < 3; ++d) {
        request.box.lo[d] += delta[d]; request.box.hi[d] += delta[d];
    }
    if (moveDomain) for (int d = 0; d < 3; ++d) {
        fixture.spec.probLo[d] += delta[d]; fixture.spec.probHi[d] += delta[d];
    }
}

} // namespace

Fixture central_fixture(bool perLevel, int levels, int boxSize)
{
    std::vector<amr::RefinementBox> regions{region(C,C,C)};
    if (levels == 3) regions.push_back(region(D,D,D));
    auto result = make(std::string(levels == 3 ? "D3-G3-" : "D3-G2-")+(perLevel ? "P" : "A"),
        perLevel,regions,boxSize);
    result.requireEmptyFineRank = levels == 2 && boxSize == 16;
    return result;
}

std::vector<Fixture> layout_fixtures(bool mpi)
{
    std::vector<Fixture> result;
    for (bool perLevel : {false,true}) {
        const std::string suffix = perLevel ? "-P" : "-A";
        result.push_back(central_fixture(perLevel,2));
        result.push_back(central_fixture(perLevel,3,8));
        if (!mpi) result.push_back(central_fixture(perLevel,3,16));
        auto vertex = make("D3-Vertex"+suffix,perLevel,{region(W,W,W)},8);
        vertex.requireCrossRank = true;
        result.push_back(vertex);
        if (!mpi && perLevel) {
            vertex.spec.maxGridSize = {amrex::IntVect(16)};
            result.push_back(vertex);
        }
        auto wrap3 = make("D3-Wrap3"+suffix,perLevel,
            {region(C,C,{0.75,1.25}),region(D,D,{0.875,1.125})},8);
        wrap3.requireCrossRank = true;
        result.push_back(wrap3);
        for (int axis = 0; axis < 3; ++axis) {
            const std::string label(1,"xyz"[axis]);
            std::array<Interval,3> bounds{C,C,C};
            bounds[axis] = U;
            auto coarseRegion = region(bounds[0],bounds[1],bounds[2]);
            result.push_back(make("D3-T2-"+label+suffix,perLevel,{coarseRegion}));
            bounds = {D,D,D}; bounds[axis] = U;
            auto through3 = make("D3-T3-"+label+suffix,perLevel,
                {coarseRegion,region(bounds[0],bounds[1],bounds[2])});
            through3.requireCrossRank = axis == 2;
            result.push_back(through3);
            if (!mpi && perLevel && axis == 2) {
                through3.spec.maxGridSize = {amrex::IntVect(8)};
                result.push_back(through3);
            }
            if (mpi) continue;
            bounds = {C,C,C}; bounds[axis] = W;
            result.push_back(make("D3-Face-"+label+suffix,perLevel,
                {region(bounds[0],bounds[1],bounds[2])}));
            bounds = {W,W,W}; bounds[axis] = C;
            const std::string edge = axis == 0 ? "yz" : axis == 1 ? "xz" : "xy";
            result.push_back(make("D3-Edge-"+edge+suffix,perLevel,
                {region(bounds[0],bounds[1],bounds[2])}));
            bounds = {U,U,U}; bounds[axis] = C;
            result.push_back(make("D3-Slab-"+label+suffix,perLevel,
                {region(bounds[0],bounds[1],bounds[2])}));
            bounds = {C,C,C}; bounds[axis] = B;
            result.push_back(make("D3-Side-"+label+suffix,perLevel,
                {region(bounds[0],bounds[1],bounds[2])}));
        }
        if (!mpi) {
            result.push_back(make("D3-Full"+suffix,perLevel,{region(U,U,U)}));
            result.push_back(make("D3-Side-corner"+suffix,perLevel,{region(B,B,B)}));
        }
    }
    if (mpi) return result;
    result.push_back(make("D3-small-A",false,
        {region({0.25,9.0/32},{0.25,9.0/32},{0.25,9.0/32})},16,2));
    auto shifted = central_fixture(true,3,8);
    shifted.name += "-shifted-origin";
    translate(shifted,{2,-1,3},true);
    result.push_back(shifted);
    const auto originals = result;
    for (auto const& original : originals) {
        if ((original.name != "D3-Vertex-P" || original.spec.maxGridSize[0][0] != 8) &&
            (original.name != "D3-T3-z-P" || original.spec.maxGridSize[0][0] != 16)) continue;
        for (int d = 0; d < 3; ++d) for (int sign : {-1,1}) {
            auto shiftedPeriod = original;
            shiftedPeriod.name += "-shift-"+std::string(1,"xyz"[d])+(sign < 0 ? "-minus" : "-plus");
            amrex::Array<Real,3> delta{}; delta[d] = sign;
            translate(shiftedPeriod,delta,false);
            result.push_back(std::move(shiftedPeriod));
        }
    }
    auto noncube = central_fixture(true,2);
    noncube.name = "D3-noncube-G2-P";
    noncube.spec.nCell = {16,32,24}; noncube.spec.probHi = {1,2,1.5};
    noncube.spec.regions = {{1,region(C,{0.5,1.5},{0.375,1.125})}};
    noncube.requireEmptyFineRank = false;
    result.push_back(noncube);
    noncube.name = "D3-noncube-Face-z-P";
    noncube.spec.regions = {{1,region(C,{0.5,1.5},{1.3125,1.875})}};
    result.push_back(noncube);
    return result;
}

} // namespace lbm::tests::static_hierarchy::d3
