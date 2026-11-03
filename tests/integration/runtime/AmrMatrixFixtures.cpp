#include "integration/runtime/AmrMatrixFixtures.H"
#include "integration/runtime/StaticHierarchy2DFixtures.H"
#include "integration/runtime/StaticHierarchy3DFixtures.H"

namespace lbm::tests::amr_evolution {
namespace {
std::string prefix() { return "D"+std::to_string(Descriptor::d)+"-"; }

Fixture adapt(std::string name, amr::StaticHierarchySpec spec)
{ return {std::move(name),std::move(spec)}; }
}

Fixture through_fixture(int box)
{
    if constexpr (Descriptor::d==2) {
        auto source=static_hierarchy::d2_layout::through_x(true,3);
        source.spec.maxGridSize={amrex::IntVect(box,box,1)};
        auto result=adapt(prefix()+source.name+"-box"+std::to_string(box),std::move(source.spec));
        result.requireRemoteDonor=box==8;
        return result;
    } else {
        for (auto const& source : static_hierarchy::d3::layout_fixtures(false))
            if (source.name=="D3-T3-z-P" && source.spec.maxGridSize.front()[0]==box) {
                auto result=adapt(source.name+"-box"+std::to_string(box),source.spec);
                result.requireRemoteDonor=box==8;
                return result;
            }
        amrex::Abort("missing D3 through-z fixture");
        return {};
    }
}

Fixture retained_hole_fixture()
{
    bool const d3=Descriptor::d==3;
    Fixture result;
    result.name=prefix()+"Retained-hole";
    auto& s=result.spec;
    s.nCell={32,32,d3 ? 32 : 1}; s.probHi={1,1,d3 ? 1 : Real(1)/16};
    s.refRatio=amrex::IntVect(2,2,d3 ? 2 : 1);
    s.maxGridSize={amrex::IntVect(8,8,d3 ? 8 : 1)};
    s.gapFillRadiusCells=s.coarseNestingMarginCells=1;
    // Nonoverlapping slabs of [4,16)^d minus [8,12)^d, in L0 face indices.
    std::vector<amrex::Box> pieces;
    auto slab=[&](int x0,int x1,int y0,int y1,int z0,int z1) {
        pieces.emplace_back(amrex::IntVect(x0,y0,d3 ? z0 : 0),
                            amrex::IntVect(x1-1,y1-1,d3 ? z1-1 : 0));
    };
    slab(4,8,4,16,4,16); slab(12,16,4,16,4,16);
    slab(8,12,4,8,4,16); slab(8,12,12,16,4,16);
    if (d3) { slab(8,12,8,12,4,8); slab(8,12,8,12,12,16); }
    result.expectedCoverage.push_back({{amrex::Box(amrex::IntVect(0),amrex::IntVect(31,31,d3 ? 31 : 0))},1});
    std::vector<amrex::Box> fine;
    for (auto const& b : pieces) {
        amr::RefinementBox physical;
        for (int d=0; d<3; ++d) {
            Real const h=s.probHi[d]/s.nCell[d];
            physical.lo[d]=b.smallEnd(d)*h;
            physical.hi[d]=(b.bigEnd(d)+1)*h;
        }
        s.regions.push_back({1,physical});
        fine.push_back(amrex::refine(b,s.refRatio));
    }
    result.expectedCoverage.push_back({std::move(fine),1});
    result.expectedFinestTagCells=d3 ? 1664 : 128;
    return result;
}

namespace {
Fixture narrow_gap_fixture()
{
    auto result=central_fixture(1,8);
    result.name=prefix()+"Narrow-gap";
    auto& spec=result.spec;
    int const n=spec.nCell[0];
    std::vector<amrex::Box> fine;
    // Three coarse cells remain between two periodic, through-domain bands.
    // Radius-one closing would fill a two-cell gap, so that old fixture is insufficient.
    for (int lo : {1,8}) {
        spec.regions.push_back({1,{{Real(lo)/n,0,0},{Real(lo+4)/n,1,spec.probHi[2]}}});
        fine.emplace_back(amrex::IntVect(2*lo,0,0),
            amrex::IntVect(2*(lo+4)-1,2*n-1,Descriptor::d==3 ? 2*n-1 : 0));
    }
    result.expectedCoverage.push_back({std::move(fine),2});
    return result;
}

} // namespace

std::vector<Fixture> matrix_fixtures()
{
    std::vector<Fixture> result;
    for (auto const& f : static_hierarchy::topology::supplement_fixtures(Descriptor::d,false))
        if (f.name==prefix()+"Topo-L2Separated-AutoL1Merged") result.push_back(f);
    if constexpr (Descriptor::d==2) {
        auto source=static_hierarchy::d2_layout::corner(true,8);
        result.push_back(adapt(prefix()+source.name+"-box8",std::move(source.spec)));
    } else {
        for (auto const& source : static_hierarchy::d3::layout_fixtures(false)) {
            if (source.name=="D3-Vertex-P" || source.name=="D3-Wrap3-P")
                result.push_back(adapt(source.name+"-box"+std::to_string(source.spec.maxGridSize.front()[0]),source.spec));
        }
    }
    result.push_back(through_fixture(Descriptor::d==2 ? 32 : 16));
    result.push_back(through_fixture(8));
    for (auto const& f : static_hierarchy::topology::fixtures(Descriptor::d,false))
        if (f.name==prefix()+"Topo-Nonconvex" || f.name==prefix()+"Topo-Through-all") result.push_back(f);
    result.push_back(narrow_gap_fixture());
    result.push_back(retained_hole_fixture());
    auto const remoteThrough=through_fixture(8).name;
    for (auto& fixture : result) {
        // The old static fixtures use this flag for same-level donors; C6 selects CF origins.
        fixture.requireRemoteDonor=fixture.name==remoteThrough ||
            (fixture.name.find("box8")!=std::string::npos &&
             (fixture.name.find("corner")!=std::string::npos || fixture.name.find("Vertex")!=std::string::npos ||
              fixture.name.find("Wrap3")!=std::string::npos));
    }
    return result;
}

} // namespace lbm::tests::amr_evolution
