#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTestSupport.H"

#include <AMReX_ParmParse.H>

namespace lbm::tests::static_hierarchy::topology {
namespace {

Fixture base(std::string const& name, int dimensions, int cells, int radius = 4, int margin = 4)
{
    Fixture f;
    f.name = "D"+std::to_string(dimensions)+"-Topo-"+name;
    f.spec.nCell = {cells,cells,dimensions == 3 ? cells : 1};
    f.spec.probHi = {1,1,dimensions == 3 ? 1.0 : 1.0/16};
    f.spec.refRatio = amrex::IntVect(2,2,dimensions == 3 ? 2 : 1);
    f.spec.maxGridSize = {amrex::IntVect(32,32,dimensions == 3 ? 32 : 1)};
    f.spec.gapFillRadiusCells = radius;
    f.spec.coarseNestingMarginCells = margin;
    return f;
}

void add(Fixture& f, int level, std::array<int,3> lo, std::array<int,3> hi)
{
    amr::RefinementBox box;
    for (int d = 0; d < 3; ++d) {
        const int n = f.spec.nCell[d]*(f.spec.refRatio[d] == 2 ? 1 << (level-1) : 1);
        const Real h = (f.spec.probHi[d]-f.spec.probLo[d])/n;
        box.lo[d] = f.spec.probLo[d]+lo[d]*h;
        box.hi[d] = f.spec.probLo[d]+hi[d]*h;
    }
    f.spec.regions.push_back({level,box});
}

void cube(Fixture& f, int level, int lo, int hi)
{ add(f,level,{lo,lo,f.spec.nCell[2] == 1 ? 0 : lo},{hi,hi,f.spec.nCell[2] == 1 ? 1 : hi}); }

} // namespace

std::vector<Fixture> fixtures(int dimensions, bool mpi)
{
    const bool d3 = dimensions == 3;
    std::vector<Fixture> result;
    auto addResult = [&](Fixture f) { result.push_back(std::move(f)); };
    auto f = base("Union",dimensions,d3 ? 32 : 64);
    if (d3) {
        cube(f,1,6,14); add(f,1,{10,8,8},{18,16,16});
    } else {
        cube(f,1,14,34); add(f,1,{28,26,0},{48,42,1});
    }
    f.spec.regions.push_back(f.spec.regions.front());
    cube(f,1,d3 ? 8 : 16,d3 ? 10 : 18);
    f.expectedFinestTagCells = d3 ? 880 : 672;
    addResult(f);

    for (int gap : {7,8,9}) {
        if (d3 && gap != 8) continue;
        f = base("Gap-"+std::to_string(gap),dimensions,40);
        add(f,1,{5,0,0},{6,40,d3 ? 40 : 1});
        add(f,1,{gap+6,0,0},{gap+7,40,d3 ? 40 : 1});
        f.expectedFinestTagCells = (gap <= 8 ? gap+2 : 2)*(d3 ? 1600 : 40);
        f.requireRemoteDonor = true;
        addResult(f);
    }
    for (int radius : {1,2}) {
        f = base("Gap-override-"+std::to_string(radius),dimensions,24,radius,radius);
        add(f,1,{5,0,0},{6,24,d3 ? 24 : 1});
        add(f,1,{6+2*radius,0,0},{7+2*radius,24,d3 ? 24 : 1});
        f.expectedFinestTagCells = (2*radius+2)*(d3 ? 576 : 24);
        addResult(f);
    }
    f = base("CrossBox",dimensions,32,1,1);
    cube(f,1,8,24);
    f.spec.maxGridSize = {amrex::IntVect(8,8,d3 ? 8 : 1),amrex::IntVect(32,32,d3 ? 32 : 1)};
    addResult(f);
    f = base("Separated",dimensions,64);
    cube(f,1,8,12); cube(f,1,36,40); addResult(f);
    for (bool bridge : {false,true}) {
        f = base(bridge ? "Batch-joined" : "Diagonal",dimensions,12,1,1);
        cube(f,1,2,3); cube(f,1,4,5);
        if (bridge) cube(f,1,3,4);
        if (!d3) f.expectedFinestTagCells = bridge ? 3 : 17;
        addResult(f);
    }
    if (d3) {
        f = base("Diagonal-edge",3,12,1,1);
        add(f,1,{2,2,4},{3,3,5}); add(f,1,{4,4,4},{5,5,5}); addResult(f);
    }
    f = base("Chain",dimensions,32,1,1);
    add(f,1,{4,4,d3 ? 4 : 0},{5,5,d3 ? 5 : 1});
    add(f,1,{6,6,d3 ? 4 : 0},{7,7,d3 ? 5 : 1});
    add(f,1,{10,6,d3 ? 4 : 0},{11,7,d3 ? 5 : 1}); addResult(f);
    f = base("Hole",dimensions,32,1,1);
    const int z0 = d3 ? 4 : 0, z1 = d3 ? 12 : 1;
    add(f,1,{4,4,z0},{12,7,z1}); add(f,1,{4,9,z0},{12,12,z1});
    add(f,1,{4,7,z0},{7,9,z1}); add(f,1,{9,7,z0},{12,9,z1}); addResult(f);
    f = base("Nonconvex",dimensions,32,1,1);
    add(f,1,{4,4,z0},{12,6,z1}); add(f,1,{4,4,z0},{6,12,z1});
    add(f,1,{14,4,z0},{15,6,z1}); add(f,1,{24,24,z0},{26,26,z1}); addResult(f);
    for (std::string const name : {"Finest","Sparse-L1-L3","Sparse-L2-L3","TightCoarse"}) {
        f = base(name,dimensions,32);
        if (name == "TightCoarse") { cube(f,1,15,17); cube(f,2,28,36); }
        else {
            if (name == "Sparse-L1-L3") cube(f,1,8,24);
            if (name == "Sparse-L2-L3") cube(f,2,24,40);
            cube(f,3,60,68);
        }
        f.requireRemoteDonor = name == "Sparse-L1-L3";
        addResult(f);
    }
    for (int crossed = 1; crossed <= dimensions; ++crossed) {
        f = base(crossed == dimensions ? (d3 ? "Vertex" : "Corner") : crossed == 1 ? "PeriodicFace" : "PeriodicEdge",
            dimensions,12,1,1);
        std::array<int,3> lo{4,4,d3 ? 4 : 0},hi{8,8,d3 ? 8 : 1};
        for (int d = 0; d < crossed; ++d) { lo[d] = 10; hi[d] = 15; }
        add(f,1,lo,hi);
        f.spec.maxGridSize = {amrex::IntVect(8,8,d3 ? 8 : 1)};
        f.requireRemoteDonor = true;
        if (d3 && crossed == 3) f.expectedFinestTagCells = 125;
        addResult(f);
    }
    if (!d3) {
        f = base("PeriodicFace-Seam",2,32,1,1);
        add(f,1,{28,8,0},{36,24,1}); addResult(f);
        f = base("Periodic-near",2,32,1,1);
        add(f,1,{1,0,0},{5,32,1}); add(f,1,{29,0,0},{31,32,1}); addResult(f);
    }
    for (int through = 1; through <= dimensions; ++through) {
        f = base(through == 1 ? (d3 ? "Through-z" : "Through-x") : through == dimensions ? "Through-all" : "Through-xy",
            dimensions,16,1,1);
        std::array<int,3> lo{4,4,d3 ? 4 : 0},hi{12,12,d3 ? 12 : 1};
        for (int n = 0; n < through; ++n) {
            const int d = through == 1 && d3 ? 2 : n;
            lo[d] = 0; hi[d] = 16;
        }
        add(f,1,lo,hi); f.requireRemoteDonor = true; addResult(f);
    }
    f = base("Side",dimensions,16,1,1);
    add(f,1,{0,4,d3 ? 4 : 0},{8,12,d3 ? 12 : 1}); addResult(f);
    f = base("Side-corner",dimensions,16,1,1); cube(f,1,0,8); addResult(f);
    f = base("MultiPeriod-default",dimensions,8); cube(f,1,2,3); addResult(f);
    f = base("MultiPeriod-wide",dimensions,2,8,8); cube(f,1,0,1); addResult(f);
    f = base("Feedback",dimensions,32,1,1);
    f.spec.gridEfficiency = 0.7;
    // The U fits its bounding rectangle at efficiency 224/288; filling that
    // rectangle reduces its gap to the independent upper block from 3 to 2.
    for (auto const& b : std::vector<std::array<int,4>>{{4,28,4,8},{4,12,8,16},{20,28,8,16},{15,17,18,20}})
        add(f,1,{b[0],b[2],d3 ? 8 : 0},{b[1],b[3],d3 ? 24 : 1});
    f.requireFeedback = true; addResult(f);

    if (mpi) {
        const std::vector<std::string> selected = d3
            ? std::vector<std::string>{"Vertex","Sparse-L1-L3","Feedback","Through-z","CrossBox"}
            : std::vector<std::string>{"Gap-8","Feedback","Through-x","CrossBox"};
        std::vector<Fixture> subset;
        for (auto const& candidate : result)
            for (auto const& name : selected)
                if (candidate.name == "D"+std::to_string(dimensions)+"-Topo-"+name) subset.push_back(candidate);
        return subset;
    }
    return result;
}

std::vector<Fixture> output_fixtures(int dimensions, bool mpi)
{
    const std::vector<std::string> names = dimensions == 3
        ? (mpi ? std::vector<std::string>{"Vertex","Sparse-L1-L3","Feedback"}
               : std::vector<std::string>{"Union","Vertex","Through-z","Sparse-L1-L3","Sparse-L2-L3","TightCoarse","Feedback"})
        : (mpi ? std::vector<std::string>{"Gap-8"}
               : std::vector<std::string>{"Gap-8","PeriodicFace-Seam","Finest"});
    std::vector<Fixture> result;
    for (auto const& candidate : fixtures(dimensions))
        for (auto const& name : names)
            if (candidate.name == "D"+std::to_string(dimensions)+"-Topo-"+name) result.push_back(candidate);
    return result;
}

std::vector<Fixture> supplement_fixtures(int dimensions, bool mpi)
{
    const bool d3 = dimensions == 3;
    const auto box = [=](int x0, int x1, int y0, int y1, int z0, int z1) {
        return amrex::Box(amrex::IntVect(x0,y0,d3 ? z0 : 0),
            amrex::IntVect(x1-1,y1-1,d3 ? z1-1 : 0));
    };
    std::vector<Fixture> result;
    for (bool explicitL2 : {false,true}) {
        if (mpi && !explicitL2) continue;
        auto f = base(explicitL2 ? "L3Separated-L2Overlap-AutoL1" : "L2Separated-AutoL1Merged",
            dimensions,32);
        if (explicitL2) {
            add(f,2,{16,24,d3 ? 24 : 0},{34,40,d3 ? 40 : 1});
            add(f,2,{30,24,d3 ? 24 : 0},{48,40,d3 ? 40 : 1});
            add(f,3,{40,56,d3 ? 56 : 0},{48,72,d3 ? 72 : 1});
            add(f,3,{72,56,d3 ? 56 : 0},{80,72,d3 ? 72 : 1});
            f.expectedCoverage = {
                {{box(0,32,0,32,0,32)},1},
                {{box(12,52,20,44,20,44)},1},
                {{box(32,96,48,80,48,80)},1},
                {{box(80,96,112,144,112,144),box(144,160,112,144,112,144)},2}};
        } else {
            add(f,2,{16,24,d3 ? 24 : 0},{24,40,d3 ? 40 : 1});
            add(f,2,{34,24,d3 ? 24 : 0},{42,40,d3 ? 40 : 1});
            f.expectedCoverage = {
                {{box(0,32,0,32,0,32)},1},
                {{box(12,46,20,44,20,44)},1},
                {{box(32,48,48,80,48,80),box(68,84,48,80,48,80)},2}};
        }
        if (!mpi) result.push_back(f);
        f.name += "-half-x";
        for (auto& request : f.spec.regions) {
            request.box.lo[0] += 0.5;
            request.box.hi[0] += 0.5;
        }
        for (std::size_t lev = 1; lev < f.expectedCoverage.size(); ++lev)
            for (auto& expected : f.expectedCoverage[lev].boxes)
                expected.shift(0,16*(1 << lev));
        result.push_back(std::move(f));
    }
    return result;
}

std::vector<Fixture> supplement_output_fixtures(bool mpi)
{
    std::vector<Fixture> result;
    if (!mpi) for (auto const& f : fixtures(3))
        if (f.name == "D3-Topo-Gap-8") result.push_back(f);
    for (auto const& f : supplement_fixtures(3))
        if (f.name.find("-half-x") == std::string::npos &&
            (!mpi || f.name == "D3-Topo-L3Separated-L2Overlap-AutoL1")) result.push_back(f);
    return result;
}

amr::StaticHierarchySpec read_spec_inputs()
{
    amrex::ParmParse pp("static_test");
    amr::StaticHierarchySpec spec;
    amrex::Vector<int> cells,periodic,ratio,grid,levels;
    amrex::Vector<Real> lo,hi;
    pp.getarr("n_cell",cells); pp.getarr("is_periodic",periodic);
    pp.getarr("ref_ratio",ratio); pp.getarr("max_grid_size",grid);
    pp.getarr("prob_lo",lo); pp.getarr("prob_hi",hi);
    require(cells.size() == 3 && periodic.size() == 3 && ratio.size() == 3 &&
        grid.size() == 3 && lo.size() == 3 && hi.size() == 3,"test input directional lengths");
    for (int d = 0; d < 3; ++d) {
        spec.nCell[d] = cells[d]; spec.isPeriodic[d] = periodic[d]; spec.refRatio[d] = ratio[d];
        spec.probLo[d] = lo[d]; spec.probHi[d] = hi[d];
    }
    spec.maxGridSize = {amrex::IntVect(grid[0],grid[1],grid[2])};
    pp.get("gap_fill_radius_cells",spec.gapFillRadiusCells);
    pp.get("coarse_nesting_margin_cells",spec.coarseNestingMarginCells);
    pp.get("grid_eff",spec.gridEfficiency);
    pp.getarr("region_levels",levels);
    for (std::size_t n = 0; n < levels.size(); ++n) {
        pp.getarr(("region_"+std::to_string(n)+"_lo").c_str(),lo);
        pp.getarr(("region_"+std::to_string(n)+"_hi").c_str(),hi);
        require(lo.size() == 3 && hi.size() == 3,"test input region coordinates");
        spec.regions.push_back({levels[n],{{lo[0],lo[1],lo[2]},{hi[0],hi[1],hi[2]}}});
    }
    return spec;
}

} // namespace lbm::tests::static_hierarchy::topology
