#ifndef LBM_TESTS_DYNAMICCONSTRAINTGRIDTESTS_HH
#define LBM_TESTS_DYNAMICCONSTRAINTGRIDTESTS_HH

#include "integration/runtime/DynamicConstraintGridTests.H"
#include "integration/runtime/DynamicConstraintTests.HH"
#include "integration/runtime/DynamicGridClusteringTests.HH"

#include <algorithm>

namespace lbm::tests::dynamic_amr::constraint_grids {
using Grids = std::vector<amrex::BoxArray>;
using Tags = std::vector<amrex::Gpu::PinnedVector<amrex::IntVect>>;

template<class Descriptor>
amr::DynamicHierarchySpec spec_for(int maxLevel = 2, int n = 32)
{
    auto spec = base_spec<Descriptor>(maxLevel);
    spec.nCell = {n,n,Descriptor::d == 3 ? 4 : 1};
    spec.probHi = {1,1,Descriptor::d == 3 ? Real(4)/n : Real(0.125)};
    spec.maxGridSize = {amrex::IntVect(16,16,Descriptor::d == 3 ? 4 : 1)};
    spec.gapFillRadiusCells = 1;
    spec.coarseNestingMarginCells = 2;
    spec.tagBufferCells = {amrex::IntVect(0)};
    spec.constraints = {{{spec.probLo,spec.probHi},0,maxLevel}};
    return spec;
}

// Integer coordinates here are physical box faces on L0. They are the test
// recipe; expected result predicates below are independent integer cell sets.
inline amr::RefinementBox rectangle(amr::DynamicHierarchySpec const& spec,
                                    int xLo, int xHi, int yLo, int yHi)
{
    amr::RefinementBox box{spec.probLo,spec.probHi};
    box.lo[0] = Real(xLo)/spec.nCell[0]; box.hi[0] = Real(xHi)/spec.nCell[0];
    box.lo[1] = Real(yLo)/spec.nCell[1]; box.hi[1] = Real(yHi)/spec.nCell[1];
    return box;
}

inline amrex::IntVect shape(amr::DynamicHierarchyLayout const& layout, int lev)
{
    auto result = layout.geometry.Domain().length();
    for (int l = 0; l < lev; ++l) result *= layout.mesh.ref_ratio[l];
    return result;
}

inline amrex::Box column(amrex::IntVect n, int xLo, int xHi, int yLo, int yHi)
{
    return {{xLo,yLo,0},{xHi-1,yHi-1,n[2]-1}};
}

inline Grids candidates(amr::DynamicHierarchyLayout const& layout, bool split = false)
{
    Grids result;
    for (int lev = 0; lev <= layout.mesh.max_level; ++lev) {
        result.emplace_back(amrex::Box(amrex::IntVect(0),shape(layout,lev)-amrex::IntVect(1)));
        if (split) result.back().maxSize(layout.mesh.max_grid_size[lev]);
    }
    return result;
}

inline void append(Tags& tags, int source, amrex::Box const& cells)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    for (amrex::BoxIterator cell(cells); cell.ok(); ++cell) tags[source].push_back(cell());
}

inline Grids finish(Grids const& candidate, amr::DynamicHierarchyLayout const& layout,
                     Tags const& tags, std::string const& label)
{
    auto result = amr::finalize_dynamic_grids(candidate,layout,tags);
    require(result.ok(),label+": "+result.error);
    require(!result.grids.empty() && result.grids[0] == candidate[0],label+": L0 changed");
    for (int lev = 0; lev < int(result.grids.size()); ++lev) {
        clustering::check_rank_order(result.grids[lev]);
        require(!result.grids[lev].empty(),label+": empty intermediate or trailing level");
        if (!lev) continue;
        auto const ratio = layout.mesh.ref_ratio[lev-1];
        for (int b = 0; b < result.grids[lev].size(); ++b)
            for (int d = 0; d < 3; ++d) {
                auto const box = result.grids[lev][b];
                require(box.smallEnd(d)%ratio[d] == 0 && (box.bigEnd(d)+1)%ratio[d] == 0,
                        label+": incomplete parent group");
                require(box.length(d) <= layout.mesh.max_grid_size[lev][d],
                        label+": max_grid_size exceeded");
            }
    }
    amrex::Print() << "dynamic_constraint_grid_sample " << label;
    for (int lev = 0; lev < int(result.grids.size()); ++lev)
        amrex::Print() << " L" << lev << '=' << result.grids[lev].numPts();
    amrex::Print() << '\n';
    return std::move(result.grids);
}

template<class Descriptor>
void check_buffer_and_rejections()
{
    auto spec = spec_for<Descriptor>(1);
    spec.isPeriodic[0] = 0;
    spec.tagBufferCells = {amrex::IntVect(2,2,Descriptor::d == 3 ? 2 : 0)};
    spec.constraints = {{rectangle(spec,0,8,0,32),0,0},
                        {rectangle(spec,24,32,0,32),0,0}};
    auto layout = constraints::prepared(spec,Descriptor::d);
    Tags tags(1);
    append(tags,0,column(shape(layout,0),8,9,12,13));
    auto output = finish(candidates(layout),layout,tags,"clipped-dynamic-buffer");
    require(output.size() == 2,"legal edge tag was rejected with its optional buffer");
    constraints::check_cells(output[1],shape(layout,1),
        [](auto p) { return p[0]>=16 && p[0]<22 && p[1]>=20 && p[1]<30; },
        "buffer is clipped before closure, not the legal source tag");

    spec.constraints.push_back({rectangle(spec,8,12,12,16),1,1});
    layout = constraints::prepared(spec,Descriptor::d);
    output = finish(candidates(layout),layout,{},"clipped-fixed-ring");
    constraints::check_cells(output[1],shape(layout,1),
        [](auto p) { return p[0]>=16 && p[0]<28 && p[1]>=20 && p[1]<36; },
        "fixed ring is optional and buffered only once");

    spec = spec_for<Descriptor>(1);
    spec.constraints = {{rectangle(spec,11,13,0,32),0,0}};
    layout = constraints::prepared(spec,Descriptor::d);
    tags.assign(1,{});
    append(tags,0,column(shape(layout,0),8,11,0,32));
    append(tags,0,column(shape(layout,0),13,16,0,32));
    append(tags,0,column(shape(layout,0),24,26,0,32));
    auto expected = [](auto p) { return (p[0]>=16 && p[0]<22) || (p[0]>=48 && p[0]<52); };
    output = finish(candidates(layout),layout,tags,"reject-gap-batch-keep-later");
    constraints::check_cells(output[1],shape(layout,1),expected,
        "rejected gap closure must not contaminate accepted requests");
    if (amrex::ParallelDescriptor::IOProcessor()) {
        tags[0].push_back(tags[0].front());
        std::reverse(tags[0].begin(),tags[0].end());
    }
    auto reordered = finish(candidates(layout,true),layout,tags,"reordered-duplicate-tags");
    require(reordered.size() == output.size() && reordered[1] == output[1],
            "source order or candidate Box organization changed final fine grids");

    spec = spec_for<Descriptor>();
    spec.constraints = {{rectangle(spec,6,8,0,32),0,0}};
    layout = constraints::prepared(spec,Descriptor::d);
    tags.assign(2,{});
    append(tags,0,column(shape(layout,0),24,26,0,32));
    append(tags,1,column(shape(layout,1),16,18,0,64));
    output = finish(candidates(layout),layout,tags,"reject-required-nesting");
    require(output.size() == 2,"inadmissible L2 batch left a trailing level");
    constraints::check_cells(output[1],shape(layout,1),
        [](auto p) { return p[0]>=48 && p[0]<52; },"rejected L2 support did not leak into L1");
}

template<class Descriptor>
void check_periodic_components_and_error()
{
    auto spec = spec_for<Descriptor>();
    spec.constraints = {{rectangle(spec,1,2,0,32),0,0}};
    auto layout = constraints::prepared(spec,Descriptor::d);
    Tags tags(2);
    append(tags,1,column(shape(layout,1),0,1,0,64));
    append(tags,1,column(shape(layout,1),63,64,0,64));
    auto output = finish(candidates(layout),layout,tags,"periodic-component-rejected-together");
    require(output.size() == 1,
        "periodic face-neighbors were split: x=63 alone fits, x=0 makes shared support invalid");

    // The no-request case ignores AMReX clustering extras, including an entire
    // fine domain. It is not legal to reconstruct optional tags from candidates.
    output = finish(candidates(layout,true),layout,{},"no-raw-request");
    require(output.size() == 1,"candidate extras became permanent refinement requests");

    spec = spec_for<Descriptor>(1);
    spec.constraints = {{rectangle(spec,8,12,8,12),1,1}};
    layout = constraints::prepared(spec,Descriptor::d);
    // Direct pure-geometry caller supplies an inconsistent value object to check
    // error delivery. Normal case preparation already rejects this combination.
    layout.constraints->allowedCells[1].clear();
    auto failed = amr::finalize_dynamic_grids(candidates(layout),layout,{});
    require(!failed.ok() && failed.grids.empty() &&
            failed.error.find("invalid_refinement_constraints:") == 0,
            "hard conflict must return an error without publishing candidate grids");
}

template<class Descriptor>
void check_clustering_fallbacks()
{
    auto spec = spec_for<Descriptor>(1);
    spec.gridEfficiency = 0.4;
    spec.maxGridSize = {amrex::IntVect(64,64,Descriptor::d == 3 ? 8 : 1)};
    spec.constraints = {{rectangle(spec,8,12,8,24),1,1},
                        {rectangle(spec,12,24,8,12),1,1},
                        {rectangle(spec,12,24,12,24),0,0}};
    auto layout = constraints::prepared(spec,Descriptor::d);
    auto output = finish(candidates(layout),layout,{},"infeasible-cluster-exact-fallback");
    // 112/256 > .4: the first cluster fills the 16x16 bounding square.
    // Its missing corner is forbidden, so the original L must survive exactly.
    auto lShape = [](auto p) {
        return p[0]>=16 && p[0]<48 && p[1]>=16 && p[1]<48 &&
               (p[0]<24 || p[1]<24);
    };
    constraints::check_cells(output[1],shape(layout,1),lShape,
        "eff=1 fallback retained the feasible L rather than its forbidden corner");

    spec = spec_for<Descriptor>(1,64);
    spec.gridEfficiency = 0.43;
    spec.maxGridSize = {amrex::IntVect(64,64,Descriptor::d == 3 ? 8 : 1)};
    spec.constraints = {{rectangle(spec,4,8,4,20),1,1},
                        {rectangle(spec,8,20,4,8),1,1},
                        {rectangle(spec,34,38,4,20),1,1},
                        {rectangle(spec,22,34,16,20),1,1}};
    layout = constraints::prepared(spec,Descriptor::d);
    // The two opposing Ls are farther than three source cells apart, so the
    // initial geometric closure leaves them separate. Their joint density is
    // 224/544 < .43; the empty x=20,21 columns split clustering into two Ls,
    // each 112/256 > .43. Filled rectangles then have a two-cell gap, which
    // closure fills. An exact second clustering must retain that new bridge.
    constraints::check_cells(layout.constraints->minimumCells[1],shape(layout,1),
        [](auto p) {
            int const x=p[0]/2,y=p[1]/2;
            return (x>=4 && x<20 && y>=4 && y<20 && (x<8 || y<8)) ||
                   (x>=22 && x<38 && y>=4 && y<20 && (x>=34 || y>=16));
        },"raw opposing Ls");
    output = finish(candidates(layout),layout,{},"grown-closure-exact-fallback");
    constraints::check_cells(output[1],shape(layout,1),
        [](auto p) { return p[0]>=8 && p[0]<76 && p[1]>=8 && p[1]<40; },
        "new gap support is retained after successful clustering closure");
}

template<class Descriptor>
void check_tandem_connection()
{
    auto spec = spec_for<Descriptor>();
    spec.constraints = {{rectangle(spec,8,12,12,16),2,2},
                        {rectangle(spec,20,24,12,16),2,2}};
    auto layout = constraints::prepared(spec,Descriptor::d);
    auto initial = finish(candidates(layout),layout,{},"tandem-initial-separated");
    require(initial.size() == 3,"tandem hard L2 missing");
    auto separated = [](auto p) {
        return p[1]>=48 && p[1]<64 &&
            ((p[0]>=32 && p[0]<48) || (p[0]>=80 && p[0]<96));
    };
    constraints::check_cells(initial[2],shape(layout,2),separated,
        "two exact separated finest rectangles");
    constraints::check_cells(initial[1],shape(layout,1),
        [](auto p) { return p[1]>=22 && p[1]<34 &&
            ((p[0]>=14 && p[0]<26) || (p[0]>=38 && p[0]<50)); },
        "independent parent support around two protected rectangles");
    Tags tags(2);
    append(tags,1,column(shape(layout,1),24,40,24,32));
    auto connected = finish(initial,layout,tags,"tandem-tags-connect");
    constraints::check_cells(connected[2],shape(layout,2),
        [](auto p) { return p[0]>=32 && p[0]<96 && p[1]>=48 && p[1]<64; },
        "deterministic bridge connects actual L2");
    constraints::check_cells(connected[1],shape(layout,1),
        [](auto p) { return p[0]>=14 && p[0]<50 && p[1]>=22 && p[1]<34; },
        "connected L2 required parent support");
    auto restored = finish(connected,layout,{},"tandem-optional-bridge-removed");
    constraints::check_cells(restored[2],shape(layout,2),separated,
        "optional coverage from the previous event is not a hard seed");
}

template<class Descriptor>
void run_checks()
{
    check_buffer_and_rejections<Descriptor>();
    check_periodic_components_and_error<Descriptor>();
    check_clustering_fallbacks<Descriptor>();
    check_tandem_connection<Descriptor>();
    amrex::Print() << "dynamic_constraints_finalize_pass d=" << Descriptor::d
                   << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
}
} // namespace lbm::tests::dynamic_amr::constraint_grids
#endif
