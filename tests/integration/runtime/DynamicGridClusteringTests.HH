#ifndef LBM_TESTS_DYNAMICGRIDCLUSTERINGTESTS_HH
#define LBM_TESTS_DYNAMICGRIDCLUSTERINGTESTS_HH

#include "integration/runtime/DynamicAmrTestSupport.H"
#include <AMReX_BoxIterator.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace lbm::tests::dynamic_amr::clustering {

// Integer occupancy is independent of production Box subtraction and repair.
struct Cells {
    amrex::IntVect shape;
    std::vector<std::uint8_t> value;
    explicit Cells(amrex::IntVect size)
        : shape(size), value(std::size_t(size[0])*size[1]*size[2],0) {}
    std::size_t index(amrex::IntVect p) const
    {
        for (int d = 0; d < 3; ++d) p[d] = (p[d]%shape[d]+shape[d])%shape[d];
        return (std::size_t(p[2])*shape[1]+p[1])*shape[0]+p[0];
    }
    amrex::IntVect position(std::size_t i) const
    { return {int(i%shape[0]),int(i/shape[0]%shape[1]),int(i/(std::size_t(shape[0])*shape[1]))}; }
};

inline Cells occupancy(amrex::BoxArray const& boxes, amrex::IntVect shape)
{
    Cells result(shape);
    amrex::Box domain(amrex::IntVect(0),shape-amrex::IntVect(1));
    for (int b = 0; b < boxes.size(); ++b) {
        require(domain.contains(boxes[b]),"clustering box outside domain");
        for (amrex::BoxIterator it(boxes[b]); it.ok(); ++it) {
            auto& count = result.value[result.index(it())];
            require(count == 0,"clustering duplicate valid cell");
            count = 1;
        }
    }
    return result;
}

inline Cells coarse(Cells const& fine, amrex::IntVect ratio)
{
    Cells result(fine.shape/ratio);
    for (std::size_t i = 0; i < fine.value.size(); ++i)
        if (fine.value[i]) result.value[result.index(fine.position(i)/ratio)] = 1;
    return result;
}

// Separable Boolean stencil; no production growth/complement operations.
inline Cells dilate(Cells cells, int radius, int dimensions)
{
    for (int axis = 0; axis < dimensions; ++axis) {
        Cells next(cells.shape);
        for (std::size_t i = 0; i < cells.value.size(); ++i) {
            auto p = cells.position(i);
            for (int offset = -radius; offset <= radius; ++offset) {
                auto q = p; q[axis] += offset;
                if (cells.value[cells.index(q)]) { next.value[i] = 1; break; }
            }
        }
        cells = std::move(next);
    }
    return cells;
}

inline void invert(Cells& cells)
{ for (auto& value : cells.value) value = !value; }

inline void check_gap_closure(Cells const& cells, int radius, int dimensions)
{
    auto closed = dilate(cells,radius,dimensions);
    invert(closed);
    closed = dilate(std::move(closed),radius,dimensions);
    invert(closed);
    require(closed.value == cells.value,"clustering final coverage has an unfilled gap");

    std::vector<int> component(cells.value.size(),-1);
    std::vector<std::size_t> queue;
    int count = 0;
    int const zRadius = dimensions == 3 ? 1 : 0;
    for (std::size_t seed = 0; seed < cells.value.size(); ++seed) {
        if (!cells.value[seed] || component[seed] >= 0) continue;
        component[seed] = count++;
        queue.assign(1,seed);
        for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
            auto const p = cells.position(queue[cursor]);
            for (int z = -zRadius; z <= zRadius; ++z)
                for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
                    auto const i = cells.index(p+amrex::IntVect(x,y,z));
                    if (cells.value[i] && component[i] < 0) {
                        component[i] = component[seed];
                        queue.push_back(i);
                    }
                }
        }
    }
    if (count < 2) return;
    int const reach = 2*radius+1;
    for (std::size_t i = 0; i < cells.value.size(); ++i) if (cells.value[i]) {
        auto const p = cells.position(i);
        for (int z = dimensions == 3 ? -reach : 0; z <= (dimensions == 3 ? reach : 0); ++z)
            for (int y = -reach; y <= reach; ++y) for (int x = -reach; x <= reach; ++x) {
                auto const j = cells.index(p+amrex::IntVect(x,y,z));
                require(!cells.value[j] || component[i] == component[j],
                        "clustering final components violate the gap separation");
            }
    }
}

inline void check_rank_order(amrex::BoxArray const& boxes)
{
    std::vector<int> local;
    for (int b = 0; b < boxes.size(); ++b)
        for (int d = 0; d < 3; ++d) {
            local.push_back(boxes[b].smallEnd(d));
            local.push_back(boxes[b].bigEnd(d));
        }
    int size = int(local.size());
    int const root = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(&size,1,root);
    require(size == int(local.size()),"clustering rank box count differs");
    auto reference = local;
    amrex::ParallelDescriptor::Bcast(reference.data(),reference.size(),root);
    require(reference == local,"clustering rank BA/FAB order differs");
}

inline void check_geometry(std::vector<amrex::BoxArray> const& input,
                           std::vector<amrex::BoxArray> const& output,
                           amr::DynamicHierarchyLayout const& layout)
{
    require(output.size() == input.size() && output[0] == input[0],"clustering changed L0 or active levels");
    auto shape = layout.geometry.Domain().length();
    auto const ratio = layout.mesh.ref_ratio.front();
    std::vector<Cells> actual;
    for (int lev = 0; lev < int(output.size()); ++lev) {
        check_rank_order(output[lev]);
        auto original = occupancy(input[lev],shape);
        actual.push_back(occupancy(output[lev],shape));
        auto const& current = actual.back();
        for (std::size_t i = 0; i < original.value.size(); ++i)
            require(!original.value[i] || current.value[i],"clustering lost requested coverage");
        for (int b = 0; b < output[lev].size(); ++b) {
            auto const& box = output[lev][b];
            for (int d = 0; d < 3; ++d) {
                require(box.length(d) <= layout.mesh.max_grid_size[lev][d],"clustering exceeds max_grid_size");
                if (lev) {
                    int const alignment = std::max(ratio[d],layout.mesh.blocking_factor[lev][d]);
                    require(box.smallEnd(d)%alignment == 0 && (box.bigEnd(d)+1)%alignment == 0,
                            "clustering broke blocking/refinement alignment");
                }
            }
        }
        if (lev) check_gap_closure(coarse(current,ratio),layout.gapFillRadiusCells,layout.activeDimensions);
        if (lev > 1) {
            auto support = dilate(coarse(current,ratio),layout.mesh.n_proper,layout.activeDimensions);
            auto const& parent = actual[lev-1];
            for (std::size_t i = 0; i < support.value.size(); ++i)
                require(!support.value[i] || parent.value[i],"clustering omitted periodic coarse support");
        }
        shape *= ratio;
    }
}

inline amrex::Box slab(int n, int lo, int hi, int dimensions)
{ return {{lo,0,0},{hi-1,n-1,dimensions == 3 ? n-1 : 0}}; }

inline amrex::BoxArray boxes(std::initializer_list<amrex::Box> list)
{
    amrex::BoxList result;
    for (auto const& box : list) result.push_back(box);
    return amrex::BoxArray(std::move(result));
}

inline std::vector<amrex::BoxArray> check(
    std::vector<amrex::BoxArray> const& candidate,
    amr::DynamicHierarchyLayout const& layout, std::string const& label)
{
    auto finalized = amr::finalize_dynamic_grids(candidate,layout);
    require(finalized.ok(), "unconstrained finalization failed: " + finalized.error);
    auto result = std::move(finalized.grids);
    check_geometry(candidate,result,layout);
    auto repeated = amr::finalize_dynamic_grids(result,layout);
    require(repeated.ok() && repeated.grids == result,"clustering final layout is not idempotent");
    amrex::Print() << "dynamic_clustering_sample " << label;
    for (int lev = 0; lev < int(result.size()); ++lev)
        amrex::Print() << " L" << lev << '=' << result[lev].size() << '/' << result[lev].numPts();
    amrex::Print() << '\n';
    return result;
}

template<class Descriptor>
void run_checks()
{
    constexpr int dimensions = Descriptor::d;
    auto spec = base_spec<Descriptor>();
    spec.nCell = {16,16,dimensions == 3 ? 16 : 1};
    spec.gapFillRadiusCells = 1;
    spec.coarseNestingMarginCells = 2;
    auto layout = *amr::prepare_dynamic_layout(spec,dimensions).layout;
    auto base = amrex::BoxArray(layout.geometry.Domain());
    base.maxSize(layout.mesh.max_grid_size[0]);
    require(check({base},layout,"L0-only") == std::vector<amrex::BoxArray>{base},"L0-only changed");

    // Preserve actual partition order, including a deliberately reversed order.
    auto stable = boxes({slab(32,8,24,dimensions)});
    stable.maxSize(layout.mesh.max_grid_size[1]);
    amrex::BoxList reversed;
    for (int b = int(stable.size())-1; b >= 0; --b) reversed.push_back(stable[b]);
    stable = amrex::BoxArray(std::move(reversed));
    require(check({base,stable},layout,"no-growth")[1] == stable,"equal coverage repartitioned");

    auto merged = check({base,boxes({slab(32,4,12,dimensions),slab(32,16,24,dimensions)})},
                        layout,"gap-fill");
    require(occupancy(merged[1],{32,32,dimensions == 3 ? 32 : 1}).value ==
            occupancy(boxes({slab(32,4,24,dimensions)}),{32,32,dimensions == 3 ? 32 : 1}).value,
            "hand-derived filled slab mismatch");

    auto periodic = check({base,boxes({slab(32,2,10,dimensions),slab(32,22,30,dimensions)})},
                          layout,"periodic-gap");
    require(occupancy(periodic[1],{32,32,dimensions == 3 ? 32 : 1}).value ==
            occupancy(boxes({slab(32,0,10,dimensions),slab(32,22,32,dimensions)}),
                      {32,32,dimensions == 3 ? 32 : 1}).value,"hand-derived periodic seam mismatch");

    auto nested = check({base,boxes({slab(32,8,24,dimensions)}),
                         boxes({slab(64,28,36,dimensions),slab(64,40,48,dimensions)})},layout,"nesting");
    require(occupancy(nested[1],{32,32,dimensions == 3 ? 32 : 1}).value ==
            occupancy(boxes({slab(32,8,26,dimensions)}),{32,32,dimensions == 3 ? 32 : 1}).value,
            "hand-derived parent expansion mismatch");
    require(occupancy(nested[2],{64,64,dimensions == 3 ? 64 : 1}).value ==
            occupancy(boxes({slab(64,28,48,dimensions)}),{64,64,dimensions == 3 ? 64 : 1}).value,
            "hand-derived fine expansion mismatch");

    // Disjoint fine cells share one coarse tag: alignment must deduplicate it.
    auto partial = boxes({{{8,8,dimensions == 3 ? 8 : 0},{8,8,dimensions == 3 ? 8 : 0}},
                          {{9,8,dimensions == 3 ? 8 : 0},{9,8,dimensions == 3 ? 8 : 0}}});
    auto aligned = check({base,partial},layout,"coarse-tag-duplicates");
    auto const alignedBox = amrex::Box({8,8,dimensions == 3 ? 8 : 0},{9,9,dimensions == 3 ? 9 : 0});
    require(aligned[1].numPts() == (1 << dimensions) && aligned[1].contains(alignedBox),
            "coarse tag group was duplicated or lost");

    auto blocked = layout;
    blocked.mesh.blocking_factor[1] = {4,4,dimensions == 3 ? 4 : 1};
    auto blockedResult = check({base,partial},blocked,"blocking-scale");
    require(blockedResult[1].numPts() == (dimensions == 3 ? 64 : 16),
            "blocking conversion did not preserve the enlarged coarse group");

    layout.mesh.grid_eff = 0.4;
    // L1 already supports the gap-closed L2 L-shape, so the first repair
    // leaves L1 unchanged. L2 clustering fills its bounding square (112/256
    // tags >= 0.4); only the following repair can add its missing L1 corner.
    auto support = boxes({{{6,6,0},{13,25,dimensions == 3 ? 31 : 0}},
                          {{14,6,0},{25,13,dimensions == 3 ? 31 : 0}}});
    support.maxSize(layout.mesh.max_grid_size[1]);
    auto fine = boxes({{{16,16,0},{23,47,dimensions == 3 ? 63 : 0}},
                       {{28,16,0},{47,23,dimensions == 3 ? 63 : 0}}});
    require(!support.contains(amrex::IntVect(24,24,0)),"second repair witness already covered");
    auto repeated = check({base,support,fine},layout,"second-repair");
    auto square = boxes({{{6,6,0},{25,25,dimensions == 3 ? 31 : 0}}});
    require(occupancy(repeated[1],{32,32,dimensions == 3 ? 32 : 1}).value ==
            occupancy(square,{32,32,dimensions == 3 ? 32 : 1}).value,
            "hand-derived second repair support mismatch");

    if constexpr (dimensions == 2) {
        spec.nCell = {128,128,1};
        spec.maxGridSize = {amrex::IntVect(32,32,1)};
        spec.gridEfficiency = 0.7;
        spec.gapFillRadiusCells = 4;
        spec.coarseNestingMarginCells = 4;
        layout = *amr::prepare_dynamic_layout(spec,dimensions).layout;
        for (int step : {832,1280,1664}) {
            std::ifstream source(std::string(LBM_DYNAMIC_GRID_FIXTURE_DIR)+"/candidate-"+std::to_string(step)+".boxes");
            require(bool(source),"missing saved AMReX candidate");
            std::array<amrex::BoxList,3> levels;
            int lev, x0, x1, y0, y1, z0, z1;
            while (source >> lev >> x0 >> x1 >> y0 >> y1 >> z0 >> z1) {
                require(lev >= 0 && lev < 3,"invalid snapshot level");
                levels[lev].push_back(amrex::Box({x0,y0,z0},{x1-1,y1-1,z1-1}));
            }
            require(source.eof(),"malformed saved candidate");
            std::vector<amrex::BoxArray> candidate;
            for (auto& level : levels) candidate.emplace_back(std::move(level));
            std::array<int,3> const counts = step == 832 ? std::array<int,3>{16,47,98}
                : step == 1280 ? std::array<int,3>{16,99,247} : std::array<int,3>{16,132,460};
            for (int level = 0; level < 3; ++level)
                require(candidate[level].size() == counts[level],"snapshot source count changed");
            auto final = check(candidate,layout,"saved-"+std::to_string(step));
            if (step == 832) require(final == candidate,"saved no-growth sample changed");
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::filesystem::create_directories("artifacts/dynamic_clustering");
                std::ofstream output("artifacts/dynamic_clustering/final-"+std::to_string(step)+".boxes");
                for (int level = 0; level < 3; ++level) for (int b = 0; b < final[level].size(); ++b) {
                    output << level;
                    for (int d = 0; d < 3; ++d)
                        output << ' ' << final[level][b].smallEnd(d) << ' ' << final[level][b].bigEnd(d)+1;
                    output << '\n';
                }
                require(bool(output),"failed to write final snapshot");
            }
        }
    }
    amrex::Print() << "dynamic_clustering_pass d=" << dimensions
                   << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
}

} // namespace lbm::tests::dynamic_amr::clustering
#endif
