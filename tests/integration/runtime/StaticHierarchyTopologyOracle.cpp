#include "integration/runtime/StaticHierarchyTopologyOracle.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"

#include <cstdint>

namespace lbm::tests::static_hierarchy::topology {
namespace {

struct Cells {
    std::array<int,3> size;
    int dimensions;
    std::vector<std::uint8_t> value;

    Cells(std::array<int,3> shape, int dims)
        : size(shape), dimensions(dims), value(std::size_t(shape[0])*shape[1]*shape[2],0) {}
    std::size_t index(int x, int y, int z) const
    {
        const auto wrap = [](int v, int n) { return (v%n+n)%n; };
        return (std::size_t(wrap(z,size[2]))*size[1]+wrap(y,size[1]))*size[0]+wrap(x,size[0]);
    }
    std::array<int,3> position(std::size_t i) const
    { return {int(i%size[0]),int((i/size[0])%size[1]),int(i/(std::size_t(size[0])*size[1]))}; }
    void unite(Cells const& other)
    { for (std::size_t i = 0; i < value.size(); ++i) value[i] |= other.value[i]; }
};

// Separable rolling windows directly enumerate the periodic Boolean stencil.
Cells grow(Cells source, int radius)
{
    for (int axis = 0; axis < source.dimensions; ++axis) {
        Cells result(source.size,source.dimensions);
        std::array<int,3> p{};
        const int a = (axis+1)%3, b = (axis+2)%3, length = source.size[axis];
        for (p[a] = 0; p[a] < source.size[a]; ++p[a])
            for (p[b] = 0; p[b] < source.size[b]; ++p[b]) {
                if (2*std::int64_t(radius)+1 >= length) {
                    bool occupied = false;
                    for (p[axis] = 0; p[axis] < length; ++p[axis])
                        occupied |= source.value[source.index(p[0],p[1],p[2])] != 0;
                    for (p[axis] = 0; p[axis] < length; ++p[axis])
                        result.value[result.index(p[0],p[1],p[2])] = occupied;
                    continue;
                }
                int occupied = 0;
                for (p[axis] = -radius; p[axis] <= radius; ++p[axis])
                    occupied += source.value[source.index(p[0],p[1],p[2])];
                for (int x = 0; x < length; ++x) {
                    p[axis] = x;
                    result.value[result.index(p[0],p[1],p[2])] = occupied > 0;
                    p[axis] = x-radius;
                    occupied -= source.value[source.index(p[0],p[1],p[2])];
                    p[axis] = x+radius+1;
                    occupied += source.value[source.index(p[0],p[1],p[2])];
                }
            }
        source = std::move(result);
    }
    return source;
}

Cells complement(Cells cells)
{
    for (auto& v : cells.value) v = !v;
    return cells;
}

std::vector<std::vector<std::size_t>> components(Cells const& cells)
{
    std::vector<std::vector<std::size_t>> result;
    std::vector<bool> visited(cells.value.size(),false);
    const int zRadius = cells.dimensions == 3 ? 1 : 0;
    for (std::size_t seed = 0; seed < cells.value.size(); ++seed) {
        if (!cells.value[seed] || visited[seed]) continue;
        result.emplace_back(1,seed);
        auto& component = result.back();
        visited[seed] = true;
        for (std::size_t cursor = 0; cursor < component.size(); ++cursor) {
            auto const p = cells.position(component[cursor]);
            for (int dz = -zRadius; dz <= zRadius; ++dz)
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                    const auto next = cells.index(p[0]+dx,p[1]+dy,p[2]+dz);
                    if (cells.value[next] && !visited[next]) {
                        visited[next] = true;
                        component.push_back(next);
                    }
                }
        }
    }
    return result;
}

Cells repair(Cells cells, int radius)
{
    for (;;) {
        const auto before = cells.value;
        cells = complement(grow(complement(grow(std::move(cells),radius)),radius));
        const auto groups = components(cells);
        Cells participants(cells.size,cells.dimensions);
        for (std::size_t g = 0; g < groups.size() && groups.size() > 1; ++g) {
            Cells one(cells.size,cells.dimensions);
            for (auto i : groups[g]) one.value[i] = 1;
            const auto vicinity = grow(one,2*radius+1);
            bool tooNear = false;
            for (std::size_t h = 0; h < groups.size() && !tooNear; ++h) {
                if (g == h) continue;
                for (auto i : groups[h]) if (vicinity.value[i]) { tooNear = true; break; }
            }
            if (tooNear) participants.unite(one);
        }
        cells.unite(grow(std::move(participants),radius));
        if (cells.value == before) return cells;
    }
}

Cells coarsen(Cells const& cells)
{
    auto shape = cells.size;
    for (int d = 0; d < cells.dimensions; ++d) shape[d] /= 2;
    Cells result(shape,cells.dimensions);
    for (std::size_t i = 0; i < cells.value.size(); ++i) if (cells.value[i]) {
        auto p = cells.position(i);
        for (int d = 0; d < cells.dimensions; ++d) p[d] /= 2;
        result.value[result.index(p[0],p[1],p[2])] = 1;
    }
    return result;
}

std::vector<Cells> empty_levels(amr::StaticHierarchySpec const& spec, int dimensions)
{
    int finest = 0;
    for (auto const& request : spec.regions) finest = std::max(finest,request.level);
    std::vector<Cells> levels;
    for (int lev = 0; lev <= finest; ++lev) {
        std::array<int,3> shape{spec.nCell[0],spec.nCell[1],spec.nCell[2]};
        for (int d = 0; d < dimensions; ++d) shape[d] *= 1 << std::max(lev-1,0);
        levels.emplace_back(shape,dimensions);
    }
    std::fill(levels[0].value.begin(),levels[0].value.end(),1);
    return levels;
}

std::vector<Cells> minimum(amr::StaticHierarchySpec const& spec, int dimensions)
{
    auto levels = empty_levels(spec,dimensions);
    for (auto const& request : spec.regions) {
        auto& target = levels[request.level];
        std::array<int,3> lo{},hi{};
        for (int d = 0; d < 3; ++d) {
            const Real scale = target.size[d]/(spec.probHi[d]-spec.probLo[d]);
            lo[d] = int(std::llround((request.box.lo[d]-spec.probLo[d])*scale));
            hi[d] = int(std::llround((request.box.hi[d]-spec.probLo[d])*scale));
        }
        for (int z = lo[2]; z < hi[2]; ++z) for (int y = lo[1]; y < hi[1]; ++y)
            for (int x = lo[0]; x < hi[0]; ++x) target.value[target.index(x,y,z)] = 1;
    }
    for (int lev = int(levels.size())-1; lev > 0; --lev) {
        levels[lev] = repair(std::move(levels[lev]),spec.gapFillRadiusCells);
        if (lev > 1) levels[lev-1].unite(coarsen(grow(levels[lev],spec.coarseNestingMarginCells)));
    }
    return levels;
}

} // namespace

void validate_coverage(amr::StaticHierarchySpec const& spec,
    std::vector<Frame> const& frames, int dimensions)
{
    auto required = minimum(spec,dimensions);
    auto actual = empty_levels(spec,dimensions);
    require(frames.size() == required.size(),"topology oracle missing level");
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        auto const& f = frames[lev].populations;
        auto& a = actual[lev];
        std::fill(a.value.begin(),a.value.end(),0);
        const int children = lev ? (1 << dimensions) : 1;
        for (std::size_t row = 0; row < f.owners.size(); ++row) {
            auto p = f.cell_at(row);
            for (int d = 0; d < dimensions; ++d) if (lev) p[d] /= 2;
            ++a.value[a.index(p[0],p[1],p[2])];
        }
        for (std::size_t i = 0; i < a.value.size(); ++i) {
            require(a.value[i] == 0 || a.value[i] == children,"partial coarse tag cell in actual valid coverage");
            a.value[i] = a.value[i] != 0;
            require(!required[lev].value[i] || a.value[i],"actual coverage misses independent minimum");
        }
        if (spec.gridEfficiency == 1.0)
            require(a.value == required[lev].value,"grid_eff=1 complete independent coverage mismatch at level="+std::to_string(lev));
        if (lev) require(repair(a,spec.gapFillRadiusCells).value == a.value,"actual same-level repair is not a fixed point");
    }
    for (std::size_t lev = 2; lev < actual.size(); ++lev) {
        const auto needed = coarsen(grow(actual[lev],spec.coarseNestingMarginCells));
        for (std::size_t i = 0; i < needed.value.size(); ++i)
            require(!needed.value[i] || actual[lev-1].value[i],"actual periodic coarse nesting margin is missing");
    }
}

void validate_coverage(Fixture const& fixture, std::vector<Frame> const& frames, int dimensions)
{
    if (fixture.expectedCoverage.empty()) {
        validate_coverage(fixture.spec,frames,dimensions);
        return;
    }
    require(frames.size() == fixture.expectedCoverage.size(),"hand-derived hierarchy level count");
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        auto const& snapshot = frames[lev].populations;
        auto const& expected = fixture.expectedCoverage[lev];
        std::vector<std::size_t> ids;
        for (auto const& box : expected.boxes) for (amrex::BoxIterator it(box); it.ok(); ++it) {
            auto cell = it();
            for (int d = 0; d < dimensions; ++d) {
                const int n = snapshot.domain.length(d);
                cell[d] = (cell[d]%n+n)%n;
            }
            ids.push_back(snapshot.linear(cell));
        }
        std::sort(ids.begin(),ids.end());
        ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        require(snapshot.cellIndices == ids,"hand-derived complete cell set mismatch at level="+std::to_string(lev));

        // Every expected refined tag is a complete 2^d block. Coarsening these
        // blocks preserves Chebyshev connectivity without a dense finest mesh.
        const int scale = lev ? 2 : 1;
        std::array<int,3> shape{snapshot.domain.length(0)/scale,
            snapshot.domain.length(1)/scale,snapshot.domain.length(2)/(dimensions == 3 ? scale : 1)};
        Cells actual(shape,dimensions);
        for (std::size_t row = 0; row < snapshot.cellIndices.size(); ++row) {
            auto cell = snapshot.cell_at(row);
            for (int d = 0; d < dimensions; ++d) cell[d] /= scale;
            actual.value[actual.index(cell[0],cell[1],cell[2])] = 1;
        }
        const auto count = components(actual).size();
        require(count == std::size_t(expected.components),"hand-derived periodic component count at level="+std::to_string(lev));
        amrex::Print() << "topology_components sample=" << fixture.name << " level=" << lev
            << " valid_cells=" << ids.size() << " components=" << count << '\n';
    }
}

} // namespace lbm::tests::static_hierarchy::topology
