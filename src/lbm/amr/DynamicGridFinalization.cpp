#include "lbm/amr/DynamicRefinement.H"
#include "lbm/amr/StaticRegionTopology.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_Cluster.H>
#include <AMReX_ParallelDescriptor.H>
#include <algorithm>
#include <unordered_set>
#include <utility>

namespace lbm::amr {
namespace {
using Grids = std::vector<amrex::BoxArray>;
using Points = std::vector<amrex::IntVect>;

Points enumerate(amrex::BoxArray const& cells)
{
    Points points;
    points.reserve(cells.numPts());
    for (int i = 0; i < cells.size(); ++i)
        for (amrex::BoxIterator cell(cells[i]); cell.ok(); ++cell) points.push_back(cell());
    return points;
}

amrex::BoxArray cluster_coverage(amrex::BoxArray const& coverage,
                                amrex::Box const& coarseDomain, int level,
                                DynamicHierarchyLayout const& layout, amrex::Real efficiency)
{
    if (coverage.empty()) return {};
    auto const& ratio = layout.mesh.ref_ratio[level-1];
    amrex::IntVect blocking;
    for (int d = 0; d < 3; ++d)
        blocking[d] = std::max(1, layout.mesh.blocking_factor[level][d] / ratio[d]);
    auto cells = detail::coarse_cells(coverage, ratio);
    cells = detail::periodic_grow(cells, coarseDomain, layout.geometry.periodicity(),
                                  layout.activeDimensions, 0);
    cells = detail::coarse_cells(cells, blocking);
    // Unique parent tags make efficiency=1 an exact coverage operation.
    auto tags = enumerate(cells);
    amrex::ClusterList clusters(tags.data(), static_cast<amrex::Long>(tags.size()));
    if (layout.mesh.use_new_chop) clusters.new_chop(efficiency);
    else clusters.chop(efficiency);
    auto boxes = clusters.boxList();
    amrex::BoxArray result(std::move(boxes), layout.mesh.max_grid_size[level] / (ratio * blocking));
    result.refine(blocking);
    result = amrex::intersect(result, coarseDomain);
    result.refine(ratio);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(detail::covers(result, coverage),
        "dynamic reclustering lost required coverage");
    return result;
}

void repair(Grids& grids, DynamicHierarchyLayout const& layout)
{
    std::vector<unsigned> reasons(grids.size(), 0);
    detail::repair_regions(grids, layout.geometry.Domain(), layout.mesh.ref_ratio.front(),
        layout.geometry.periodicity(), layout.activeDimensions,
        layout.gapFillRadiusCells, layout.mesh.n_proper, reasons);
}

bool close_within_allowed(Grids& grids, DynamicHierarchyLayout const& layout)
{
    repair(grids, layout);
    for (std::size_t lev = 1; lev < grids.size(); ++lev)
        if (!detail::covers(layout.constraints->allowedCells[lev], grids[lev])) return false;
    return true;
}

std::vector<amrex::Box> level_domains(std::size_t count, DynamicHierarchyLayout const& layout)
{
    std::vector<amrex::Box> domains(count, layout.geometry.Domain());
    for (std::size_t lev = 1; lev < count; ++lev)
        domains[lev] = amrex::refine(domains[lev-1], layout.mesh.ref_ratio[lev-1]);
    return domains;
}

Grids finalize_unconstrained(Grids candidate, DynamicHierarchyLayout const& layout)
{
    auto const domains = level_domains(candidate.size(), layout);
    for (;;) {
        auto repaired = candidate;
        repair(repaired, layout);
        bool grew = false;
        for (int lev = 1; lev < static_cast<int>(candidate.size()); ++lev) {
            if (detail::covers(candidate[lev], repaired[lev])) continue;
            candidate[lev] = cluster_coverage(repaired[lev], domains[lev-1], lev,
                                             layout, layout.mesh.grid_eff);
            grew = true;
        }
        // Preserve the 7-01 path: every continued round adds finite-domain coverage.
        if (!grew) return candidate;
    }
}

// Pack source requests into x-runs before set operations. These are temporary
// request regions; final computation boxes always come from ClusterList.
amrex::BoxArray pack_points(Points& points)
{
    std::sort(points.begin(), points.end(), [](auto const& a, auto const& b) {
        for (int d = 2; d >= 0; --d) if (a[d] != b[d]) return a[d] < b[d];
        return false;
    });
    amrex::BoxList boxes;
    for (std::size_t begin = 0; begin < points.size();) {
        std::size_t end = begin;
        while (end+1 < points.size() && points[end+1][1] == points[begin][1] &&
               points[end+1][2] == points[begin][2] && points[end+1][0] == points[end][0]+1)
            ++end;
        boxes.push_back(amrex::Box(points[begin], points[end]));
        begin = end+1;
    }
    boxes.simplify();
    return amrex::BoxArray(std::move(boxes));
}

template<class Consume>
void for_each_component(Points points, amrex::Box const& domain,
                        DynamicHierarchyLayout const& layout, Consume consume)
{
    std::sort(points.begin(), points.end(), [](auto const& a, auto const& b) {
        for (int d = 0; d < 3; ++d) if (a[d] != b[d]) return a[d] < b[d];
        return false;
    });
    points.erase(std::unique(points.begin(), points.end()), points.end());
    std::unordered_set<amrex::IntVect, amrex::IntVect::hasher> remaining(points.begin(), points.end());
    Points component;
    for (auto const& seed : points) {
        if (!remaining.erase(seed)) continue;
        component.assign(1, seed);
        for (std::size_t cursor = 0; cursor < component.size(); ++cursor) {
            auto const p = component[cursor];
            for (int d = 0; d < layout.activeDimensions; ++d) for (int offset : {-1, 1}) {
                auto neighbor = p;
                neighbor[d] += offset;
                if (neighbor[d] < domain.smallEnd(d) || neighbor[d] > domain.bigEnd(d)) {
                    if (!layout.geometry.isPeriodic(d)) continue;
                    neighbor[d] = neighbor[d] < domain.smallEnd(d) ? domain.bigEnd(d) : domain.smallEnd(d);
                }
                if (remaining.erase(neighbor)) component.push_back(neighbor);
            }
        }
        // Sorted seeds define source-level lexicographic component order,
        // independently of original Box/rank ordering or the hash iteration order.
        consume(pack_points(component));
    }
}

amrex::BoxArray buffered(amrex::BoxArray const& cells, amrex::IntVect const& buffer,
                          amrex::Box const& domain, DynamicHierarchyLayout const& layout)
{
    auto boxes = cells.boxList();
    boxes.accrete(buffer);
    return detail::periodic_grow(amrex::BoxArray(std::move(boxes)), domain,
                                 layout.geometry.periodicity(), layout.activeDimensions, 0);
}

DynamicGridResult finalize_constrained(Grids const& candidate, DynamicHierarchyLayout const& layout,
    std::vector<amrex::Gpu::PinnedVector<amrex::IntVect>> const& dynamicTags)
{
    auto const& constraints = *layout.constraints;
    auto accepted = constraints.minimumCells;
    accepted[0] = candidate[0];
    auto coverage = accepted;
    if (!close_within_allowed(coverage, layout))
        return {{}, "invalid_refinement_constraints: hard coverage or support exceeds allowed cells"};
    auto const domains = level_domains(accepted.size(), layout);

    auto try_batch = [&](int source, amrex::BoxArray cells) {
        cells.refine(layout.mesh.ref_ratio[source]);
        cells = amrex::intersect(cells, constraints.allowedCells[source+1]);
        if (cells.empty()) return;
        auto trial = accepted;
        trial[source+1] = detail::cell_union(trial[source+1], cells);
        auto closed = trial;
        if (close_within_allowed(closed, layout)) {
            accepted = std::move(trial);
            coverage = std::move(closed);
        }
    };

    // Fixed geometry buffers are optional rings from raw H, never from its closure.
    for (int source = 0; source < static_cast<int>(constraints.requiredTags.size()); ++source) {
        auto const& tags = constraints.requiredTags[source];
        if (tags.empty()) continue;
        auto grown = buffered(tags, layout.mesh.n_error_buf[source], domains[source], layout);
        auto ring = amrex::intersect(grown, amrex::complementIn(domains[source], tags));
        for_each_component(enumerate(ring), domains[source], layout,
            [&](amrex::BoxArray cells) { try_batch(source, std::move(cells)); });
    }
    auto const sourceCount = std::min(dynamicTags.size(), constraints.requiredTags.size());
    for (int source = 0; source < static_cast<int>(sourceCount); ++source) {
        for_each_component(Points(dynamicTags[source].begin(), dynamicTags[source].end()),
            domains[source], layout, [&](amrex::BoxArray cells) {
                try_batch(source, buffered(cells, layout.mesh.n_error_buf[source], domains[source], layout));
            });
    }

    auto cluster = [&](Grids grids, amrex::Real efficiency) {
        for (int lev = 1; lev < static_cast<int>(grids.size()); ++lev)
            grids[lev] = cluster_coverage(grids[lev], domains[lev-1], lev, layout, efficiency);
        return grids;
    };
    auto boxes = cluster(coverage, layout.mesh.grid_eff);
    auto closed = boxes;
    if (!close_within_allowed(closed, layout)) {
        boxes = cluster(std::move(coverage), 1);
    } else {
        bool grew = false;
        for (std::size_t lev = 1; lev < closed.size(); ++lev)
            grew = grew || !detail::covers(boxes[lev], closed[lev]);
        if (grew) boxes = cluster(std::move(closed), 1);
    }
    while (boxes.size() > 1 && boxes.back().empty()) boxes.pop_back();
    return {std::move(boxes), {}};
}
} // namespace

DynamicGridResult finalize_dynamic_grids(Grids candidate, DynamicHierarchyLayout const& layout,
    std::vector<amrex::Gpu::PinnedVector<amrex::IntVect>> const& dynamicTagsOnIo)
{
    DynamicGridResult result;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (layout.constraints)
            result = finalize_constrained(candidate, layout, dynamicTagsOnIo);
        else
            result.grids = finalize_unconstrained(std::move(candidate), layout);
    }
    auto const root = amrex::ParallelDescriptor::IOProcessorNumber();
    int errorSize = static_cast<int>(result.error.size());
    amrex::ParallelDescriptor::Bcast(&errorSize, 1, root);
    result.error.resize(errorSize);
    if (errorSize) {
        amrex::ParallelDescriptor::Bcast(result.error.data(), errorSize, root);
        return result;
    }
    int numLevels = static_cast<int>(result.grids.size());
    amrex::ParallelDescriptor::Bcast(&numLevels, 1, root);
    result.grids.resize(numLevels);
    for (auto& grid : result.grids) {
        auto boxes = grid.boxList();
        boxes.Bcast();
        grid = amrex::BoxArray(std::move(boxes));
    }
    return result;
}
} // namespace lbm::amr
