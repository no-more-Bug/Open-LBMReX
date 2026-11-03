#include "lbm/amr/RefinementConstraints.H"
#include "lbm/amr/StaticRegionTopology.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace lbm::amr {
namespace {
constexpr auto roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon();

// Physical rectangles are normalized once, then snapped to complete parent cells.
// Splitting before integer conversion also avoids overflow across a periodic seam.
using Intervals = std::array<std::vector<std::pair<amrex::Real, amrex::Real>>, 3>;

bool normalize(RefinementBox const& box, HierarchyGeometry const& layout,
               Intervals& intervals)
{
    auto const& geometry = layout.geometry;
    for (int d = 0; d < 3; ++d) {
        auto const length = geometry.ProbLength(d);
        auto const width = box.hi[d] - box.lo[d];
        auto start = box.lo[d] - geometry.ProbLo(d);
        if (!std::isfinite(start) || !std::isfinite(box.hi[d]) ||
            !std::isfinite(width) || width <= 0 || width > length + roundoff * length)
            return false;
        if (d >= layout.activeDimensions &&
            (box.lo[d] != geometry.ProbLo(d) || box.hi[d] != geometry.ProbHi(d)))
            return false;
        if (!geometry.isPeriodic(d) &&
            (start < -roundoff * length || start + width > length + roundoff * length))
            return false;
        if (std::abs(width - length) <= roundoff * length) {
            intervals[d].emplace_back(0, length);
            continue;
        }
        if (geometry.isPeriodic(d)) {
            start = std::fmod(start, length);
            if (start < 0) start += length;
        }
        auto const end = start + width;
        intervals[d].emplace_back(std::max(amrex::Real(0), start), std::min(end, length));
        if (end > length && geometry.isPeriodic(d))
            intervals[d].emplace_back(0, end - length);
    }
    return true;
}

amrex::BoxArray outward_cells(Intervals const& intervals, amrex::Geometry const& geometry)
{
    std::array<std::vector<std::pair<int, int>>, 3> indices;
    auto snap = [](amrex::Real value) {
        auto const nearest = std::round(value);
        return std::abs(value - nearest) <= roundoff * std::max(amrex::Real(1), std::abs(value))
            ? nearest : value;
    };
    for (int d = 0; d < 3; ++d) {
        for (auto const& interval : intervals[d]) {
            auto const lo = snap(interval.first / geometry.CellSize(d));
            auto const hi = snap(interval.second / geometry.CellSize(d));
            auto const begin = std::max(0, static_cast<int>(std::floor(lo)));
            auto const end = std::min(geometry.Domain().bigEnd(d), static_cast<int>(std::ceil(hi)) - 1);
            if (begin <= end) indices[d].emplace_back(begin, end);
        }
    }
    amrex::BoxList boxes;
    for (auto const& x : indices[0]) for (auto const& y : indices[1])
        for (auto const& z : indices[2])
            boxes.push_back(amrex::Box(amrex::IntVect(x.first, y.first, z.first),
                                      amrex::IntVect(x.second, y.second, z.second)));
    return detail::cell_union(std::move(boxes));
}

amrex::BoxArray subtract(amrex::BoxArray const& cells, amrex::BoxArray const& forbidden)
{
    amrex::BoxList remaining;
    for (int i = 0; i < cells.size(); ++i) remaining.join(forbidden.complementIn(cells[i]));
    return detail::cell_union(std::move(remaining));
}
} // namespace

RefinementConstraintsResult prepare_refinement_constraints(
    std::vector<RefinementConstraint> const& requests, HierarchyGeometry const& layout)
{
    RefinementConstraintsResult result;
    auto fail = [&result](std::string const& message) {
        result.errors.push_back("invalid_refinement_constraints: " + message);
    };
    auto const maxLevel = layout.mesh.max_level;
    auto const& ratio = layout.mesh.ref_ratio.front();
    std::vector<amrex::Geometry> geometries(maxLevel + 1, layout.geometry);
    RefinementConstraints constraints;
    constraints.minimumCells.resize(maxLevel + 1);
    constraints.allowedCells.resize(maxLevel + 1);
    constraints.requiredTags.resize(maxLevel);
    constraints.minimumCells[0] = amrex::BoxArray(layout.geometry.Domain());
    for (int lev = 0; lev <= maxLevel; ++lev) {
        if (lev > 0) geometries[lev] = amrex::refine(geometries[lev - 1], ratio);
        constraints.allowedCells[lev] = amrex::BoxArray(geometries[lev].Domain());
    }
    for (std::size_t index = 0; index < requests.size(); ++index) {
        auto const& request = requests[index];
        Intervals intervals;
        if (request.minLevel < 0 || request.minLevel > request.maxLevel ||
            request.maxLevel > maxLevel || !normalize(request.box, layout, intervals)) {
            fail("request=" + std::to_string(index) + " requires a valid box and 0<=min<=max<=Lmax");
            return result;
        }
        for (int lev = 1; lev <= request.minLevel; ++lev) {
            auto cells = outward_cells(intervals, geometries[lev - 1]);
            cells.refine(ratio);
            constraints.minimumCells[lev] = detail::cell_union(constraints.minimumCells[lev], cells);
        }
        if (request.maxLevel < maxLevel) {
            // An exact-level region caps the same outward-snapped region it requires.
            // Otherwise the cap snaps on the parent of the first forbidden level.
            auto const parent = request.minLevel == request.maxLevel && request.minLevel > 0
                ? request.minLevel - 1 : request.maxLevel;
            auto forbidden = outward_cells(intervals, geometries[parent]);
            for (int lev = parent + 1; lev <= maxLevel; ++lev) {
                forbidden.refine(ratio);
                if (lev > request.maxLevel)
                    constraints.allowedCells[lev] = subtract(constraints.allowedCells[lev], forbidden);
            }
        }
    }

    // This excludes fine contact with a physical face; the case supplies the
    // additional distance needed by its boundary and interpolation stencils.
    if (maxLevel > 0) {
        auto const parents = detail::coarse_cells(constraints.allowedCells[1], ratio);
        auto interior = layout.geometry.Domain();
        for (int d = 0; d < layout.activeDimensions; ++d)
            if (!layout.geometry.isPeriodic(d)) interior.grow(d, -1);
        if (!detail::covers(amrex::BoxArray(interior), parents)) {
            fail("allowed L1 coverage reaches a nonperiodic domain face");
            return result;
        }
    }
    auto closed = constraints.minimumCells;
    std::vector<unsigned> reasons(maxLevel + 1, 0);
    detail::repair_regions(closed, layout.geometry.Domain(), ratio, layout.geometry.periodicity(),
        layout.activeDimensions, layout.gapFillRadiusCells, layout.mesh.n_proper, reasons);
    for (int lev = 1; lev <= maxLevel; ++lev) {
        if (!detail::covers(constraints.allowedCells[lev], closed[lev])) {
            fail("minimum coverage or its required support exceeds allowed level=" + std::to_string(lev));
            return result;
        }
        constraints.requiredTags[lev - 1] = detail::coarse_cells(constraints.minimumCells[lev], ratio);
    }
    result.constraints.emplace(std::move(constraints));
    return result;
}
} // namespace lbm::amr
