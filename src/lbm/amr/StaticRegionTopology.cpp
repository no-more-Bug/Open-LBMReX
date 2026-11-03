#include "lbm/amr/StaticRegionTopology.H"

#include <algorithm>
#include <array>
#include <numeric>
#include <utility>

namespace lbm::amr::detail {

amrex::BoxArray cell_union(amrex::BoxList boxes)
{
    amrex::BoxArray result(std::move(boxes));
    if (!result.empty()) result.removeOverlap();
    return result;
}

amrex::BoxArray cell_union(amrex::BoxArray const& a, amrex::BoxArray const& b)
{
    auto boxes = a.boxList();
    boxes.join(b.boxList());
    return cell_union(std::move(boxes));
}

amrex::BoxArray coarse_cells(amrex::BoxArray const& cells, amrex::IntVect const& ratio)
{
    auto boxes = cells.boxList();
    boxes.coarsen(ratio);
    return cell_union(std::move(boxes));
}

bool covers(amrex::BoxArray const& owners, amrex::BoxArray const& needed)
{
    return needed.empty() || owners.contains(needed, true);
}

namespace {
using Interval = std::pair<int, int>;

std::vector<Interval> periodic_interval(long long lo, long long hi, int lower, int length)
{
    const auto width = hi - lo + 1;
    if (width >= length) return {{lower, lower + length - 1}};
    auto start = (lo - lower) % length;
    if (start < 0) start += length;
    const auto end = start + width - 1;
    if (end < length) return {{lower + static_cast<int>(start), lower + static_cast<int>(end)}};
    return {{lower + static_cast<int>(start), lower + length - 1},
            {lower, lower + static_cast<int>(end - length)}};
}

amrex::BoxArray complement(amrex::BoxArray const& cells, amrex::Box const& domain)
{
    return amrex::BoxArray(cells.complementIn(domain));
}

amrex::BoxArray close_gaps(amrex::BoxArray const& cells, amrex::Box const& domain,
                           amrex::Periodicity const& periodicity, int dimensions, int radius)
{
    auto grown = periodic_grow(cells, domain, periodicity, dimensions, radius);
    auto missing = complement(grown, domain);
    return complement(periodic_grow(missing, domain, periodicity, dimensions, radius), domain);
}

amrex::BoxArray repair_gaps(amrex::BoxArray cells, amrex::Box const& domain,
                            amrex::Periodicity const& periodicity, int dimensions, int radius)
{
    if (cells.empty()) return cells;
    for (;;) {
        auto closed = close_gaps(cells, domain, periodicity, dimensions, radius);
        std::vector<int> roots(closed.size());
        std::iota(roots.begin(), roots.end(), 0);
        auto root = [&roots](int i) {
            while (roots[i] != i) {
                roots[i] = roots[roots[i]];
                i = roots[i];
            }
            return i;
        };
        int componentCount = closed.size();
        std::vector<std::pair<int, amrex::Box>> hits;
        // Every Box is connected; periodic grow(1) joins face, edge and vertex neighbors.
        for (int i = 0; i < closed.size(); ++i) {
            auto neighbors = periodic_grow(amrex::BoxArray(closed[i]), domain, periodicity, dimensions, 1);
            for (int j = 0; j < neighbors.size(); ++j) {
                closed.intersections(neighbors[j], hits);
                for (auto const& hit : hits) {
                    const int a = root(i), b = root(hit.first);
                    if (a != b) { roots[b] = a; --componentCount; }
                }
            }
        }
        std::vector<bool> participating(closed.size(), false);
        if (componentCount > 1) {
            for (int i = 0; i < closed.size(); ++i) {
                auto neighbors = periodic_grow(amrex::BoxArray(closed[i]), domain,
                                               periodicity, dimensions, 2LL * radius + 1);
                for (int j = 0; j < neighbors.size(); ++j) {
                    closed.intersections(neighbors[j], hits);
                    for (auto const& hit : hits) {
                        const int a = root(i), b = root(hit.first);
                        if (a != b) participating[a] = participating[b] = true;
                    }
                }
            }
        }
        amrex::BoxList selected;
        for (int i = 0; i < closed.size(); ++i)
            if (participating[root(i)]) selected.push_back(closed[i]);
        // Closing is idempotent; without a violating pair both exit conditions hold.
        if (selected.isEmpty()) return closed;
        // Decide all participants before growing any of them.
        // Positive growth of distinct, separated components adds at least one cell.
        cells = cell_union(closed, periodic_grow(amrex::BoxArray(std::move(selected)),
                                                domain, periodicity, dimensions, radius));
    }
}
} // namespace

amrex::BoxArray periodic_grow(amrex::BoxArray const& cells, amrex::Box const& domain,
                              amrex::Periodicity const& periodicity, int activeDimensions, long long radius)
{
    amrex::BoxList pieces;
    for (int i = 0; i < cells.size(); ++i) {
        std::array<std::vector<Interval>, 3> intervals;
        for (int d = 0; d < 3; ++d) {
            const auto grow = d < activeDimensions ? radius : 0;
            const auto lo = static_cast<long long>(cells[i].smallEnd(d)) - grow;
            const auto hi = static_cast<long long>(cells[i].bigEnd(d)) + grow;
            if (periodicity.isPeriodic(d)) {
                intervals[d] = periodic_interval(lo, hi, domain.smallEnd(d), domain.length(d));
            } else {
                const auto clippedLo = std::max(lo, static_cast<long long>(domain.smallEnd(d)));
                const auto clippedHi = std::min(hi, static_cast<long long>(domain.bigEnd(d)));
                if (clippedLo <= clippedHi) {
                    intervals[d].emplace_back(static_cast<int>(clippedLo), static_cast<int>(clippedHi));
                }
            }
        }
        for (auto const& x : intervals[0]) for (auto const& y : intervals[1])
            for (auto const& z : intervals[2])
                pieces.push_back(amrex::Box(amrex::IntVect(x.first, y.first, z.first),
                                           amrex::IntVect(x.second, y.second, z.second)));
    }
    return cell_union(std::move(pieces));
}

void repair_regions(std::vector<amrex::BoxArray>& cells, amrex::Box const& coarseDomain,
                    amrex::IntVect const& ratio, amrex::Periodicity const& periodicity, int activeDimensions,
                    int gapRadius, int nestingMargin, std::vector<unsigned>& reasons)
{
    std::vector<amrex::Box> domains(cells.size(), coarseDomain);
    for (std::size_t lev = 1; lev < cells.size(); ++lev)
        domains[lev] = amrex::refine(domains[lev-1], ratio);
    for (int lev = static_cast<int>(cells.size()) - 1; lev > 0; --lev) {
        auto tags = coarse_cells(cells[lev], ratio);
        auto aligned = amrex::refine(tags, ratio);
        if (!covers(cells[lev], aligned)) reasons[lev] |= CoarseNesting;
        auto repaired = repair_gaps(tags, domains[lev-1], periodicity, activeDimensions, gapRadius);
        if (!covers(tags, repaired)) reasons[lev] |= GapFill;
        cells[lev] = amrex::refine(repaired, ratio);
        if (lev > 1) {
            auto needed = periodic_grow(repaired, domains[lev-1], periodicity, activeDimensions, nestingMargin);
            if (!covers(cells[lev-1], needed)) {
                reasons[lev-1] |= CoarseNesting;
                cells[lev-1] = cell_union(cells[lev-1], needed);
            }
        }
    }
}

} // namespace lbm::amr::detail
