#include "lbm/amr/StaticRefinement.H"
#include "lbm/amr/StaticRegionTopology.H"

#include <AMReX_Print.H>

#include <AMReX_BoxList.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace lbm::amr {
namespace {

constexpr auto roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon();

std::string point(amrex::IntVect const& p)
{
    std::ostringstream out;
    out << p;
    return out.str();
}

std::string region_context(int index, int lev, RefinementBox const& r)
{
    std::ostringstream out;
    out << std::setprecision(17) << " level=" << lev << " request=" << index
        << " box=[(" << r.lo[0] << ',' << r.lo[1] << ',' << r.lo[2] << "),("
        << r.hi[0] << ',' << r.hi[1] << ',' << r.hi[2] << ")]";
    return out.str();
}

std::optional<amrex::IntVect> first_missing(amrex::BoxArray const& owners,
                                          amrex::BoxArray const& needed)
{
    for (int i = 0; i < needed.size(); ++i) {
        auto missing = owners.complementIn(needed[i]);
        if (!missing.isEmpty()) return missing.front().smallEnd();
    }
    return {};
}

std::optional<amrex::BoxArray> discretize(RefinementBox const& region,
    amrex::Geometry const& coarse, amrex::IntVect const& ratio, int activeDimensions,
    std::string const& context, std::vector<std::string>& errors)
{
    amrex::IntVect lower(0), upper(0);
    for (int d = 0; d < 3; ++d) {
        const auto width = region.hi[d] - region.lo[d];
        const auto length = coarse.ProbLength(d);
        const auto offset = region.lo[d] - coarse.ProbLo(d);
        if (!std::isfinite(region.lo[d]) || !std::isfinite(region.hi[d]) ||
            !std::isfinite(width) || !std::isfinite(offset) || width <= 0 ||
            width > length + roundoff * length) {
            errors.push_back("invalid_region:" + context + " direction=" + std::to_string(d) +
                " requires finite lo<hi and width<=one period");
            return {};
        }
        if (!coarse.isPeriodic(d) &&
            (offset < -roundoff * length || offset + width > length + roundoff * length)) {
            errors.push_back("invalid_region:" + context + " direction=" + std::to_string(d) +
                " nonperiodic region must lie inside the physical domain");
            return {};
        }
        if (d >= activeDimensions &&
            (region.lo[d] != coarse.ProbLo(d) || region.hi[d] != coarse.ProbHi(d))) {
            errors.push_back("inactive_region_extent:" + context +
                " inactive direction must span the physical domain");
            return {};
        }
        const int count = coarse.Domain().length(d);
        if (std::abs(width - length) <= roundoff * length) {
            lower[d] = 0; upper[d] = count-1;
            continue;
        }
        auto start = offset;
        if (coarse.isPeriodic(d)) {
            start = std::fmod(offset, length);
            if (start < 0) start += length;
        }
        const auto a = start / coarse.CellSize(d);
        const auto b = (start + width) / coarse.CellSize(d);
        const auto ia = std::round(a), ib = std::round(b);
        const auto tolerance = roundoff * std::max({amrex::Real(1), std::abs(a), std::abs(b)});
        if (std::abs(a-ia) > tolerance || std::abs(b-ib) > tolerance || ib <= ia) {
            errors.push_back("unaligned_region:" + context + " direction=" + std::to_string(d) +
                " boundaries must coincide with coarse cell faces");
            return {};
        }
        // Normalize the start before forming the endpoint so large valid domains cannot overflow.
        const auto begin = coarse.isPeriodic(d) ? static_cast<long long>(ia) % count : static_cast<long long>(ia);
        const auto end = begin + static_cast<long long>(ib-ia) - 1;
        if (end > std::numeric_limits<int>::max()) {
            errors.push_back("index_overflow:" + context);
            return {};
        }
        lower[d] = static_cast<int>(begin); upper[d] = static_cast<int>(end);
    }
    auto cells = detail::periodic_grow(amrex::BoxArray(amrex::Box(lower, upper)),
                                       coarse.Domain(), coarse.periodicity(), activeDimensions, 0);
    cells.refine(ratio);
    return cells;
}

} // namespace

StaticLayoutResult prepare_static_layout(StaticHierarchySpec const& spec, int activeDimensions)
{
    StaticLayoutResult result;
    auto& errors = result.errors;
    int maxLevel = 0;
    for (auto const& region : spec.regions) {
        if (region.level < 1)
            errors.push_back("invalid_level_count: level=0 requested refinement level must be positive");
        maxLevel = std::max(maxLevel, region.level);
    }
    if (!errors.empty()) return result;
    auto prepared = prepare_hierarchy_geometry(spec, activeDimensions, maxLevel);
    if (!prepared.ok()) {
        result.errors = std::move(prepared.errors);
        return result;
    }
    StaticHierarchyLayout layout;
    static_cast<HierarchyGeometry&>(layout) = std::move(*prepared.layout);
    auto const& ratio = spec.refRatio;
    auto const& domain = layout.geometry.Domain();
    layout.requestedCells.resize(maxLevel+1);
    layout.requestedCells[0] = amrex::BoxArray(domain);
    layout.repairReasons.assign(maxLevel+1, 0);
    std::vector<amrex::Geometry> geometries(maxLevel+1, layout.geometry);
    for (int lev = 1; lev <= maxLevel; ++lev)
        geometries[lev] = amrex::refine(geometries[lev-1], ratio);
    std::vector<amrex::BoxList> requests(maxLevel+1);
    for (std::size_t index = 0; index < spec.regions.size(); ++index) {
        auto const& request = spec.regions[index];
        auto cells = discretize(request.box, geometries[request.level-1], ratio, activeDimensions,
            region_context(static_cast<int>(index), request.level, request.box), errors);
        if (!cells) return result;
        requests[request.level].join(cells->boxList());
    }
    // Repair the complete request union, never one region at a time.
    for (int lev = 1; lev <= maxLevel; ++lev)
        layout.requestedCells[lev] = detail::cell_union(std::move(requests[lev]));
    layout.targetCells = layout.requestedCells;
    detail::repair_regions(layout.targetCells, domain, ratio, layout.geometry.periodicity(), activeDimensions,
                           layout.gapFillRadiusCells, layout.mesh.n_proper, layout.repairReasons);
    layout.tagCells.resize(maxLevel);
    for (int lev = 0; lev < maxLevel; ++lev)
        layout.tagCells[lev] = detail::coarse_cells(layout.targetCells[lev+1], ratio);
    result.layout.emplace(std::move(layout));
    return result;
}

StaticLayoutUpdate update_static_layout(StaticHierarchyLayout& layout, amrex::AmrCore const& actual)
{
    StaticLayoutUpdate result;
    if (actual.finestLevel() != layout.mesh.max_level) {
        result.errors.push_back("missing_level: required=" + std::to_string(layout.mesh.max_level) +
                                " generated=" + std::to_string(actual.finestLevel()));
        return result;
    }
    std::vector<amrex::BoxArray> proposed(layout.targetCells.size());
    for (int lev = 0; lev <= actual.finestLevel(); ++lev) {
        auto const& boxes = actual.boxArray(lev);
        if (auto gap = first_missing(boxes, layout.targetCells[lev])) {
            result.errors.push_back("coverage_mismatch: level=" + std::to_string(lev) +
                                    " missing_cell=" + point(*gap));
        }
        if (!detail::covers(layout.targetCells[lev], boxes))
            layout.repairReasons[lev] |= detail::AmrexCoverage;
        proposed[lev] = boxes; // M is contained in A, so M union A is exactly A.
    }
    if (!result.errors.empty()) return result;
    auto const& ratio = layout.mesh.ref_ratio.front();
    detail::repair_regions(proposed, layout.geometry.Domain(), ratio, layout.geometry.periodicity(), layout.activeDimensions,
                           layout.gapFillRadiusCells, layout.mesh.n_proper, layout.repairReasons);
    for (int lev = 1; lev <= actual.finestLevel(); ++lev) {
        result.rebuild = result.rebuild || !detail::covers(actual.boxArray(lev), proposed[lev]);
    }
    if (result.rebuild) {
        // M is contained in A: any proposed cell outside A is also strictly new to M.
        layout.targetCells = std::move(proposed);
        for (int lev = 0; lev < actual.finestLevel(); ++lev)
            layout.tagCells[lev] = detail::coarse_cells(layout.targetCells[lev+1], ratio);
    }
    return result;
}

void report_static_layout(StaticHierarchyLayout const& layout,
                         amrex::AmrCore const& actual, int initializationPasses)
{
    if (actual.finestLevel() == 0) return;
    amrex::Print() << "static_hierarchy_initialized passes=" << initializationPasses << '\n';
    for (int lev = 1; lev <= actual.finestLevel(); ++lev) {
        const auto requested = layout.requestedCells[lev].numPts();
        const auto final = actual.boxArray(lev).numPts();
        const auto added = final - requested;
        if (added == 0) continue;
        const auto reasons = layout.repairReasons[lev];
        amrex::Print() << "static_refinement level=" << lev << " requested_valid_cells=" << requested
            << " final_valid_cells=" << final << " added_valid_cells=" << added
            << " gap_fill=" << bool(reasons & detail::GapFill)
            << " coarse_nesting=" << bool(reasons & detail::CoarseNesting)
            << " amrex_coverage=" << bool(reasons & detail::AmrexCoverage) << '\n';
    }
}

} // namespace lbm::amr
