#include "lbm/amr/HierarchyGeometry.H"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace lbm::amr {

HierarchyGeometryResult prepare_hierarchy_geometry(
    HierarchyGeometrySpec const& spec, int activeDimensions, int maxLevel)
{
    HierarchyGeometryResult result;
    auto& errors = result.errors;
    constexpr auto roundoff = 64 * std::numeric_limits<amrex::Real>::epsilon();
    if (maxLevel < 0) {
        errors.push_back("invalid_level_count: level=0 maxLevel must be nonnegative");
        return result;
    }
    auto const& ratio = spec.refRatio;
    if ((activeDimensions != 2 && activeDimensions != 3) ||
        ratio != amrex::IntVect(2,2,activeDimensions == 3 ? 2 : 1)) {
        errors.push_back("invalid_ref_ratio: level=0 ratio must match the descriptor's active dimensions");
        return result;
    }
    for (int d = 0; d < 3; ++d) {
        const auto length = spec.probHi[d] - spec.probLo[d];
        if (spec.nCell[d] <= 0 || !std::isfinite(spec.probLo[d]) ||
            !std::isfinite(spec.probHi[d]) || !std::isfinite(length) || length <= 0.0 ||
            (spec.nCell[d] > 0 && length / spec.nCell[d] <= 0.0)) {
            errors.push_back("invalid_mesh: level=0 finite increasing bounds and positive cell counts required");
        }
        if (spec.isPeriodic[d] != 0 && spec.isPeriodic[d] != 1) {
            errors.push_back("invalid_periodicity: level=0 periodic flags must be zero or one");
        }
        if (d >= activeDimensions && spec.nCell[d] != 1) {
            errors.push_back("invalid_cell_count: level=0 inactive counts must be one");
        }
    }
    if (spec.maxGridSize.size() != 1 && spec.maxGridSize.size() != static_cast<std::size_t>(maxLevel)+1) {
        errors.push_back("invalid_max_grid_size: level=0 supply one size or one per level");
    }
    for (auto const& size : spec.maxGridSize) {
        for (int d = 0; d < 3; ++d) {
            if (size[d] < ratio[d] || size[d] % ratio[d] || (d >= activeDimensions && size[d] != 1)) {
                errors.push_back("invalid_max_grid_size: level=0 sizes must match active directions and ratio");
            }
        }
    }
    if (spec.gapFillRadiusCells < 1 || spec.coarseNestingMarginCells < 1)
        errors.push_back("invalid_nesting_controls: level=0 positive gapFillRadiusCells and coarseNestingMarginCells required");
    if (!std::isfinite(spec.gridEfficiency) || spec.gridEfficiency <= 0 || spec.gridEfficiency > 1)
        errors.push_back("invalid_grid_efficiency: level=0 gridEfficiency must be in (0,1]");
    if (!errors.empty()) return result;
    const auto volume = (spec.probHi[0]-spec.probLo[0]) *
        (spec.probHi[1]-spec.probLo[1]) * (spec.probHi[2]-spec.probLo[2]);
    if (!std::isfinite(volume) || volume <= 0) {
        errors.push_back("invalid_mesh: level=0 physical volume must be finite and representable");
        return result;
    }
    const auto dx = (spec.probHi[0]-spec.probLo[0]) / spec.nCell[0];
    for (int d = 1; d < activeDimensions; ++d) {
        const auto spacing = (spec.probHi[d]-spec.probLo[d]) / spec.nCell[d];
        if (std::abs(dx-spacing) > roundoff * std::max(dx,spacing)) {
            errors.push_back("unequal_active_spacing: level=0 active lattice spacings must agree");
            return result;
        }
    }
    std::array<amrex::Long,3> counts{spec.nCell[0],spec.nCell[1],spec.nCell[2]};
    for (int lev = 0; lev <= maxLevel; ++lev) {
        amrex::Long total = 1;
        for (int d = 0; d < 3; ++d) {
            const auto padding = std::max(1, spec.coarseNestingMarginCells);
            if (counts[d] + 2LL*padding > std::numeric_limits<int>::max()) {
                errors.push_back("index_overflow: level=" + std::to_string(lev) + " refined domain/ghost size is not representable");
                return result;
            }
            if (total > std::numeric_limits<amrex::Long>::max() / counts[d]) {
                errors.push_back("index_overflow: level=" + std::to_string(lev) + " domain cell count is not representable");
                return result;
            }
            total *= counts[d];
            counts[d] *= ratio[d];
        }
    }
    for (int d = 0; d < 3; ++d) {
        const auto lo = spec.probLo[d], hi = spec.probHi[d];
        const auto positionResolution = std::max(std::nextafter(lo,hi)-lo, hi-std::nextafter(hi,lo));
        long long count = spec.nCell[d];
        for (int lev = 0; lev <= maxLevel; ++lev) {
            const auto spacing = (hi-lo) / count;
            // Geometry computes reciprocal spacing and requires distinguishable end faces.
            if (!std::isfinite(1.0/spacing) || spacing <= positionResolution ||
                !(lo+(count-1)*spacing < lo+count*spacing)) {
                errors.push_back("unrepresentable_geometry: level=" + std::to_string(lev) +
                    " direction=" + std::to_string(d) + " cell coordinates cannot be resolved");
                return result;
            }
            count *= ratio[d];
        }
    }
    const amrex::Box domain(amrex::IntVect(0), amrex::IntVect(spec.nCell[0]-1,spec.nCell[1]-1,spec.nCell[2]-1));
    const amrex::RealBox physical(spec.probLo.data(), spec.probHi.data());
    HierarchyGeometry layout;
    layout.activeDimensions = activeDimensions;
    layout.geometry = amrex::Geometry(domain, &physical, 0, spec.isPeriodic.data());
    layout.mesh.max_level = maxLevel;
    layout.mesh.ref_ratio.assign(maxLevel+1, ratio);
    layout.mesh.blocking_factor.assign(maxLevel+1, amrex::IntVect(1));
    layout.mesh.max_grid_size.assign(spec.maxGridSize.begin(), spec.maxGridSize.end());
    layout.mesh.max_grid_size.resize(maxLevel+1, spec.maxGridSize.back());
    layout.mesh.n_error_buf.assign(maxLevel+1, amrex::IntVect(0));
    layout.mesh.n_proper = spec.coarseNestingMarginCells;
    layout.mesh.grid_eff = spec.gridEfficiency;
    layout.mesh.refine_grid_layout = false;
    layout.mesh.refine_grid_layout_dims = amrex::IntVect(0);
    layout.gapFillRadiusCells = spec.gapFillRadiusCells;
    result.layout.emplace(std::move(layout));
    return result;
}

} // namespace lbm::amr
