#include "lbm/amr/DynamicRefinement.H"

#include <limits>
#include <utility>

namespace lbm::amr {

DynamicLayoutResult prepare_dynamic_layout(DynamicHierarchySpec const& spec, int activeDimensions)
{
    DynamicLayoutResult result;
    auto prepared = prepare_hierarchy_geometry(spec, activeDimensions, spec.maxLevel);
    if (!prepared.ok()) {
        result.errors = std::move(prepared.errors);
        return result;
    }
    auto& errors = result.errors;
    if (spec.regridInterval < 0)
        errors.push_back("invalid_regrid_interval: regridInterval must be nonnegative");
    if (spec.coarseNestingMarginCells < 2)
        errors.push_back("invalid_nesting_controls: dynamic velocity interpolation requires at least two coarse cells");
    const auto numLevels = static_cast<std::size_t>(spec.maxLevel) + 1;
    if (!spec.tagBufferCells.empty() && spec.tagBufferCells.size() != 1 &&
        spec.tagBufferCells.size() != numLevels)
        errors.push_back("invalid_tag_buffer: supply one buffer or one per allowed level");
    if (!errors.empty()) return result;

    DynamicHierarchyLayout layout;
    static_cast<HierarchyGeometry&>(layout) = std::move(*prepared.layout);
    layout.regridInterval = spec.regridInterval;
    auto counts = layout.geometry.Domain().length();
    for (int lev = 0; lev <= spec.maxLevel; ++lev) {
        const auto buffer = spec.tagBufferCells.empty()
            ? amrex::IntVect(2, 2, activeDimensions == 3 ? 2 : 0)
            : spec.tagBufferCells[spec.tagBufferCells.size() == 1 ? 0 : lev];
        for (int d = 0; d < 3; ++d) {
            if (buffer[d] < 0 || (d >= activeDimensions && buffer[d] != 0) ||
                counts[d] + 2LL * buffer[d] > std::numeric_limits<int>::max()) {
                errors.push_back("invalid_tag_buffer: level=" + std::to_string(lev) +
                    " requires representable nonnegative buffers with zero inactive extent");
            }
        }
        layout.mesh.n_error_buf[lev] = buffer;
        if (lev < spec.maxLevel) counts *= spec.refRatio;
    }
    if (!errors.empty()) return result;
    auto constraints = prepare_refinement_constraints(spec.constraints, layout);
    if (!constraints.ok()) {
        result.errors = std::move(constraints.errors);
        return result;
    }
    if (!spec.constraints.empty()) layout.constraints = std::move(constraints.constraints);
    result.layout.emplace(std::move(layout));
    return result;
}

} // namespace lbm::amr
