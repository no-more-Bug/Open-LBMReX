#ifndef LBM_AMR_DYNAMICREFINEMENT_H
#define LBM_AMR_DYNAMICREFINEMENT_H

#include "lbm/amr/RefinementConstraints.H"

#include <AMReX_GpuContainers.H>

namespace lbm::amr {

struct DynamicHierarchySpec : HierarchyGeometrySpec {
    int maxLevel = 1;
    int regridInterval = 8;
    // Empty selects two cells per active direction; one entry broadcasts,
    // otherwise supply one entry per allowed level (including L0 and Lmax).
    std::vector<amrex::IntVect> tagBufferCells;
    std::vector<RefinementConstraint> constraints;
};

struct DynamicHierarchyLayout : HierarchyGeometry {
    int regridInterval = 8;
    std::optional<RefinementConstraints> constraints;
};

struct DynamicLayoutResult {
    std::optional<DynamicHierarchyLayout> layout;
    std::vector<std::string> errors;
    bool ok() const noexcept { return layout.has_value() && errors.empty(); }
};

DynamicLayoutResult prepare_dynamic_layout(DynamicHierarchySpec const& spec, int activeDimensions);


struct DynamicGridResult {
    std::vector<amrex::BoxArray> grids;
    std::string error;
    bool ok() const noexcept { return error.empty(); }
};

// Decide this event's final coverage without migrating fields. Raw optional tags
// are collated by the caller; their full contents are needed only on the IO rank.
DynamicGridResult finalize_dynamic_grids(
    std::vector<amrex::BoxArray> candidate, DynamicHierarchyLayout const& layout,
    std::vector<amrex::Gpu::PinnedVector<amrex::IntVect>> const& dynamicTagsOnIo = {});

} // namespace lbm::amr
#endif
