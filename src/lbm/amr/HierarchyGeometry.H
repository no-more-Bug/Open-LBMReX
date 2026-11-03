#ifndef LBM_AMR_HIERARCHYGEOMETRY_H
#define LBM_AMR_HIERARCHYGEOMETRY_H

#include <AMReX_AmrCore.H>

#include <optional>
#include <string>
#include <vector>

namespace lbm::amr {

/** Shared physical domain and mesh controls; refinement decisions belong to the caller. */
struct HierarchyGeometrySpec {
    amrex::IntVect refRatio{2, 2, 1};
    amrex::Array<int, 3> nCell{32, 32, 1};
    amrex::Array<amrex::Real, 3> probLo{0.0, 0.0, 0.0};
    amrex::Array<amrex::Real, 3> probHi{1.0, 1.0, 0.0625};
    amrex::Array<int, 3> isPeriodic{1, 1, 1};
    std::vector<amrex::IntVect> maxGridSize{amrex::IntVect(32, 32, 1)};
    int gapFillRadiusCells = 4;
    int coarseNestingMarginCells = 4;
    amrex::Real gridEfficiency = 1.0;
};

struct HierarchyGeometry {
    int activeDimensions = 2;
    amrex::Geometry geometry;
    amrex::AmrInfo mesh;
    int gapFillRadiusCells = 4;
};

struct HierarchyGeometryResult {
    std::optional<HierarchyGeometry> layout;
    std::vector<std::string> errors;
    bool ok() const noexcept { return layout.has_value() && errors.empty(); }
};

HierarchyGeometryResult prepare_hierarchy_geometry(
    HierarchyGeometrySpec const& spec, int activeDimensions, int maxLevel);

} // namespace lbm::amr
#endif
