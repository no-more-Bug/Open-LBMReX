#ifndef LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYQUERY_H
#define LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYQUERY_H

#include "lbm/boundary/internal/AnalyticCylinderGeometry.H"

#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <vector>

namespace lbm::boundary {

struct InternalGeometryQueryBuildResult;
class InternalGeometryQueryRegistry {
public:
    [[nodiscard]] bool empty() const noexcept;

    GeometryQueryStatus fillCenterPhase(
        amrex::iMultiFab& locations,
        amrex::iMultiFab& surfaceProvenance,
        amrex::Geometry const& geom,
        PhaseBuildReport& report) const;
    GeometryQueryStatus decodeSurfaceProvenance(
        int token,
        GeometrySurfaceProvenance& result) const noexcept;
    GeometryQueryStatus classifyPoint(
        amrex::RealArray const& point,
        amrex::Geometry const& geom,
        CenterPointClassification& result) const;
    [[nodiscard]] LinkIntersectionResult intersectLink(
        LinkIntersectionRequest const& request,
        amrex::Geometry const& geom) const;
    [[nodiscard]] GeometryIntersectionStatus segmentTopology(
        amrex::RealArray const& first,
        amrex::RealArray const& second,
        bool firstOnSurface,
        bool secondOnSurface,
        amrex::Geometry const& geom) const noexcept;
    bool makeEbCylinderFunctions(
        std::vector<amrex::EB2::CylinderIF>& functions) const;

private:
    friend struct InternalGeometryQueryBuildResult;
    friend InternalGeometryQueryBuildResult build_internal_geometry_query(
        ResolvedBoundarySet const& resolved, amrex::Geometry const& geom);

    GeometryQueryStatus classifyDisjoint(
        amrex::RealArray const& point,
        amrex::Geometry const& geom,
        CenterPointClassification& result,
        int& insideLeaf) const;

    std::vector<AnalyticCylinderGeometry> leaves_;
};

struct InternalGeometryQueryBuildResult {
    InternalGeometryQueryRegistry query;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
    [[nodiscard]] std::string errorMessage() const;
};

/** Prepare images intersecting the physical domain and one-cell read neighborhood. */
InternalGeometryQueryBuildResult build_internal_geometry_query(
    ResolvedBoundarySet const& resolved, amrex::Geometry const& geom);

} // namespace lbm::boundary

#endif
