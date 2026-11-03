#ifndef LBM_BOUNDARY_INTERNAL_ANALYTICCYLINDERGEOMETRY_H
#define LBM_BOUNDARY_INTERNAL_ANALYTICCYLINDERGEOMETRY_H

#include "lbm/boundary/internal/InternalGeometryTypes.H"

#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_Geometry.H>
#include <AMReX_iMultiFab.H>

#include <cstdint>
#include <limits>

namespace lbm::boundary {

struct SurfaceTolerance {
    amrex::Real physicalLength = 0.0;
    bool valid = false;
};

enum class AnalyticCylinderRootStatus : std::uint8_t {
    NoTransverseMotion,
    NoRealRoot,
    Tangent,
    TwoRoots,
    Invalid
};

struct AnalyticCylinderRootSet {
    AnalyticCylinderRootStatus status = AnalyticCylinderRootStatus::Invalid;
    amrex::Array<amrex::Real, 2> roots{
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN()};
    amrex::Real tangentQ =
        std::numeric_limits<amrex::Real>::quiet_NaN();
};

/** @brief Immutable z-cylinder shared by project and AMReX adapters. */
class AnalyticCylinderGeometry {
public:
    explicit AnalyticCylinderGeometry(ResolvedInternalBoundary const& boundary);

    [[nodiscard]] BoundaryId boundaryId() const noexcept { return boundaryId_; }
    [[nodiscard]] ObjectId objectId() const noexcept { return objectId_; }
    [[nodiscard]] amrex::RealArray const& center() const noexcept
    {
        return center_;
    }
    [[nodiscard]] amrex::EB2::CylinderIF const& implicitFunction() const noexcept
    {
        return implicit_;
    }

    [[nodiscard]] SurfaceTolerance surfaceTolerance(
        amrex::Geometry const& geom) const noexcept;
    GeometryQueryStatus classifyPoint(
        amrex::RealArray const& point,
        amrex::Real tolerance,
        PointLocation& location) const noexcept;
    GeometryQueryStatus fillCenterPhase(
        amrex::iMultiFab& locations,
        amrex::Geometry const& geom) const;
    [[nodiscard]] AnalyticCylinderRootSet exactSegmentRoots(
        amrex::RealArray const& fluidPoint,
        amrex::RealArray const& solidPoint) const noexcept;

private:
    BoundaryId boundaryId_;
    ObjectId objectId_;
    amrex::RealArray center_;
    amrex::Real radius_ = 0.0;
    int axis_ = 2;
    amrex::EB2::CylinderIF implicit_;
};

} // namespace lbm::boundary

#endif
