#include "lbm/boundary/internal/AnalyticCylinderGeometry.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lbm::boundary {

AnalyticCylinderGeometry::AnalyticCylinderGeometry(
    ResolvedInternalBoundary const& boundary)
    : boundaryId_(boundary.boundaryId)
    , objectId_(boundary.objectId)
    , center_(boundary.centerPhys)
    , radius_(boundary.radiusPhys)
    , axis_(boundary.axis)
    , implicit_(
          boundary.radiusPhys,
          boundary.axis,
          boundary.centerPhys,
          false)
{}

SurfaceTolerance AnalyticCylinderGeometry::surfaceTolerance(
    amrex::Geometry const& geom) const noexcept
{
    const amrex::Real scale = std::max({
        std::abs(center_[0]), std::abs(center_[1]), std::abs(center_[2]),
        std::abs(radius_), std::abs(geom.ProbLo(0)),
        std::abs(geom.ProbLo(1)), std::abs(geom.ProbLo(2)),
        std::abs(geom.ProbHi(0)), std::abs(geom.ProbHi(1)),
        std::abs(geom.ProbHi(2)), std::abs(geom.CellSize(0)),
        std::abs(geom.CellSize(1))});
    const amrex::Real value =
        64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
    const amrex::Real limit =
        1.0e-6 * std::min(geom.CellSize(0), geom.CellSize(1));
    return {value, std::isfinite(value) && value >= 0.0 && value <= limit};
}

GeometryQueryStatus AnalyticCylinderGeometry::classifyPoint(
    amrex::RealArray const& point,
    amrex::Real tolerance,
    PointLocation& location) const noexcept
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
        !std::isfinite(point[2]) || !std::isfinite(tolerance) ||
        tolerance < 0.0) {
        return GeometryQueryStatus::InvalidRequest;
    }
    const amrex::Real rho =
        std::hypot(point[0] - center_[0], point[1] - center_[1]);
    if (!(rho < radius_ - tolerance) &&
        !(rho > radius_ + tolerance)) {
        location = PointLocation::OnSurface;
    } else {
        location = implicit_(point) > 0.0 ? PointLocation::Inside
                                          : PointLocation::Outside;
    }
    return GeometryQueryStatus::Success;
}

GeometryQueryStatus AnalyticCylinderGeometry::fillCenterPhase(
    amrex::iMultiFab& locations,
    amrex::Geometry const& geom) const
{
    if (!locations.ixType().cellCentered() || locations.nComp() != 1) {
        return GeometryQueryStatus::InvalidRequest;
    }
    const SurfaceTolerance tolerance = surfaceTolerance(geom);
    if (!tolerance.valid) {
        return GeometryQueryStatus::InvalidRequest;
    }
    auto const probLo = geom.ProbLoArray();
    auto const cellSize = geom.CellSizeArray();
    const amrex::Real cx = center_[0];
    const amrex::Real cy = center_[1];
    const amrex::Real radius = radius_;
    const amrex::Real band = tolerance.physicalLength;
    const amrex::EB2::CylinderIF implicit = implicit_;
    for (amrex::MFIter mfi(locations); mfi.isValid(); ++mfi) {
        auto const output = locations.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const amrex::Real x =
                    probLo[0] + (static_cast<amrex::Real>(i) + 0.5) *
                        cellSize[0];
                const amrex::Real y =
                    probLo[1] + (static_cast<amrex::Real>(j) + 0.5) *
                        cellSize[1];
                const amrex::Real z =
                    probLo[2] + (static_cast<amrex::Real>(k) + 0.5) *
                        cellSize[2];
                const amrex::Real rho = std::hypot(x - cx, y - cy);
                const PointLocation location =
                    (!(rho < radius - band) && !(rho > radius + band))
                    ? PointLocation::OnSurface
                    : (implicit(AMREX_D_DECL(x, y, z)) > 0.0
                           ? PointLocation::Inside
                           : PointLocation::Outside);
                output(i, j, k) = point_location_code(location);
            });
    }
    return GeometryQueryStatus::Success;
}

AnalyticCylinderRootSet AnalyticCylinderGeometry::exactSegmentRoots(
    amrex::RealArray const& fluid,
    amrex::RealArray const& solid) const noexcept
{
    AnalyticCylinderRootSet result;
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(fluid[d]) || !std::isfinite(solid[d])) {
            return result;
        }
    }
    if (!std::isfinite(radius_) || radius_ <= 0.0 || axis_ != 2) {
        return result;
    }
    const amrex::Real ux = (fluid[0] - center_[0]) / radius_;
    const amrex::Real uy = (fluid[1] - center_[1]) / radius_;
    const amrex::Real vx = (solid[0] - fluid[0]) / radius_;
    const amrex::Real vy = (solid[1] - fluid[1]) / radius_;
    const amrex::Real a = vx * vx + vy * vy;
    const amrex::Real b = 2.0 * (ux * vx + uy * vy);
    const amrex::Real c = ux * ux + uy * uy - 1.0;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
        a < 0.0) {
        return result;
    }
    if (a == 0.0) {
        result.status = AnalyticCylinderRootStatus::NoTransverseMotion;
        return result;
    }
    const amrex::Real discriminant = b * b - 4.0 * a * c;
    const amrex::Real tolerance =
        128.0 * std::numeric_limits<amrex::Real>::epsilon() *
        (b * b + 4.0 * std::abs(a * c));
    if (!std::isfinite(discriminant) || !std::isfinite(tolerance)) {
        return result;
    }
    if (discriminant < -tolerance) {
        result.status = AnalyticCylinderRootStatus::NoRealRoot;
        return result;
    }
    if (std::abs(discriminant) <= tolerance) {
        result.tangentQ = -b / (2.0 * a);
        result.status = std::isfinite(result.tangentQ)
            ? AnalyticCylinderRootStatus::Tangent
            : AnalyticCylinderRootStatus::Invalid;
        return result;
    }
    const amrex::Real squareRoot = std::sqrt(discriminant);
    const amrex::Real stableTerm =
        -0.5 * (b + std::copysign(squareRoot, b));
    if (!std::isfinite(stableTerm) || stableTerm == 0.0) {
        return result;
    }
    result.roots = {stableTerm / a, c / stableTerm};
    if (result.roots[1] < result.roots[0]) {
        std::swap(result.roots[0], result.roots[1]);
    }
    for (amrex::Real q : result.roots) {
        const amrex::Real numerator = std::abs(a * q * q + b * q + c);
        const amrex::Real denominator =
            std::abs(a) * q * q + std::abs(b) * std::abs(q) + std::abs(c);
        const amrex::Real residual =
            denominator > 0.0 ? numerator / denominator : numerator;
        if (!std::isfinite(q) || !std::isfinite(residual) ||
            residual > 512.0 * std::numeric_limits<amrex::Real>::epsilon()) {
            return result;
        }
    }
    result.status = AnalyticCylinderRootStatus::TwoRoots;
    return result;
}

} // namespace lbm::boundary
