#ifndef LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYTYPES_H
#define LBM_BOUNDARY_INTERNAL_INTERNALGEOMETRYTYPES_H

#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_INT.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lbm::boundary {

enum class PointLocation : std::uint8_t { Outside, Inside, OnSurface };

AMREX_GPU_HOST_DEVICE inline constexpr int point_location_code(
    PointLocation value) noexcept
{
    return static_cast<int>(value);
}

AMREX_GPU_HOST_DEVICE inline constexpr bool is_point_location_code(
    int value) noexcept
{
    return value == point_location_code(PointLocation::Outside) ||
        value == point_location_code(PointLocation::Inside) ||
        value == point_location_code(PointLocation::OnSurface);
}

enum class SurfaceProvenanceCode : int { Ambiguous = -1, None = 0 };

AMREX_GPU_HOST_DEVICE inline constexpr int surface_provenance_token(
    int leafIndex) noexcept
{
    return leafIndex + 1;
}

AMREX_GPU_HOST_DEVICE inline constexpr bool
is_unique_surface_provenance_token(int value) noexcept
{
    return value > 0;
}

AMREX_GPU_HOST_DEVICE inline constexpr int surface_provenance_leaf_index(
    int value) noexcept
{
    return value - 1;
}

enum class GeometryIntersectionMethod : std::uint8_t {
    None,
    AnalyticClosedForm
};

enum class GeometryIntersectionQuality : std::uint8_t {
    Unspecified,
    Exact
};

enum class GeometryQueryStatus : std::uint8_t {
    Success,
    InvalidRequest
};

enum class GeometryIntersectionStatus : std::uint8_t {
    Unsupported,
    NoHit,
    UniqueHit,
    Tangent,
    MultipleHits,
    Invalid
};

struct GeometrySurfaceProvenance {
    BoundaryId boundaryId;
    ObjectId objectId;
    bool present = false;
    bool unique = false;
};

struct CenterPointClassification {
    PointLocation location = PointLocation::Outside;
    GeometrySurfaceProvenance surface;
};

struct PhaseBuildReport {
    amrex::Long outsideCount = 0;
    amrex::Long insideCount = 0;
    amrex::Long onSurfaceCount = 0;
};

struct LinkIntersectionRequest {
    amrex::Array<amrex::Real, 3> fluidPoint{0.0, 0.0, 0.0};
    amrex::Array<amrex::Real, 3> solidPoint{0.0, 0.0, 0.0};
    BoundaryId boundaryId;
    ObjectId objectId;
    bool surfaceCoincidentFluid = false;
};

struct LinkIntersectionResult {
    GeometryIntersectionStatus status = GeometryIntersectionStatus::Invalid;
    GeometryIntersectionMethod method = GeometryIntersectionMethod::None;
    GeometryIntersectionQuality quality =
        GeometryIntersectionQuality::Unspecified;
    GeometrySurfaceProvenance provenance;
    amrex::Real q = std::numeric_limits<amrex::Real>::quiet_NaN();
    amrex::Array<amrex::Real, 3> intersection{
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN()};
    amrex::Array<amrex::Real, 3> normalSolidToFluid{
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN(),
        std::numeric_limits<amrex::Real>::quiet_NaN()};
    bool surfaceCoincidentFluid = false;

    [[nodiscard]] bool contractValid(
        LinkIntersectionRequest const& request) const noexcept
    {
        if (status != GeometryIntersectionStatus::UniqueHit ||
            method != GeometryIntersectionMethod::AnalyticClosedForm ||
            quality != GeometryIntersectionQuality::Exact ||
            !provenance.present || !provenance.unique ||
            provenance.boundaryId.value < 0 || provenance.objectId.value < 0 ||
            !std::isfinite(q) || q < 0.0 || q >= 1.0 ||
            surfaceCoincidentFluid != request.surfaceCoincidentFluid ||
            (surfaceCoincidentFluid ? q != 0.0 : q <= 0.0)) {
            return false;
        }
        if (request.boundaryId.value >= 0 &&
            (request.objectId.value < 0 ||
             request.boundaryId.value != provenance.boundaryId.value ||
             request.objectId.value != provenance.objectId.value)) {
            return false;
        }
        const amrex::Real dx = request.solidPoint[0] - request.fluidPoint[0];
        const amrex::Real dy = request.solidPoint[1] - request.fluidPoint[1];
        const amrex::Real dz = request.solidPoint[2] - request.fluidPoint[2];
        const amrex::Real scale = std::max({
            std::abs(request.fluidPoint[0]), std::abs(request.fluidPoint[1]),
            std::abs(request.fluidPoint[2]), std::abs(request.solidPoint[0]),
            std::abs(request.solidPoint[1]), std::abs(request.solidPoint[2]),
            std::abs(dx), std::abs(dy), std::abs(dz)});
        const amrex::Real positionTolerance =
            64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
        const amrex::Real displacement[3]{dx, dy, dz};
        amrex::Real normalNorm2 = 0.0;
        amrex::Real normalDotLink = 0.0;
        amrex::Real dotScale = 0.0;
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(intersection[d]) ||
                !std::isfinite(normalSolidToFluid[d]) ||
                std::abs(
                    intersection[d] -
                    (request.fluidPoint[d] + q * displacement[d])) >
                    positionTolerance) {
                return false;
            }
            normalNorm2 += normalSolidToFluid[d] * normalSolidToFluid[d];
            normalDotLink += normalSolidToFluid[d] * displacement[d];
            dotScale += std::abs(normalSolidToFluid[d] * displacement[d]);
        }
        const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
        return std::abs(std::sqrt(normalNorm2) - 1.0) <= 512.0 * epsilon &&
            normalDotLink < -512.0 * epsilon * dotScale;
    }
};

} // namespace lbm::boundary

#endif
