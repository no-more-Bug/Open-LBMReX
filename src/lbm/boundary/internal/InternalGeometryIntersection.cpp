#include "lbm/boundary/internal/InternalGeometryQuery.H"

#include <cmath>
#include <limits>

namespace lbm::boundary {

LinkIntersectionResult InternalGeometryQueryRegistry::intersectLink(
    LinkIntersectionRequest const& request,
    amrex::Geometry const& geom) const
{
    LinkIntersectionResult result;
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(request.fluidPoint[d]) ||
            !std::isfinite(request.solidPoint[d])) {
            return result;
        }
    }
    if (leaves_.empty()) {
        result.status = GeometryIntersectionStatus::Unsupported;
        return result;
    }
    const amrex::Real dx = request.solidPoint[0] - request.fluidPoint[0];
    const amrex::Real dy = request.solidPoint[1] - request.fluidPoint[1];
    const amrex::Real dz = request.solidPoint[2] - request.fluidPoint[2];
    const amrex::Real transverseLength = std::hypot(dx, dy);
    if (!(dx * dx + dy * dy + dz * dz > 0.0) ||
        !(transverseLength > 0.0)) {
        return result;
    }

    CenterPointClassification fluidClass;
    CenterPointClassification solidClass;
    int fluidInside = -1;
    int selectedLeaf = -1;
    if (classifyDisjoint(
            request.fluidPoint, geom, fluidClass, fluidInside) !=
            GeometryQueryStatus::Success ||
        classifyDisjoint(
            request.solidPoint, geom, solidClass, selectedLeaf) !=
            GeometryQueryStatus::Success ||
        solidClass.location != PointLocation::Inside || selectedLeaf < 0) {
        return result;
    }
    if (request.surfaceCoincidentFluid) {
        if (fluidClass.location != PointLocation::OnSurface ||
            !fluidClass.surface.unique || request.boundaryId.value < 0 ||
            request.objectId.value < 0 ||
            request.boundaryId.value != fluidClass.surface.boundaryId.value ||
            request.objectId.value != fluidClass.surface.objectId.value) {
            return result;
        }
    } else if (fluidClass.location != PointLocation::Outside) {
        return result;
    }

    std::vector<AnalyticCylinderRootSet> roots(leaves_.size());
    std::vector<amrex::Real> qTolerances(leaves_.size());
    const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    for (std::size_t index = 0; index < leaves_.size(); ++index) {
        auto const& cylinder = leaves_[index];
        const SurfaceTolerance surface = cylinder.surfaceTolerance(geom);
        if (!surface.valid) {
            return result;
        }
        qTolerances[index] = std::max(
            256.0 * epsilon,
            4.0 * surface.physicalLength / transverseLength);
        roots[index] = cylinder.exactSegmentRoots(
            request.fluidPoint, request.solidPoint);
        if (roots[index].status == AnalyticCylinderRootStatus::Invalid) {
            return result;
        }
    }

    int selectedRoot = -1;
    amrex::Real q = std::numeric_limits<amrex::Real>::quiet_NaN();
    auto const& selectedRoots = roots[static_cast<std::size_t>(selectedLeaf)];
    const amrex::Real selectedTolerance =
        qTolerances[static_cast<std::size_t>(selectedLeaf)];
    if (request.surfaceCoincidentFluid) {
        auto const& cylinder = leaves_[static_cast<std::size_t>(selectedLeaf)];
        if (cylinder.boundaryId().value != request.boundaryId.value ||
            cylinder.objectId().value != request.objectId.value ||
            selectedRoots.status != AnalyticCylinderRootStatus::TwoRoots) {
            return result;
        }
        selectedRoot = std::abs(selectedRoots.roots[0]) <=
                std::abs(selectedRoots.roots[1])
            ? 0
            : 1;
        if (std::abs(selectedRoots.roots[selectedRoot]) > selectedTolerance) {
            return result;
        }
        q = 0.0;
    } else {
        if (selectedRoots.status == AnalyticCylinderRootStatus::Tangent) {
            result.status = GeometryIntersectionStatus::Tangent;
            return result;
        }
        if (selectedRoots.status != AnalyticCylinderRootStatus::TwoRoots) {
            return result;
        }
        for (int root = 0; root < 2; ++root) {
            const amrex::Real candidate = selectedRoots.roots[root];
            if (candidate > selectedTolerance &&
                candidate < 1.0 - selectedTolerance) {
                if (selectedRoot >= 0) {
                    result.status = GeometryIntersectionStatus::MultipleHits;
                    return result;
                }
                selectedRoot = root;
                q = candidate;
            } else if (candidate >= -selectedTolerance &&
                       candidate <= 1.0 + selectedTolerance) {
                return result;
            }
        }
        if (selectedRoot < 0) {
            return result;
        }
    }

    for (std::size_t leaf = 0; leaf < roots.size(); ++leaf) {
        const amrex::Real tolerance = qTolerances[leaf];
        if (roots[leaf].status == AnalyticCylinderRootStatus::Tangent &&
            roots[leaf].tangentQ >= -tolerance &&
            roots[leaf].tangentQ <= 1.0 + tolerance) {
            result.status = GeometryIntersectionStatus::Tangent;
            return result;
        }
        if (roots[leaf].status != AnalyticCylinderRootStatus::TwoRoots) {
            continue;
        }
        for (int root = 0; root < 2; ++root) {
            if (static_cast<int>(leaf) == selectedLeaf &&
                root == selectedRoot) {
                continue;
            }
            const amrex::Real candidate = roots[leaf].roots[root];
            if (candidate >= -tolerance && candidate <= 1.0 + tolerance) {
                result.status = GeometryIntersectionStatus::MultipleHits;
                return result;
            }
        }
    }

    auto const& cylinder = leaves_[static_cast<std::size_t>(selectedLeaf)];
    const amrex::RealArray hit{
        request.fluidPoint[0] + q * dx,
        request.fluidPoint[1] + q * dy,
        request.fluidPoint[2] + q * dz};
    const amrex::Real nx = hit[0] - cylinder.center()[0];
    const amrex::Real ny = hit[1] - cylinder.center()[1];
    const amrex::Real norm = std::hypot(nx, ny);
    if (!(norm > 0.0) || !std::isfinite(norm) ||
        !((nx / norm) * dx + (ny / norm) * dy < 0.0)) {
        return result;
    }
    result.status = GeometryIntersectionStatus::UniqueHit;
    result.method = GeometryIntersectionMethod::AnalyticClosedForm;
    result.quality = GeometryIntersectionQuality::Exact;
    result.provenance = {
        cylinder.boundaryId(), cylinder.objectId(), true, true};
    result.q = q;
    result.intersection = hit;
    result.normalSolidToFluid = {nx / norm, ny / norm, 0.0};
    result.surfaceCoincidentFluid = request.surfaceCoincidentFluid;
    if (!result.contractValid(request)) {
        return {};
    }
    return result;
}

GeometryIntersectionStatus InternalGeometryQueryRegistry::segmentTopology(
    amrex::RealArray const& first,
    amrex::RealArray const& second,
    bool firstOnSurface,
    bool secondOnSurface,
    amrex::Geometry const& geom) const noexcept
{
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(first[d]) || !std::isfinite(second[d])) {
            return GeometryIntersectionStatus::Invalid;
        }
    }
    if (leaves_.empty()) {
        return GeometryIntersectionStatus::Unsupported;
    }
    const amrex::Real transverseLength =
        std::hypot(second[0] - first[0], second[1] - first[1]);
    if (transverseLength == 0.0) {
        return GeometryIntersectionStatus::NoHit;
    }
    if (!std::isfinite(transverseLength)) {
        return GeometryIntersectionStatus::Invalid;
    }

    int hitCount = 0;
    int authorizedEndpointHits = 0;
    const amrex::Real epsilon = std::numeric_limits<amrex::Real>::epsilon();
    for (auto const& cylinder : leaves_) {
        const SurfaceTolerance surface = cylinder.surfaceTolerance(geom);
        if (!surface.valid) {
            return GeometryIntersectionStatus::Invalid;
        }
        const amrex::Real qTolerance = std::max(
            256.0 * epsilon,
            4.0 * surface.physicalLength / transverseLength);
        auto endpointTolerance = [&](amrex::RealArray const& point) {
            const amrex::Real radius = std::hypot(
                point[0] - cylinder.center()[0],
                point[1] - cylinder.center()[1]);
            const amrex::Real radialUncertainty = std::sqrt(std::max(
                amrex::Real(0.0),
                amrex::Real(2.0) * radius * surface.physicalLength +
                    surface.physicalLength * surface.physicalLength));
            return std::max(
                qTolerance, radialUncertainty / transverseLength);
        };
        const amrex::Real firstTolerance = firstOnSurface ?
            endpointTolerance(first) : qTolerance;
        const amrex::Real secondTolerance = secondOnSurface ?
            endpointTolerance(second) : qTolerance;
        const AnalyticCylinderRootSet roots =
            cylinder.exactSegmentRoots(first, second);
        if (roots.status == AnalyticCylinderRootStatus::Invalid) {
            return GeometryIntersectionStatus::Invalid;
        }
        if (roots.status == AnalyticCylinderRootStatus::Tangent) {
            if (roots.tangentQ >= -qTolerance &&
                roots.tangentQ <= 1.0 + qTolerance) {
                if ((firstOnSurface &&
                     std::abs(roots.tangentQ) <= firstTolerance) ||
                    (secondOnSurface &&
                     std::abs(roots.tangentQ - 1.0) <= secondTolerance)) {
                    ++authorizedEndpointHits;
                    continue;
                }
                return GeometryIntersectionStatus::Tangent;
            }
            continue;
        }
        if (roots.status != AnalyticCylinderRootStatus::TwoRoots) {
            continue;
        }
        int interiorHits = 0;
        bool firstEndpointHit = false;
        bool secondEndpointHit = false;
        for (amrex::Real q : roots.roots) {
            const bool atFirstEndpoint =
                firstOnSurface && std::abs(q) <= firstTolerance;
            const bool atSecondEndpoint =
                secondOnSurface && std::abs(q - 1.0) <= secondTolerance;
            if (!atFirstEndpoint && !atSecondEndpoint &&
                (q < -qTolerance || q > 1.0 + qTolerance)) {
                continue;
            }
            if (atFirstEndpoint) {
                firstEndpointHit = true;
            } else if (atSecondEndpoint) {
                secondEndpointHit = true;
            } else {
                ++interiorHits;
            }
        }
        hitCount += interiorHits + static_cast<int>(firstEndpointHit) +
            static_cast<int>(secondEndpointHit);
        authorizedEndpointHits += static_cast<int>(firstEndpointHit) +
            static_cast<int>(secondEndpointHit);
        if (hitCount > 1) {
            return GeometryIntersectionStatus::MultipleHits;
        }
    }
    if (hitCount == 0) {
        return GeometryIntersectionStatus::NoHit;
    }
    return hitCount == 1 && authorizedEndpointHits == 1
        ? GeometryIntersectionStatus::NoHit
        : GeometryIntersectionStatus::UniqueHit;
}

} // namespace lbm::boundary
