#include "lbm/boundary/internal/InternalCylinderConstraints.H"
#include "lbm/boundary/internal/InternalBoundarySchemeTraits.H"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lbm::boundary {
namespace {

amrex::Real cylinder_center_distance(
    ResolvedInternalBoundary const& lhs,
    ResolvedInternalBoundary const& rhs,
    amrex::Geometry const& geom) noexcept
{
    amrex::Real delta[2];
    for (int axis = 0; axis < 2; ++axis) {
        delta[axis] = lhs.centerPhys[axis] - rhs.centerPhys[axis];
        if (geom.isPeriodic(axis)) {
            delta[axis] = std::remainder(delta[axis], geom.ProbLength(axis));
        }
    }
    return std::hypot(delta[0], delta[1]);
}

amrex::Real pair_tolerance(
    ResolvedInternalBoundary const& lhs,
    ResolvedInternalBoundary const& rhs) noexcept
{
    const amrex::Real scale = std::max({
        std::abs(lhs.centerPhys[0]),
        std::abs(lhs.centerPhys[1]),
        std::abs(rhs.centerPhys[0]),
        std::abs(rhs.centerPhys[1]),
        lhs.radiusPhys,
        rhs.radiusPhys});
    return amrex::Real(64.0) *
           std::numeric_limits<amrex::Real>::epsilon() * scale;
}

} // namespace

bool below_internal_minimum_beyond_roundoff(
    amrex::Real actual,
    amrex::Real required,
    amrex::Real arithmeticScale) noexcept
{
    const amrex::Real scale = std::max(
        {std::abs(actual), std::abs(required), std::abs(arithmeticScale)});
    const amrex::Real tolerance = amrex::Real(64.0) *
        std::numeric_limits<amrex::Real>::epsilon() * scale;
    return actual < required - tolerance;
}

std::string cylinder_pair_constraint_failure(
    ResolvedInternalBoundary const& lhs,
    ResolvedInternalBoundary const& rhs,
    amrex::Geometry const& geom)
{
    const amrex::Real distance = cylinder_center_distance(lhs, rhs, geom);
    const amrex::Real radiusSum = lhs.radiusPhys + rhs.radiusPhys;
    const amrex::Real radiusDifference =
        std::abs(lhs.radiusPhys - rhs.radiusPhys);
    const amrex::Real tolerance = pair_tolerance(lhs, rhs);

    if (distance <= tolerance && radiusDifference <= tolerance) {
        return "coincident";
    }
    if (std::abs(distance - radiusSum) <= tolerance ||
        std::abs(distance - radiusDifference) <= tolerance) {
        return "tangent";
    }
    if (distance < radiusDifference - tolerance) {
        return "containment";
    }
    if (distance < radiusSum - tolerance) {
        return "partial_overlap";
    }

    return {};
}

std::vector<std::string> validate_internal_boundaries_on_level(
    ResolvedBoundarySet const& resolved,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    amrex::Geometry const& geom)
{
    std::vector<std::string> errors;
    std::vector<ResolvedInternalBoundary const*> owned;
    const amrex::Real hMax = std::max(geom.CellSize(0), geom.CellSize(1));
    for (auto const& boundary : resolved.internal) {
        if (std::none_of(ownedBoundaryIds.begin(), ownedBoundaryIds.end(),
                [&](BoundaryId id) { return id.value == boundary.boundaryId.value; })) {
            continue;
        }
        owned.push_back(&boundary);
        const std::string context = "internal cylinder ('" + boundary.name + "'): ";
        if (below_internal_minimum_beyond_roundoff(boundary.radiusPhys, 4.0 * hMax, hMax)) {
            errors.push_back(context + "minimum cylinder resolution requires R/h_max >= 4 (D/h_max >= 8)");
        }
        const int clearanceCells = internal_boundary_scheme_traits(boundary.scheme)->transverseClearanceCells;
        for (int axis = 0; axis < 2; ++axis) {
            const amrex::Real required = clearanceCells * geom.CellSize(axis);
            if (geom.isPeriodic(axis)) {
                continue;
            }
            const amrex::Real low = boundary.centerPhys[axis] - boundary.radiusPhys - geom.ProbLo(axis);
            const amrex::Real high = geom.ProbHi(axis) - boundary.centerPhys[axis] - boundary.radiusPhys;
            const amrex::Real lowScale = std::max({std::abs(boundary.centerPhys[axis]), boundary.radiusPhys, std::abs(geom.ProbLo(axis))});
            const amrex::Real highScale = std::max({std::abs(boundary.centerPhys[axis]), boundary.radiusPhys, std::abs(geom.ProbHi(axis))});
            const std::string message = context + "cylinder surface must remain at least " +
                std::to_string(clearanceCells) + " cells from non-periodic " + std::string(axis_to_string(axis));
            if (below_internal_minimum_beyond_roundoff(low, required, lowScale)) {
                errors.push_back(message + "lo face");
            }
            if (below_internal_minimum_beyond_roundoff(high, required, highScale)) {
                errors.push_back(message + "hi face");
            }
        }
    }
    for (std::size_t lhs = 0; lhs < owned.size(); ++lhs) {
        for (std::size_t rhs = lhs + 1; rhs < owned.size(); ++rhs) {
            auto const& a = *owned[lhs];
            auto const& b = *owned[rhs];
            const int clearanceCells = std::max(
                internal_boundary_scheme_traits(a.scheme)->transverseClearanceCells,
                internal_boundary_scheme_traits(b.scheme)->transverseClearanceCells);
            const amrex::Real distance = cylinder_center_distance(a, b, geom);
            const amrex::Real radiusSum = a.radiusPhys + b.radiusPhys;
            const amrex::Real required = clearanceCells * hMax;
            if (below_internal_minimum_beyond_roundoff(
                    distance - radiusSum, required, std::max({distance, radiusSum, required}))) {
                errors.push_back("internal cylinders ('" + a.name + "', '" + b.name +
                    "'): relation='insufficient_clearance': surfaces must remain at least " +
                    std::to_string(clearanceCells) + " transverse cells apart on their owning level");
            }
        }
    }
    return errors;
}

} // namespace lbm::boundary
