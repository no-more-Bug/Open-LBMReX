#ifndef LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYRESOLVER_HH
#define LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYRESOLVER_HH

#include "lbm/boundary/internal/InternalBoundarySchemeTraits.H"
#include "lbm/boundary/internal/InternalCylinderConstraints.H"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <utility>

namespace lbm::boundary {

template <class Descriptor>
InternalBoundaryResolveResult resolve_internal_boundary_definitions(
    std::vector<InternalBoundarySpec> const& specs,
    amrex::Geometry const& geom,
    int boundaryIdOffset,
    std::set<std::string> const& reservedBoundaryNames,
    std::vector<DomainBoundarySpec> const& domainSpecs)
{
    InternalBoundaryResolveResult result;
    std::set<std::string> boundaryNames = reservedBoundaryNames;
    std::map<std::string, ObjectId> objectIds;
    std::vector<std::vector<BoundaryId>> objectBoundaryLists;

    const bool hasZDomainFace = std::any_of(
        domainSpecs.begin(), domainSpecs.end(), [](DomainBoundarySpec const& spec) {
            return spec.location == DomainSide::ZLo || spec.location == DomainSide::ZHi;
        });

    for (int n = 0; n < static_cast<int>(specs.size()); ++n) {
        InternalBoundarySpec const& spec = specs[static_cast<std::size_t>(n)];
        const std::string context = "boundary.internal." + std::to_string(n) +
            " ('" + spec.name + "')";
        const std::size_t errorCountBefore = result.errors.size();
        auto error = [&](std::string const& message) {
            result.errors.push_back(context + ": " + message);
        };

        if (spec.name.empty()) {
            error("name must not be empty");
        } else if (!boundaryNames.insert(spec.name).second) {
            error("duplicate boundary name '" + spec.name +
                  "' across domain/internal specifications");
        }

        const bool cylinder = spec.geometry == "cylinder";
        if (spec.geometry.empty()) {
            error("geometry must not be empty");
        } else if (spec.geometry == "stl") {
            error("geometry='stl' is reserved but its project adapter is not implemented");
        } else if (!cylinder) {
            error("unsupported internal solid geometry '" + spec.geometry +
                  "'; Stage 4.5 registers only geometry='cylinder'");
        }

        if (!spec.hasPhysical) {
            error("physical is required");
        } else if (spec.physical != BoundaryPhysical::NoSlipWall) {
            error("Stage 4.5 supports only static physical='no_slip_wall'");
        }

        const BoundaryProfileKind profile =
            spec.profile.value_or(BoundaryProfileKind::None);
        if (!supports_internal_boundary_combination(spec.physical, profile)) {
            error("internal static no-slip solid requires profile='none'");
        }

        std::optional<InternalBoundarySchemeTraits> traits;
        if (!spec.scheme.has_value()) {
            error("internal solid scheme is required and has no default");
        } else {
            traits = internal_boundary_scheme_traits(*spec.scheme);
            if (!traits.has_value()) {
                error("scheme='" + std::string(to_string(*spec.scheme)) +
                      "' has no registered internal-solid scheme trait");
            }
        }

        if (cylinder) {
            if (!spec.hasCenterPhys) {
                error("center_phys is required for cylinder");
            } else if (!std::all_of(
                           spec.centerPhys.begin(),
                           spec.centerPhys.end(),
                           [](amrex::Real x) { return std::isfinite(x); })) {
                error("center_phys components must be finite");
            }
            if (!spec.hasRadiusPhys) {
                error("radius_phys is required for cylinder");
            } else if (!std::isfinite(spec.radiusPhys) || !(spec.radiusPhys > 0.0)) {
                error("radius_phys must be finite and positive");
            }
            if (!spec.hasAxis) {
                error("axis is required for cylinder");
            } else if (spec.axis != "z") {
                error("Stage 4.5 quasi-2D cylinder requires axis='z'");
            }
        }

        if constexpr (Descriptor::d == 3) {
            if (cylinder && !geom.isPeriodic(2)) {
                error("D3Q27 quasi-2D cylinder requires periodic z");
            }
            if (cylinder && hasZDomainFace) {
                error("D3Q27 quasi-2D cylinder cannot coexist with z-face "
                      "domain boundary specifications");
            }
        }
        if (result.errors.size() != errorCountBefore || !spec.scheme.has_value() ||
            !traits.has_value()) {
            continue;
        }

        const std::string objectName =
            spec.objectName.empty() ? spec.name : spec.objectName;
        auto [objectIt, inserted] = objectIds.emplace(
            objectName, ObjectId{static_cast<int>(objectIds.size())});
        if (inserted) {
            objectBoundaryLists.emplace_back();
        }

        ResolvedInternalBoundary resolved;
        resolved.boundaryId = BoundaryId{
            boundaryIdOffset + static_cast<int>(result.boundaries.size())};
        resolved.objectId = objectIt->second;
        resolved.name = spec.name;
        resolved.tag = spec.tag.empty() ? spec.name : spec.tag;
        resolved.objectName = objectName;
        resolved.geometry = spec.geometry;
        resolved.physical = spec.physical;
        resolved.scheme = *spec.scheme;
        resolved.schemeOrigin = BoundarySchemeOrigin::User;
        resolved.profile = profile;
        resolved.centerPhys = spec.centerPhys;
        resolved.radiusPhys = spec.radiusPhys;
        resolved.axis = 2;
        result.boundaries.push_back(resolved);
        objectBoundaryLists[static_cast<std::size_t>(objectIt->second.value)]
            .push_back(resolved.boundaryId);
    }

    for (std::size_t lhs = 0; lhs < result.boundaries.size(); ++lhs) {
        auto const& a = result.boundaries[lhs];
        for (int axis = 0; axis < 2; ++axis) {
            if (geom.isPeriodic(axis) && !below_internal_minimum_beyond_roundoff(
                    2.0 * a.radiusPhys, geom.ProbLength(axis), geom.ProbLength(axis))) {
                result.errors.push_back("internal cylinder ('" + a.name +
                    "'): periodic_self_intersection is unsupported");
            }
        }
        for (std::size_t rhs = lhs + 1; rhs < result.boundaries.size(); ++rhs) {
            auto const& b = result.boundaries[rhs];
            const std::string relation = cylinder_pair_constraint_failure(a, b, geom);
            if (!relation.empty()) {
                result.errors.push_back("internal cylinders ('" + a.name + "', '" +
                    b.name + "'): relation='" + relation +
                    "' is unsupported: internal geometries must be mutually disjoint");
            }
        }
    }

    std::vector<std::string> objectNames(objectIds.size());
    for (auto const& [name, objectId] : objectIds) {
        objectNames[static_cast<std::size_t>(objectId.value)] = name;
    }
    for (int index = 0; index < static_cast<int>(objectNames.size()); ++index) {
        ResolvedInternalObject object;
        object.objectId = ObjectId{index};
        object.name = objectNames[static_cast<std::size_t>(index)];
        object.boundaryOffset = static_cast<int>(result.objectBoundaries.size());
        auto const& boundaries = objectBoundaryLists[static_cast<std::size_t>(index)];
        object.boundaryCount = static_cast<int>(boundaries.size());
        result.objectBoundaries.insert(
            result.objectBoundaries.end(), boundaries.begin(), boundaries.end());
        result.objects.push_back(std::move(object));
    }
    return result;
}

} // namespace lbm::boundary

#endif
