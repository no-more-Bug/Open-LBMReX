#ifndef LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYRESOLVER_HH
#define LBM_BOUNDARY_DOMAIN_DOMAINBOUNDARYRESOLVER_HH

#include "lbm/boundary/core/DomainBoundaryCompatibility.H"
#include "lbm/boundary/domain/DomainBoundaryPolicy.H"

#include <cmath>
#include <optional>
#include <set>
#include <string>

namespace lbm::boundary {

template <class Descriptor>
BoundaryResolveResult resolve_domain_boundary_specs(
    std::vector<DomainBoundarySpec> const& specs,
    amrex::Geometry const& geom)
{
    BoundaryResolveResult result;
    std::set<std::string> names;
    std::set<int> locations;

    for (int n = 0; n < static_cast<int>(specs.size()); ++n) {
        DomainBoundarySpec const& spec = specs[static_cast<std::size_t>(n)];
        const std::string context = "boundary.domain." + std::to_string(n) +
            " ('" + spec.name + "')";
        const std::size_t errorCountBefore = result.errors.size();
        auto error = [&](std::string const& message) {
            append_boundary_resolve_error(result.errors, context, message);
        };

        if (spec.name.empty()) {
            error("name must not be empty");
        } else if (!names.insert(spec.name).second) {
            error("duplicate boundary name '" + spec.name + "'");
        }

        const int sideIndex = domain_side_index(spec.location);
        if (!locations.insert(sideIndex).second) {
            error("duplicate domain location '" +
                  std::string(to_string(spec.location)) + "'");
        }

        const int axis = domain_side_axis(spec.location);
        if (axis >= Descriptor::d) {
            error(std::string(Descriptor::name()) +
                  " does not support domain location '" +
                  std::string(to_string(spec.location)) + "'");
        }
        if (axis < AMREX_SPACEDIM && geom.isPeriodic(axis)) {
            error("periodic direction cannot also appear in boundary.domain list at '" +
                  std::string(to_string(spec.location)) + "'");
        }

        auto profile = resolve_domain_profile(spec, context, result.errors);
        if (profile.has_value()) {
            validate_domain_profile_values(spec, *profile, context, result.errors);
            if (*profile == BoundaryProfileKind::Poiseuille &&
                spec.hasProfileAxis &&
                (spec.profileAxis < 0 || spec.profileAxis >= Descriptor::d)) {
                error(std::string(Descriptor::name()) +
                      " does not support poiseuille profile_axis=" +
                      std::to_string(spec.profileAxis));
            }
            if ((is_velocity_boundary(spec.physical) ||
                 is_prescribed_state_boundary(spec.physical) ||
                 spec.physical == BoundaryPhysical::MovingWall) &&
                spec.hasPrescribedULb) {
                for (int velocityAxis = Descriptor::d; velocityAxis < 3;
                     ++velocityAxis) {
                    if (std::abs(spec.prescribedULb[velocityAxis]) > 0.0) {
                        error(std::string(Descriptor::name()) +
                              " does not support " +
                              (spec.physical == BoundaryPhysical::MovingWall
                                   ? "moving_wall "
                                   : is_prescribed_state_boundary(spec.physical)
                                       ? "prescribed "
                                       : "velocity ") +
                              "u_lb component on axis '" +
                              std::string(axis_to_string(velocityAxis)) + "'");
                    }
                }
            }
        }

        BoundarySchemeOrigin schemeOrigin = BoundarySchemeOrigin::User;
        std::optional<BoundaryScheme> scheme = spec.scheme;
        if (!scheme.has_value()) {
            scheme = default_domain_scheme(spec.physical);
            schemeOrigin = BoundarySchemeOrigin::Default;
            if (!scheme.has_value()) {
                error("no default domain boundary scheme for physical='" +
                      std::string(to_string(spec.physical)) + "'");
            }
        }
        if (scheme.has_value() && profile.has_value() &&
            !is_supported_domain_boundary_combination(
                spec.physical, *scheme, *profile)) {
            error("scheme='" + std::string(to_string(*scheme)) +
                  "', profile='" + std::string(to_string(*profile)) +
                  "' is not supported for domain physical='" +
                  std::string(to_string(spec.physical)) +
                  "' in the current implementation");
        }
        if (scheme == BoundaryScheme::GuoExtrapolation &&
            profile == BoundaryProfileKind::Poiseuille &&
            spec.hasProfileAxis && spec.profileAxis >= 0 &&
            spec.profileAxis < Descriptor::d &&
            geom.Domain().length(spec.profileAxis) < 2) {
            error("Guo poiseuille profile_axis='" +
                  std::string(axis_to_string(spec.profileAxis)) +
                  "' requires at least 2 valid nodes");
        }
        const std::optional<BoundaryExecutionStage> stage =
            scheme.has_value() ? execution_stage_for_scheme(*scheme) : std::nullopt;
        if (scheme.has_value() && !stage.has_value()) {
            error("scheme='" + std::string(to_string(*scheme)) +
                  "' has no executable domain-boundary stage");
        }

        if (result.errors.size() != errorCountBefore || !profile.has_value() ||
            !scheme.has_value() || !stage.has_value()) {
            continue;
        }

        if (spec.physical == BoundaryPhysical::MovingWall &&
            std::abs(spec.prescribedULb[axis]) > 0.0) {
            result.resolved.warnings.push_back(
                context + ": moving_wall has non-zero normal u_lb component on " +
                std::string(to_string(spec.location)));
        }

        ResolvedDomainBoundary resolved;
        resolved.boundaryId =
            ResolvedBoundaryId{static_cast<int>(result.resolved.domain.size())};
        resolved.name = spec.name;
        resolved.tag = spec.tag.empty() ? spec.name : spec.tag;
        resolved.side = spec.location;
        resolved.physical = spec.physical;
        resolved.scheme = *scheme;
        resolved.schemeOrigin = schemeOrigin;
        resolved.stage = *stage;
        resolved.profile = *profile;
        resolved.prescribedULb = spec.prescribedULb;
        resolved.hasPrescribedULb = spec.hasPrescribedULb;
        resolved.rhoLb = spec.rhoLb;
        resolved.hasRhoLb = spec.hasRhoLb;
        resolved.profileAxis = spec.profileAxis;
        resolved.hasProfileAxis = spec.hasProfileAxis;
        if (*profile == BoundaryProfileKind::Poiseuille) {
            if (is_prescribed_state_boundary(spec.physical)) {
                amrex::Real speedSqr = 0.0;
                for (amrex::Real component : spec.prescribedULb) {
                    speedSqr += component * component;
                }
                resolved.uMaxLb = std::sqrt(speedSqr);
            } else {
                resolved.uMaxLb =
                    spec.hasUMaxLb ? spec.uMaxLb : 1.5 * spec.uMeanLb;
            }
            resolved.hasUMaxLb = true;
            if (!is_prescribed_state_boundary(spec.physical) && spec.hasUMeanLb) {
                result.resolved.warnings.push_back(
                    context + ": converted u_mean_lb=" +
                    std::to_string(spec.uMeanLb) + " to u_max_lb=" +
                    std::to_string(resolved.uMaxLb));
            }
        }
        result.resolved.domain.push_back(std::move(resolved));
    }

    for (int lhs = 0; lhs < static_cast<int>(result.resolved.domain.size()); ++lhs) {
        auto const& a = result.resolved.domain[static_cast<std::size_t>(lhs)];
        for (int rhs = lhs + 1;
             rhs < static_cast<int>(result.resolved.domain.size()); ++rhs) {
            auto const& b = result.resolved.domain[static_cast<std::size_t>(rhs)];
            if (domain_side_axis(a.side) == domain_side_axis(b.side)) {
                if (a.stage != b.stage) {
                    append_boundary_resolve_error(
                        result.errors,
                        "opposite domain faces ('" + a.name + "', '" + b.name + "')",
                        "cannot mix valid-node and domain-face geometry on the same axis");
                }
                continue;
            }

            const std::string context =
                "domain edge/corner ('" + a.name + "', '" + b.name + "')";
            if (a.stage != b.stage) {
                append_boundary_resolve_error(
                    result.errors,
                    context,
                    "cannot intersect stage='" + std::string(to_string(a.stage)) +
                        "' with stage='" + std::string(to_string(b.stage)) + "'");
            } else if (a.stage == BoundaryExecutionStage::PostStreamingValidNode &&
                       a.physical != BoundaryPhysical::NoSlipWall &&
                       b.physical != BoundaryPhysical::NoSlipWall) {
                append_boundary_resolve_error(
                    result.errors,
                    context,
                    "post-stream Guo edge/corner requires wall-wins or a single "
                    "non-wall physical boundary");
            }
        }
    }

    return result;
}

} // namespace lbm::boundary

#endif
