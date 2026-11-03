#include "lbm/boundary/domain/DomainBoundaryPolicy.H"

#include <cmath>

namespace lbm::boundary {

void append_boundary_resolve_error(
    std::vector<std::string>& errors,
    std::string const& context,
    std::string const& message)
{
    errors.push_back(context + ": " + message);
}

std::optional<BoundaryScheme> default_domain_scheme(BoundaryPhysical physical)
{
    if (physical == BoundaryPhysical::NoSlipWall ||
        physical == BoundaryPhysical::MovingWall ||
        is_velocity_boundary(physical)) {
        return BoundaryScheme::HalfwayBounceBack;
    }
    if (physical == BoundaryPhysical::FreeOutflow) {
        return BoundaryScheme::GuoExtrapolation;
    }
    return std::nullopt;
}

std::optional<BoundaryProfileKind> resolve_domain_profile(
    DomainBoundarySpec const& spec,
    std::string const& context,
    std::vector<std::string>& errors)
{
    if (spec.profile.has_value()) {
        return *spec.profile;
    }
    if (spec.physical == BoundaryPhysical::NoSlipWall ||
        spec.physical == BoundaryPhysical::Outflow ||
        spec.physical == BoundaryPhysical::FreeOutflow ||
        spec.physical == BoundaryPhysical::Symmetry) {
        return BoundaryProfileKind::None;
    }
    append_boundary_resolve_error(
        errors,
        context,
        "profile is required for physical='" +
            std::string(to_string(spec.physical)) + "'");
    return std::nullopt;
}

void validate_domain_profile_values(
    DomainBoundarySpec const& spec,
    BoundaryProfileKind profile,
    std::string const& context,
    std::vector<std::string>& errors)
{
    auto error = [&](std::string const& message) {
        append_boundary_resolve_error(errors, context, message);
    };
    auto requireFiniteVelocity = [&](char const* message) {
        for (amrex::Real component : spec.prescribedULb) {
            if (!std::isfinite(component)) {
                error(message);
                return;
            }
        }
    };

    if (spec.physical == BoundaryPhysical::NoSlipWall) {
        if (profile != BoundaryProfileKind::None) {
            error("no_slip_wall requires profile='none'");
        }
        return;
    }

    if (spec.physical == BoundaryPhysical::MovingWall) {
        if (profile == BoundaryProfileKind::UserDef) {
            error("user_def wall profiles are not registered");
        } else if (profile != BoundaryProfileKind::Uniform) {
            error("moving_wall requires profile='uniform'");
        } else if (!spec.hasPrescribedULb) {
            error("moving_wall with profile='uniform' requires u_lb");
        } else {
            requireFiniteVelocity("moving_wall requires finite u_lb components");
        }
        return;
    }

    if (is_velocity_boundary(spec.physical)) {
        if (profile == BoundaryProfileKind::UserDef) {
            error("user_def velocity profiles are not registered");
        }
        if (profile == BoundaryProfileKind::Uniform) {
            if (!spec.hasPrescribedULb) {
                error(std::string(to_string(spec.physical)) +
                      " with profile='uniform' requires u_lb");
            } else {
                requireFiniteVelocity(
                    "velocity boundary requires finite u_lb components");
            }
        }
        if (profile == BoundaryProfileKind::Poiseuille) {
            const int normalAxis = domain_side_axis(spec.location);
            if (!spec.hasProfileAxis) {
                error(std::string(to_string(spec.physical)) +
                      " with profile='poiseuille' requires profile_axis");
            } else if (spec.profileAxis == normalAxis) {
                error("poiseuille profile_axis must be tangential to domain face");
            }
            if (spec.hasUMaxLb == spec.hasUMeanLb) {
                error(std::string(to_string(spec.physical)) +
                      " with profile='poiseuille' requires exactly one of "
                      "u_max_lb or u_mean_lb");
            } else {
                const amrex::Real u = spec.hasUMaxLb ? spec.uMaxLb : spec.uMeanLb;
                if (!std::isfinite(u) || !(u > 0.0)) {
                    error("poiseuille velocity scale must be finite and positive");
                } else if (spec.hasUMeanLb && !std::isfinite(1.5 * u)) {
                    error("poiseuille mean-to-peak velocity conversion must be finite");
                }
            }
        }
        if (profile == BoundaryProfileKind::None ||
            profile == BoundaryProfileKind::Constant) {
            error(std::string(to_string(spec.physical)) +
                  " requires a velocity profile");
        }
        return;
    }

    if (is_pressure_boundary(spec.physical)) {
        if (profile == BoundaryProfileKind::UserDef) {
            error("user_def density profiles are not registered");
        }
        if (profile != BoundaryProfileKind::Constant) {
            error(std::string(to_string(spec.physical)) +
                  " requires profile='constant'");
        } else if (!spec.hasRhoLb) {
            error(std::string(to_string(spec.physical)) +
                  " with profile='constant' requires rho_lb");
        } else if (!std::isfinite(spec.rhoLb) || !(spec.rhoLb > 0.0)) {
            error(std::string(to_string(spec.physical)) +
                  " with profile='constant' requires finite positive rho_lb");
        }
        return;
    }

    if (is_prescribed_state_boundary(spec.physical)) {
        if (profile != BoundaryProfileKind::Uniform &&
            profile != BoundaryProfileKind::Poiseuille) {
            error("prescribed_state requires profile='uniform' or 'poiseuille'");
            return;
        }
        if (!spec.hasRhoLb) {
            error("prescribed_state requires rho_lb");
        } else if (!std::isfinite(spec.rhoLb) || !(spec.rhoLb > 0.0)) {
            error("prescribed_state requires finite positive rho_lb");
        }
        if (!spec.hasPrescribedULb) {
            error("prescribed_state requires u_lb");
        } else {
            requireFiniteVelocity("prescribed_state requires finite u_lb components");
        }
        if (spec.hasUMaxLb || spec.hasUMeanLb) {
            error("prescribed_state uses u_lb for the complete velocity; "
                  "omit u_max_lb and u_mean_lb");
        }
        if (profile == BoundaryProfileKind::Uniform && spec.hasProfileAxis) {
            error("prescribed_state with profile='uniform' must omit profile_axis");
        }
        if (profile == BoundaryProfileKind::Poiseuille) {
            const int normalAxis = domain_side_axis(spec.location);
            if (!spec.hasProfileAxis) {
                error("prescribed_state with profile='poiseuille' requires profile_axis");
            } else if (spec.profileAxis == normalAxis) {
                error("poiseuille profile_axis must be tangential to domain face");
            }
            amrex::Real speedSqr = 0.0;
            for (amrex::Real component : spec.prescribedULb) {
                speedSqr += component * component;
            }
            if (spec.hasPrescribedULb &&
                (!std::isfinite(speedSqr) || !(speedSqr > 0.0))) {
                error("prescribed_state poiseuille requires a finite non-zero peak u_lb vector");
            }
        }
        return;
    }

    if ((spec.physical == BoundaryPhysical::Outflow ||
         spec.physical == BoundaryPhysical::FreeOutflow ||
         spec.physical == BoundaryPhysical::Symmetry) &&
        profile != BoundaryProfileKind::None) {
        error(std::string(to_string(spec.physical)) + " requires profile='none'");
    }
}

} // namespace lbm::boundary
