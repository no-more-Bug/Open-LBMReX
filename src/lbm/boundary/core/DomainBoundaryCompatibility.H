#ifndef LBM_BOUNDARY_CORE_DOMAINBOUNDARYCOMPATIBILITY_H
#define LBM_BOUNDARY_CORE_DOMAINBOUNDARYCOMPATIBILITY_H

#include "lbm/boundary/core/BoundaryTypes.H"

namespace lbm::boundary {

/**
 * @brief Test the domain physical/scheme/profile combinations implemented by
 *        the current executable.
 *
 * This is the single compatibility policy shared by setup-time resolution and
 * runtime-metadata invariant checks. Recognized enum values not listed here
 * remain fail-fast rather than reaching a kernel without an implementation.
 *
 * Profile value/presence checks remain the resolver's responsibility; this
 * predicate deliberately preserves the executable-family acceptance matrix
 * that preceded its extraction.
 */
inline constexpr bool is_supported_domain_boundary_combination(
    BoundaryPhysical physical,
    BoundaryScheme scheme,
    BoundaryProfileKind profile) noexcept
{
    if (scheme == BoundaryScheme::HalfwayBounceBack) {
        if (physical == BoundaryPhysical::NoSlipWall ||
            physical == BoundaryPhysical::MovingWall) {
            return true;
        }
        return is_velocity_boundary(physical) &&
               (profile == BoundaryProfileKind::Uniform ||
                profile == BoundaryProfileKind::Poiseuille);
    }

    if (scheme != BoundaryScheme::GuoExtrapolation) {
        return false;
    }
    if (physical == BoundaryPhysical::NoSlipWall) {
        return profile == BoundaryProfileKind::None;
    }
    if (physical == BoundaryPhysical::MovingWall) {
        return profile == BoundaryProfileKind::Uniform;
    }
    if (is_velocity_boundary(physical)) {
        return profile == BoundaryProfileKind::Uniform ||
               profile == BoundaryProfileKind::Poiseuille;
    }
    if (is_pressure_boundary(physical)) {
        return profile == BoundaryProfileKind::Constant;
    }
    if (is_prescribed_state_boundary(physical)) {
        return profile == BoundaryProfileKind::Uniform ||
               profile == BoundaryProfileKind::Poiseuille;
    }
    return physical == BoundaryPhysical::FreeOutflow &&
           profile == BoundaryProfileKind::None;
}

} // namespace lbm::boundary

#endif
