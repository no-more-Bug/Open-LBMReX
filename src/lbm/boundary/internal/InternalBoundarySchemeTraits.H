#ifndef LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYSCHEMETRAITS_H
#define LBM_BOUNDARY_INTERNAL_INTERNALBOUNDARYSCHEMETRAITS_H

#include "lbm/boundary/core/BoundaryTypes.H"

#include <optional>
#include <string_view>

namespace lbm::boundary {

enum class InternalBoundaryHalfLinkPolicy { PiecewiseAtHalf, Unified };

struct InternalBoundarySchemeTraits {
    BoundaryScheme scheme;
    InternalBoundaryHalfLinkPolicy halfLinkPolicy;
    int transverseClearanceCells;
};

inline constexpr std::optional<InternalBoundarySchemeTraits>
internal_boundary_scheme_traits(BoundaryScheme scheme) noexcept
{
    if (scheme == BoundaryScheme::Bouzidi) {
        return InternalBoundarySchemeTraits{
            scheme,
            InternalBoundaryHalfLinkPolicy::PiecewiseAtHalf,
            3};
    }
    if (scheme == BoundaryScheme::Yms) {
        return InternalBoundarySchemeTraits{
            scheme,
            InternalBoundaryHalfLinkPolicy::Unified,
            3};
    }
    return std::nullopt;
}

inline constexpr bool supports_internal_boundary_combination(
    BoundaryPhysical physical,
    BoundaryProfileKind profile) noexcept
{
    return physical == BoundaryPhysical::NoSlipWall &&
        profile == BoundaryProfileKind::None;
}

inline constexpr std::string_view to_string(
    InternalBoundaryHalfLinkPolicy policy) noexcept
{
    return policy == InternalBoundaryHalfLinkPolicy::PiecewiseAtHalf
        ? "piecewise_at_half"
        : "unified";
}

} // namespace lbm::boundary

#endif
