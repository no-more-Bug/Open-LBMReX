#ifndef LBM_BOUNDARY_CORE_BOUNDARYTYPES_H
#define LBM_BOUNDARY_CORE_BOUNDARYTYPES_H

#include <AMReX_GpuQualifiers.H>

#include <array>
#include <optional>
#include <string_view>

namespace lbm::boundary {

/**
 * @brief Physical intent prescribed by the input boundary specification.
 *
 * This enum describes the macroscopic condition only. Its executable geometry
 * and place in the LBM step are determined later by BoundaryScheme and
 * BoundaryExecutionStage.
 */
enum class BoundaryPhysical {
    NoSlipWall,
    MovingWall,
    VelocityInlet,
    VelocityOutlet,
    PressureInlet,
    PressureOutlet,
    PrescribedState,
    Outflow,
    FreeOutflow,
    Symmetry
};

/**
 * @brief Numerical closure used to produce boundary populations.
 *
 * Resolution validates the physical/profile combination and assigns a stage
 * supported by the current domain-boundary runtime.
 */
enum class BoundaryScheme {
    HalfwayBounceBack,
    AntiBounceBack,
    Bouzidi,
    Yms,
    ZouHe,
    GuoExtrapolation
};

/**
 * @brief Shape of the macroscopic data supplied to a physical boundary.
 *
 * Profile values are boundary-owned actual values, not global LBM reference
 * quantities, and must not be used to derive viscosity or relaxation data.
 */
enum class BoundaryProfileKind {
    None,
    Constant,
    Uniform,
    Poiseuille,
    UserDef
};

/**
 * @brief Population time-step location at which a domain scheme executes.
 *
 * PreStreamingPhysicalGhost represents a physical boundary on the face between
 * a valid cell and its ghost. PostStreamingValidNode represents an on-lattice
 * physical boundary coincident with the outer valid cell center.
 */
enum class BoundaryExecutionStage {
    PreStreamingPhysicalGhost,
    PostStreamingValidNode
};

/**
 * @brief One oriented face of the three-dimensional AMReX index domain.
 *
 * Descriptor dimensionality is validated separately: D2Q9 cannot resolve the
 * z faces even though AMREX_SPACEDIM is three for this executable.
 */
enum class DomainSide {
    XLo,
    XHi,
    YLo,
    YHi,
    ZLo,
    ZHi
};

inline constexpr int DomainSideCount = 6;

AMREX_GPU_HOST_DEVICE
inline constexpr bool is_velocity_boundary(BoundaryPhysical physical) noexcept
{
    return physical == BoundaryPhysical::VelocityInlet ||
           physical == BoundaryPhysical::VelocityOutlet;
}

inline constexpr bool is_pressure_boundary(BoundaryPhysical physical) noexcept
{
    return physical == BoundaryPhysical::PressureInlet ||
           physical == BoundaryPhysical::PressureOutlet;
}

inline constexpr bool is_prescribed_state_boundary(
    BoundaryPhysical physical) noexcept
{
    return physical == BoundaryPhysical::PrescribedState;
}

inline constexpr std::array<DomainSide, DomainSideCount> AllDomainSides{
    DomainSide::XLo,
    DomainSide::XHi,
    DomainSide::YLo,
    DomainSide::YHi,
    DomainSide::ZLo,
    DomainSide::ZHi};

inline constexpr int domain_side_index(DomainSide side) noexcept
{
    return side == DomainSide::XLo ? 0 :
           side == DomainSide::XHi ? 1 :
           side == DomainSide::YLo ? 2 :
           side == DomainSide::YHi ? 3 :
           side == DomainSide::ZLo ? 4 : 5;
}

inline constexpr int domain_side_axis(DomainSide side) noexcept
{
    return side == DomainSide::XLo || side == DomainSide::XHi ? 0 :
           side == DomainSide::YLo || side == DomainSide::YHi ? 1 : 2;
}

inline constexpr bool domain_side_is_low(DomainSide side) noexcept
{
    return side == DomainSide::XLo || side == DomainSide::YLo || side == DomainSide::ZLo;
}

inline constexpr std::string_view to_string(BoundaryPhysical value) noexcept
{
    switch (value) {
    case BoundaryPhysical::NoSlipWall:
        return "no_slip_wall";
    case BoundaryPhysical::MovingWall:
        return "moving_wall";
    case BoundaryPhysical::VelocityInlet:
        return "velocity_inlet";
    case BoundaryPhysical::VelocityOutlet:
        return "velocity_outlet";
    case BoundaryPhysical::PressureInlet:
        return "pressure_inlet";
    case BoundaryPhysical::PressureOutlet:
        return "pressure_outlet";
    case BoundaryPhysical::PrescribedState:
        return "prescribed_state";
    case BoundaryPhysical::Outflow:
        return "outflow";
    case BoundaryPhysical::FreeOutflow:
        return "free_outflow";
    case BoundaryPhysical::Symmetry:
        return "symmetry";
    }
    return "unknown";
}

inline constexpr std::string_view to_string(BoundaryScheme value) noexcept
{
    switch (value) {
    case BoundaryScheme::HalfwayBounceBack:
        return "halfway_bounce_back";
    case BoundaryScheme::AntiBounceBack:
        return "anti_bounce_back";
    case BoundaryScheme::Bouzidi:
        return "bouzidi";
    case BoundaryScheme::Yms:
        return "yms";
    case BoundaryScheme::ZouHe:
        return "zou_he";
    case BoundaryScheme::GuoExtrapolation:
        return "guo_extrapolation";
    }
    return "unknown";
}

inline constexpr std::string_view to_string(BoundaryProfileKind value) noexcept
{
    return value == BoundaryProfileKind::None ? "none" :
           value == BoundaryProfileKind::Constant ? "constant" :
           value == BoundaryProfileKind::Uniform ? "uniform" :
           value == BoundaryProfileKind::Poiseuille ? "poiseuille" : "user_def";
}

inline constexpr std::string_view to_string(BoundaryExecutionStage value) noexcept
{
    return value == BoundaryExecutionStage::PreStreamingPhysicalGhost ?
        "pre_stream_physical_ghost" : "post_stream_valid";
}

/**
 * @brief Return the runtime stage implemented for a numerical scheme.
 *
 * @return The executable stage for a currently implemented domain scheme, or
 *         std::nullopt for a recognized-but-not-executable scheme.
 */
inline constexpr std::optional<BoundaryExecutionStage>
execution_stage_for_scheme(BoundaryScheme scheme) noexcept
{
    if (scheme == BoundaryScheme::HalfwayBounceBack) {
        return BoundaryExecutionStage::PreStreamingPhysicalGhost;
    }
    if (scheme == BoundaryScheme::GuoExtrapolation) {
        return BoundaryExecutionStage::PostStreamingValidNode;
    }
    return std::nullopt;
}

inline constexpr std::string_view to_string(DomainSide value) noexcept
{
    return value == DomainSide::XLo ? "xlo" :
           value == DomainSide::XHi ? "xhi" :
           value == DomainSide::YLo ? "ylo" :
           value == DomainSide::YHi ? "yhi" :
           value == DomainSide::ZLo ? "zlo" : "zhi";
}

inline constexpr std::string_view axis_to_string(int axis) noexcept
{
    return axis == 0 ? "x" :
           axis == 1 ? "y" :
           axis == 2 ? "z" : "unknown";
}

} // namespace lbm::boundary

#endif
