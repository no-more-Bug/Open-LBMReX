#ifndef LBM_BOUNDARY_CORE_RESOLVEDBOUNDARY_H
#define LBM_BOUNDARY_CORE_RESOLVEDBOUNDARY_H

#include "lbm/boundary/core/BoundaryTypes.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace lbm::boundary {

/** @brief Stable index into the resolved boundary collection for one setup. */
struct ResolvedBoundaryId {
    int value = -1;
};

/** @brief Public vocabulary used by internal metadata for the same stable id. */
using BoundaryId = ResolvedBoundaryId;

/** @brief Stable index reserved for a compiled internal-solid object. */
struct ObjectId {
    int value = -1;
};

/** @brief Records whether a numerical scheme was defaulted or explicitly set. */
enum class BoundarySchemeOrigin {
    Default,
    User
};

inline constexpr std::string_view to_string(BoundarySchemeOrigin value) noexcept
{
    return value == BoundarySchemeOrigin::Default ? "default" : "user";
}

/**
 * @brief Canonical, validated description of one executable domain face.
 *
 * Instances are produced once during setup and remain immutable throughout
 * time advancement. The stage fixes whether the physical boundary is a
 * valid/ghost face (pre-stream HBB) or an outer valid-cell center (post-stream
 * on-lattice reconstruction). Profile values retain lattice-unit actual
 * boundary data and do not own any AMReX field storage.
 */
struct ResolvedDomainBoundary {
    ResolvedBoundaryId boundaryId;
    std::string name;
    std::string tag;
    DomainSide side = DomainSide::XLo;
    BoundaryPhysical physical = BoundaryPhysical::NoSlipWall;
    BoundaryScheme scheme = BoundaryScheme::HalfwayBounceBack;
    BoundarySchemeOrigin schemeOrigin = BoundarySchemeOrigin::Default;
    BoundaryExecutionStage stage = BoundaryExecutionStage::PreStreamingPhysicalGhost;
    BoundaryProfileKind profile = BoundaryProfileKind::None;

    amrex::Array<amrex::Real, 3> prescribedULb{0.0, 0.0, 0.0};
    bool hasPrescribedULb = false;

    amrex::Real rhoLb = 0.0;
    bool hasRhoLb = false;

    int profileAxis = -1;
    bool hasProfileAxis = false;

    amrex::Real uMaxLb = 0.0;
    bool hasUMaxLb = false;
};

/**
 * @brief Canonical, validated description of one internal-solid boundary.
 */
struct ResolvedInternalBoundary {
    BoundaryId boundaryId;
    /** Stable force/diagnostic grouping identity. */
    ObjectId objectId;
    std::string name;
    std::string tag;
    /** Name matching objectId in the flat object table. */
    std::string objectName;
    std::string geometry;
    BoundaryPhysical physical = BoundaryPhysical::NoSlipWall;
    BoundaryScheme scheme = BoundaryScheme::Bouzidi;
    BoundarySchemeOrigin schemeOrigin = BoundarySchemeOrigin::User;
    BoundaryProfileKind profile = BoundaryProfileKind::None;
    amrex::Array<amrex::Real, 3> centerPhys{0.0, 0.0, 0.0};
    amrex::Real radiusPhys = 0.0;
    int axis = 2;
};

/**
 * @brief One physical object in the flat setup-time object registry.
 *
 * The boundary range indexes `ResolvedBoundarySet::internalObjectBoundaries`.
 * One object may group several mutually disjoint boundary specifications for
 * force/diagnostic purposes; Stage 4.5 never creates composite objects.
 */
struct ResolvedInternalObject {
    ObjectId objectId;
    std::string name;
    int boundaryOffset = 0;
    int boundaryCount = 0;
};

/**
 * @brief Setup-time result shared by diagnostics and runtime compilation.
 *
 * The vectors own their strings and scalar metadata. Runtime kernels never
 * capture this host-oriented aggregate directly; DomainFaceTable and compiled
 * region plans provide device-safe representations.
 */
struct ResolvedBoundarySet {
    std::vector<ResolvedDomainBoundary> domain;
    std::vector<ResolvedInternalBoundary> internal;
    std::vector<ResolvedInternalObject> internalObjects;
    std::vector<BoundaryId> internalObjectBoundaries;
    std::vector<std::string> warnings;
};

} // namespace lbm::boundary

#endif
