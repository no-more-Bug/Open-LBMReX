#ifndef LBM_BOUNDARY_CORE_BOUNDARYSPEC_H
#define LBM_BOUNDARY_CORE_BOUNDARYSPEC_H

#include "lbm/boundary/core/BoundaryTypes.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <optional>
#include <string>

namespace lbm::boundary {

/**
 * @brief Typed but unresolved input for one oriented domain face.
 *
 * Presence flags preserve whether the user supplied an optional value; zero
 * defaults are never interpreted as input. Resolution later supplies scheme
 * and profile defaults, validates combinations, and converts u_mean_lb to the
 * canonical peak velocity when needed. Values here are actual boundary data,
 * not global LBM reference quantities.
 */
struct DomainBoundarySpec {
    std::string name;
    DomainSide location = DomainSide::XLo;
    BoundaryPhysical physical = BoundaryPhysical::NoSlipWall;
    std::optional<BoundaryScheme> scheme;
    std::optional<BoundaryProfileKind> profile;
    std::string tag;

    amrex::Array<amrex::Real, 3> prescribedULb{0.0, 0.0, 0.0};
    bool hasPrescribedULb = false;

    amrex::Real rhoLb = 0.0;
    bool hasRhoLb = false;

    int profileAxis = -1;
    bool hasProfileAxis = false;

    amrex::Real uMaxLb = 0.0;
    bool hasUMaxLb = false;

    amrex::Real uMeanLb = 0.0;
    bool hasUMeanLb = false;
};

/**
 * @brief Typed but unresolved input for one internal-solid boundary.
 *
 * Required-field presence is retained separately from numeric values so direct
 * programs and focused tests share the same semantic resolver. Geometry
 * construction and population execution remain later setup stages.
 */
struct InternalBoundarySpec {
    std::string name;
    std::string geometry;
    BoundaryPhysical physical = BoundaryPhysical::NoSlipWall;
    bool hasPhysical = false;
    std::optional<BoundaryScheme> scheme;
    std::optional<BoundaryProfileKind> profile;
    std::string tag;
    std::string objectName;

    amrex::Array<amrex::Real, 3> centerPhys{0.0, 0.0, 0.0};
    bool hasCenterPhys = false;
    amrex::Real radiusPhys = 0.0;
    bool hasRadiusPhys = false;
    std::string axis;
    bool hasAxis = false;
};

} // namespace lbm::boundary

#endif
