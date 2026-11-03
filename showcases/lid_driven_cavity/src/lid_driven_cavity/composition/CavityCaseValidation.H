#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYCASEVALIDATION_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYCASEVALIDATION_H

#include "lid_driven_cavity/observation/CavityCenterlineDiagnostics.H"
#include "lbm/boundary/core/ResolvedBoundary.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/initialization/InitializationConfigIntent.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace lbm::cases::lid_driven_cavity {

struct CavityCaseValidationData {
    amrex::Real lidSpeedLb = 0.0;
    amrex::Real lengthLb = 0.0;
    CavityDiagnosticGeometry diagnosticGeometry;
};

struct CavityCaseValidationResult {
    CavityCaseValidationData data;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

namespace cavity_detail {

inline bool nearly_equal(amrex::Real lhs, amrex::Real rhs) noexcept
{
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const amrex::Real scale = std::max(
        std::max(std::abs(lhs), std::abs(rhs)), amrex::Real(1.0));
    return std::abs(lhs - rhs) <=
           64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

} // namespace cavity_detail

/**
 * @brief Validate the canonical lid-driven-cavity case contract.
 *
 * The validator compares already resolved LBM reference quantities, bulk
 * initialization intent, resolved boundary actual values, and compiled
 * cell-centered geometry. HBB walls are interpreted at valid/ghost faces;
 * Guo walls are interpreted at the outer valid-cell centers. The function
 * returns diagnostic geometry only and never derives or mutates viscosity,
 * reference velocity, reference length, boundary metadata, or runtime fields.
 *
 * @pre Boundary resolution and post-stream plan compilation have succeeded.
 * @post On success the moving lid equals u_ref_lb and both cavity axes equal
 *       the resolved wall-to-wall reference length in lattice and physical units.
 */
inline CavityCaseValidationResult validate_cavity_case(
    config::ResolvedLbmConfig const& lbm,
    initialization::InitializationConfigIntent const& initializationIntent,
    amrex::Geometry const& geom,
    boundary::ResolvedBoundarySet const& resolvedBoundary,
    boundary::PostStreamingBoundaryPlan const& plan)
{
    CavityCaseValidationResult result;
    std::vector<std::string>& errors = result.errors;

    if (initializationIntent.kind.value_or(
            initialization::InitializationKind::TaylorGreen) !=
        initialization::InitializationKind::Rest) {
        errors.push_back(
            "canonical cavity requires rest bulk initialization");
    }
    if (!initializationIntent.rhoLb.has_value()) {
        errors.push_back(
            "canonical cavity requires an explicit initialization density");
    } else if (!cavity_detail::nearly_equal(
                   *initializationIntent.rhoLb, lbm.rhoRefLb)) {
        errors.push_back(
            "canonical cavity initialization density must equal the "
            "resolved LBM reference density");
    }

    if (geom.isPeriodic(0) || geom.isPeriodic(1)) {
        errors.push_back(
            "canonical cavity requires non-periodic x and y directions");
    }
    if (resolvedBoundary.domain.size() != 4) {
        errors.push_back(
            "canonical cavity requires exactly xlo/xhi/ylo/yhi boundaries");
    }

    amrex::Array<int, 4> seen{0, 0, 0, 0};
    bool lidFound = false;
    boundary::BoundaryScheme commonScheme =
        boundary::BoundaryScheme::HalfwayBounceBack;
    bool commonSchemeSet = false;
    for (boundary::ResolvedDomainBoundary const& face :
         resolvedBoundary.domain) {
        const int sideIndex = boundary::domain_side_index(face.side);
        if (sideIndex < 0 || sideIndex >= 4) {
            errors.push_back(
                "canonical cavity supports only xlo/xhi/ylo/yhi faces");
            continue;
        }
        if (seen[sideIndex] != 0) {
            errors.push_back(
                "canonical cavity contains a duplicate " +
                std::string(boundary::to_string(face.side)) + " face");
        }
        seen[sideIndex] = 1;

        if (face.scheme != boundary::BoundaryScheme::HalfwayBounceBack &&
            face.scheme != boundary::BoundaryScheme::GuoExtrapolation) {
            errors.push_back(
                "canonical cavity walls require halfway_bounce_back or "
                "guo_extrapolation");
        } else if (!commonSchemeSet) {
            commonScheme = face.scheme;
            commonSchemeSet = true;
        } else if (face.scheme != commonScheme) {
            errors.push_back(
                "canonical cavity requires all four walls to use one geometry");
        }

        if (face.side == boundary::DomainSide::YHi) {
            if (face.physical != boundary::BoundaryPhysical::MovingWall ||
                face.profile != boundary::BoundaryProfileKind::Uniform ||
                !face.hasPrescribedULb ||
                !std::isfinite(face.prescribedULb[0]) ||
                !(face.prescribedULb[0] > 0.0) ||
                !cavity_detail::nearly_equal(face.prescribedULb[1], 0.0) ||
                !cavity_detail::nearly_equal(face.prescribedULb[2], 0.0)) {
                errors.push_back(
                    "canonical yhi lid must be moving_wall with uniform "
                    "u_lb=(positive U_lid,0,0)");
            } else {
                result.data.lidSpeedLb = face.prescribedULb[0];
                lidFound = true;
            }
        } else if (face.physical !=
                       boundary::BoundaryPhysical::NoSlipWall ||
                   face.profile != boundary::BoundaryProfileKind::None) {
            errors.push_back(
                "canonical xlo/xhi/ylo faces must be stationary "
                "no_slip_wall with profile=none");
        }
    }
    for (int side = 0; side < 4; ++side) {
        if (seen[side] == 0) {
            errors.push_back(
                "canonical cavity is missing " +
                std::string(boundary::to_string(
                    boundary::AllDomainSides[side])) +
                " boundary");
        }
    }
    if (lidFound &&
        !cavity_detail::nearly_equal(
            result.data.lidSpeedLb, lbm.uRefLb)) {
        errors.push_back(
            "moving-lid u_lb must equal resolved lbm.u_ref_lb; the boundary "
            "value does not define the LBM reference velocity");
    }

    const bool onLattice =
        commonSchemeSet &&
        commonScheme == boundary::BoundaryScheme::GuoExtrapolation;
    if (plan.axisGeometry[0].active != onLattice ||
        plan.axisGeometry[1].active != onLattice) {
        errors.push_back(
            "compiled cavity plan geometry does not match the common wall scheme");
    }

    amrex::Box const& domain = geom.Domain();
    result.data.diagnosticGeometry.onLatticeBoundaryNodes =
        onLattice ? 1 : 0;
    for (int d = 0; d < 2; ++d) {
        const int validNodes = domain.length(d);
        result.data.diagnosticGeometry.validNodes[d] = validNodes;
        if (onLattice) {
            boundary::OnLatticeAxisGeometry const& axis =
                plan.axisGeometry[d];
            if (!axis.active || axis.validNodes != validNodes ||
                axis.validNodes != axis.latticeIntervals + 1) {
                errors.push_back(
                    "Guo cavity requires N+1 boundary-inclusive valid nodes "
                    "for N wall-to-wall intervals on both axes");
            }
            result.data.diagnosticGeometry.latticeIntervals[d] =
                axis.latticeIntervals;
            result.data.diagnosticGeometry.physicalLo[d] =
                axis.physicalLo;
            result.data.diagnosticGeometry.physicalHi[d] =
                axis.physicalHi;
        } else {
            result.data.diagnosticGeometry.latticeIntervals[d] =
                validNodes;
            result.data.diagnosticGeometry.physicalLo[d] =
                geom.ProbLo(d);
            result.data.diagnosticGeometry.physicalHi[d] =
                geom.ProbHi(d);
        }
    }

    const amrex::Real xLengthLb = static_cast<amrex::Real>(
        result.data.diagnosticGeometry.latticeIntervals[0]);
    const amrex::Real yLengthLb = static_cast<amrex::Real>(
        result.data.diagnosticGeometry.latticeIntervals[1]);
    const amrex::Real xLengthPhys =
        result.data.diagnosticGeometry.physicalHi[0] -
        result.data.diagnosticGeometry.physicalLo[0];
    const amrex::Real yLengthPhys =
        result.data.diagnosticGeometry.physicalHi[1] -
        result.data.diagnosticGeometry.physicalLo[1];
    result.data.lengthLb = xLengthLb;

    if (!(xLengthLb > 0.0 && yLengthLb > 0.0 &&
          xLengthPhys > 0.0 && yLengthPhys > 0.0) ||
        !cavity_detail::nearly_equal(xLengthLb, yLengthLb) ||
        !cavity_detail::nearly_equal(xLengthPhys, yLengthPhys)) {
        errors.push_back(
            "canonical cavity requires equal positive wall-to-wall x/y lengths");
    }
    if (!cavity_detail::nearly_equal(lbm.lengthRefLb, xLengthLb) ||
        !cavity_detail::nearly_equal(lbm.lengthRefLb, yLengthLb) ||
        !cavity_detail::nearly_equal(lbm.lengthRefPhys, xLengthPhys) ||
        !cavity_detail::nearly_equal(lbm.lengthRefPhys, yLengthPhys)) {
        errors.push_back(
            "resolved LBM reference length must match the canonical cavity "
            "wall-to-wall geometry");
    }

    return result;
}

} // namespace lbm::cases::lid_driven_cavity

#endif
