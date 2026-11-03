#ifndef LBM_CASES_POISEUILLEFLOW_CASEVALIDATION_H
#define LBM_CASES_POISEUILLEFLOW_CASEVALIDATION_H

#include "poiseuille_flow/observation/PoiseuilleDiagnostics.H"
#include "lbm/boundary/core/ResolvedBoundary.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/initialization/PoiseuilleInitializationParams.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace lbm::cases::poiseuille_flow {

/**
 * @brief Read-only diagnostic and acceptance data for a canonical Poiseuille run.
 *
 * This value is produced during case preparation after LBM,
 * initialization, boundary, and geometry resolution. It contains no
 * population field and cannot alter any numerical runtime setup.
 */
struct CaseData {
    PoiseuilleDiagnosticGeometry diagnosticGeometry;
    PoiseuillePressureDiagnosticConfig pressureDiagnostic;
    std::string pressureScaling = "fixed_re_fixed_tau";
};

struct CaseValidationResult {
    CaseData data;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

namespace validation_detail {

inline bool nearly_equal(amrex::Real lhs, amrex::Real rhs) noexcept
{
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const amrex::Real scale = std::max(
        std::max(std::abs(lhs), std::abs(rhs)), amrex::Real(1.0));
    return std::abs(lhs - rhs) <=
           512.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

inline bool reference_velocity_equal(
    amrex::Real actual,
    amrex::Real expected) noexcept
{
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return false;
    }
    const amrex::Real scale =
        std::max(amrex::Real(1.0), std::abs(expected));
    return std::abs(actual - expected) <=
           16.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

inline bool is_supported_wall_scheme(boundary::BoundaryScheme scheme) noexcept
{
    return scheme == boundary::BoundaryScheme::HalfwayBounceBack ||
           scheme == boundary::BoundaryScheme::GuoExtrapolation;
}

inline PoiseuilleDiagnosticGeometry
compile_diagnostic_geometry(
    amrex::Geometry const& geom,
    boundary::PostStreamingBoundaryPlan const& plan,
    std::vector<std::string>& errors)
{
    constexpr int streamwiseAxis = 0;
    constexpr int profileAxis = 1;
    PoiseuilleDiagnosticGeometry output;
    amrex::Box const& domain = geom.Domain();

    output.streamwiseIndexLo = domain.smallEnd(streamwiseAxis);
    output.streamwiseIndexHi = domain.bigEnd(streamwiseAxis);
    output.streamwiseValidNodes = domain.length(streamwiseAxis);
    output.profileAxis = profileAxis;
    output.profileIndexLo = domain.smallEnd(profileAxis);
    output.profileIndexHi = domain.bigEnd(profileAxis);
    output.profileValidNodes = domain.length(profileAxis);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        output.trapezoidalAxes[d] =
            plan.axisGeometry[d].active ? 1 : 0;
    }

    auto compile_axis = [&](
        int axis,
        int& onLattice,
        int& latticeIntervals,
        amrex::Real& physicalLo,
        amrex::Real& physicalHi,
        char const* label) {
        boundary::OnLatticeAxisGeometry const& axisGeometry =
            plan.axisGeometry[axis];
        if (!axisGeometry.active) {
            onLattice = 0;
            latticeIntervals = domain.length(axis);
            physicalLo = geom.ProbLo(axis);
            physicalHi = geom.ProbHi(axis);
            return;
        }

        onLattice = 1;
        latticeIntervals = axisGeometry.latticeIntervals;
        physicalLo = axisGeometry.physicalLo;
        physicalHi = axisGeometry.physicalHi;
        if (axisGeometry.validNodes != domain.length(axis) ||
            axisGeometry.validNodes != axisGeometry.latticeIntervals + 1 ||
            axisGeometry.latticeIntervals <= 0) {
            errors.push_back(
                std::string("on-lattice Poiseuille ") + label +
                " geometry must satisfy valid_nodes=lattice_intervals+1");
        }
    };

    compile_axis(
        streamwiseAxis,
        output.streamwiseOnLatticeBoundaryNodes,
        output.streamwiseLatticeIntervals,
        output.streamwisePhysicalLo,
        output.streamwisePhysicalHi,
        "streamwise");
    compile_axis(
        profileAxis,
        output.onLatticeBoundaryNodes,
        output.profileLatticeIntervals,
        output.profilePhysicalLo,
        output.profilePhysicalHi,
        "profile");
    return output;
}

inline bool valid_uniform_peak(
    boundary::ResolvedDomainBoundary const& face,
    amrex::Real expected) noexcept
{
    return face.hasPrescribedULb && face.hasUMaxLb &&
           reference_velocity_equal(face.prescribedULb[0], expected) &&
           reference_velocity_equal(face.prescribedULb[1], 0.0) &&
           reference_velocity_equal(face.prescribedULb[2], 0.0) &&
           reference_velocity_equal(face.uMaxLb, expected);
}

} // namespace validation_detail

/**
 * @brief Validate the canonical Poiseuille case without mutation.
 *
 * The validator compares an already resolved LBM reference state, the complete
 * bulk initialization setup, resolved boundary actual values, and the compiled
 * cell-centered geometry. It accepts the existing velocity, pressure, and
 * prescribed-state drive layouts and returns only diagnostic/acceptance data.
 * It never supplies data to the initializer or boundary runtime.
 *
 * HBB walls lie on valid/ghost faces and therefore use N valid cells for N
 * wall-to-wall intervals. Guo walls coincide with the outer valid-cell centers
 * and use N+1 valid nodes for N intervals. In both cases the transverse
 * wall-to-wall interval must equal the resolved LBM reference length.
 *
 * @pre Boundary resolution, plan compilation, and initialization resolution
 *      succeeded for the same Geometry.
 * @post All input arguments retain their original values.
 */
template <class Descriptor>
CaseValidationResult validate_case(
    config::ResolvedLbmConfig const& lbm,
    initialization::PoiseuilleInitializationParams const& initial,
    amrex::Geometry const& geom,
    boundary::ResolvedBoundarySet const& resolvedBoundary,
    boundary::PostStreamingBoundaryPlan const& plan,
    std::string const& pressureScaling)
{
    using namespace validation_detail;

    CaseValidationResult result;
    result.data.pressureScaling = pressureScaling;
    std::vector<std::string>& errors = result.errors;

    if (!nearly_equal(initial.rhoLb, lbm.rhoRefLb)) {
        errors.push_back(
            "Poiseuille initialization rho_lb must equal lbm.rho_ref_lb");
    }
    if (!reference_velocity_equal(initial.uMaxLb, lbm.uRefLb)) {
        errors.push_back(
            "Poiseuille initialization u_max_lb must equal lbm.u_ref_lb");
    }

    if (geom.isPeriodic(0) || geom.isPeriodic(1)) {
        errors.push_back(
            "canonical Poiseuille requires non-periodic x and y directions");
    }
    amrex::Box const& domain = geom.Domain();
    if (domain.length(0) < 2 || domain.length(1) < 2) {
        errors.push_back(
            "canonical Poiseuille requires at least two valid cells in x and y");
    }

    result.data.diagnosticGeometry =
        compile_diagnostic_geometry(geom, plan, errors);
    PoiseuilleDiagnosticGeometry const& diagnosticGeometry =
        result.data.diagnosticGeometry;

    constexpr amrex::Real standardStreamwiseLo = 0.0;
    constexpr amrex::Real standardStreamwiseHi = 4.0;
    constexpr amrex::Real standardProfileLo = 0.0;
    constexpr amrex::Real standardProfileHi = 1.0;
    if (!nearly_equal(
            diagnosticGeometry.streamwisePhysicalLo,
            standardStreamwiseLo) ||
        !nearly_equal(
            diagnosticGeometry.streamwisePhysicalHi,
            standardStreamwiseHi) ||
        !nearly_equal(
            diagnosticGeometry.profilePhysicalLo,
            standardProfileLo) ||
        !nearly_equal(
            diagnosticGeometry.profilePhysicalHi,
            standardProfileHi)) {
        errors.push_back(
            "canonical Poiseuille physical boundary coordinates must match "
            "the built-in initialization profile domain [0,4] x [0,1]");
    }

    const amrex::Real heightLb = static_cast<amrex::Real>(
        diagnosticGeometry.profileLatticeIntervals);
    const amrex::Real heightPhys =
        diagnosticGeometry.profilePhysicalHi -
        diagnosticGeometry.profilePhysicalLo;
    if (!(heightLb > 0.0 && heightPhys > 0.0) ||
        !nearly_equal(lbm.lengthRefLb, heightLb) ||
        !nearly_equal(lbm.lengthRefPhys, heightPhys)) {
        errors.push_back(
            "resolved LBM reference length must equal the canonical "
            "Poiseuille wall-to-wall y geometry");
    }

    if (resolvedBoundary.domain.size() != 4) {
        errors.push_back(
            "canonical Poiseuille requires exactly xlo/xhi/ylo/yhi boundaries");
    }
    std::array<boundary::ResolvedDomainBoundary const*, 4> faces{
        nullptr, nullptr, nullptr, nullptr};
    for (boundary::ResolvedDomainBoundary const& face :
         resolvedBoundary.domain) {
        const int sideIndex = boundary::domain_side_index(face.side);
        if (sideIndex < 0 || sideIndex >= 4) {
            errors.push_back(
                "canonical Poiseuille supports only xlo/xhi/ylo/yhi faces");
            continue;
        }
        if (faces[sideIndex] != nullptr) {
            errors.push_back(
                "canonical Poiseuille contains duplicate " +
                std::string(boundary::to_string(face.side)) + " boundary");
        } else {
            faces[sideIndex] = &face;
        }
    }
    for (int sideIndex = 0; sideIndex < 4; ++sideIndex) {
        if (faces[sideIndex] == nullptr) {
            errors.push_back(
                "canonical Poiseuille is missing " +
                std::string(boundary::to_string(
                    boundary::AllDomainSides[sideIndex])) +
                " boundary");
        }
    }

    boundary::BoundaryScheme commonScheme =
        boundary::BoundaryScheme::HalfwayBounceBack;
    bool commonSchemeSet = false;
    for (boundary::ResolvedDomainBoundary const* face : faces) {
        if (face == nullptr) {
            continue;
        }
        if (!is_supported_wall_scheme(face->scheme)) {
            errors.push_back(
                "canonical Poiseuille supports only halfway_bounce_back or "
                "guo_extrapolation domain schemes");
            continue;
        }
        if (!commonSchemeSet) {
            commonScheme = face->scheme;
            commonSchemeSet = true;
        } else if (face->scheme != commonScheme) {
            errors.push_back(
                "canonical Poiseuille requires one common x/y boundary geometry");
        }
    }
    if (commonSchemeSet) {
        const bool onLattice =
            commonScheme == boundary::BoundaryScheme::GuoExtrapolation;
        if (plan.axisGeometry[0].active != onLattice ||
            plan.axisGeometry[1].active != onLattice) {
            errors.push_back(
                "compiled Poiseuille plan geometry does not match the common "
                "x/y boundary scheme");
        }
    }

    for (int sideIndex : {2, 3}) {
        boundary::ResolvedDomainBoundary const* wall = faces[sideIndex];
        if (wall == nullptr) {
            continue;
        }
        if (wall->physical != boundary::BoundaryPhysical::NoSlipWall ||
            wall->profile != boundary::BoundaryProfileKind::None) {
            errors.push_back(
                "canonical Poiseuille ylo/yhi boundaries must be stationary "
                "no_slip_wall with profile=none");
        }
    }

    boundary::ResolvedDomainBoundary const* xlo = faces[0];
    boundary::ResolvedDomainBoundary const* xhi = faces[1];
    enum class DriveKind {
        Invalid,
        Velocity,
        Pressure,
        PrescribedState
    };
    DriveKind drive = DriveKind::Invalid;
    if (xlo != nullptr && xhi != nullptr) {
        if (xlo->physical == boundary::BoundaryPhysical::VelocityInlet &&
            xhi->physical == boundary::BoundaryPhysical::VelocityOutlet) {
            drive = DriveKind::Velocity;
        } else if (
            xlo->physical == boundary::BoundaryPhysical::PressureInlet &&
            xhi->physical == boundary::BoundaryPhysical::PressureOutlet) {
            drive = DriveKind::Pressure;
        } else if (
            xlo->physical == boundary::BoundaryPhysical::PrescribedState &&
            xhi->physical == boundary::BoundaryPhysical::PrescribedState) {
            drive = DriveKind::PrescribedState;
        } else {
            errors.push_back(
                "canonical Poiseuille x boundaries must be one matched "
                "velocity, pressure, or prescribed_state pair");
        }
    }

    auto validate_poiseuille_profile = [&](
        boundary::ResolvedDomainBoundary const& face,
        bool requireUniformPeak) {
        if (face.profile != boundary::BoundaryProfileKind::Poiseuille ||
            !face.hasProfileAxis || face.profileAxis != 1 ||
            !face.hasUMaxLb ||
            !reference_velocity_equal(face.uMaxLb, lbm.uRefLb)) {
            errors.push_back(
                "Poiseuille x boundary '" + face.name +
                "' must use profile=poiseuille, profile_axis=y, and "
                "u_max_lb=lbm.u_ref_lb");
        }
        if (requireUniformPeak &&
            !valid_uniform_peak(face, lbm.uRefLb)) {
            errors.push_back(
                "prescribed_state Poiseuille boundary '" + face.name +
                "' must use peak u_lb=(lbm.u_ref_lb,0,0)");
        }
    };

    if (drive == DriveKind::Velocity) {
        validate_poiseuille_profile(*xlo, false);
        validate_poiseuille_profile(*xhi, false);
        if (initial.linearDensity != 0) {
            errors.push_back(
                "velocity-driven canonical Poiseuille initialization must "
                "use constant density");
        }
    }

    amrex::Real rhoIn = 0.0;
    amrex::Real rhoOut = 0.0;
    if (drive == DriveKind::Pressure ||
        drive == DriveKind::PrescribedState) {
        if (commonScheme != boundary::BoundaryScheme::GuoExtrapolation) {
            errors.push_back(
                "density-driven canonical Poiseuille requires "
                "guo_extrapolation on all x/y boundaries");
        }
        if (drive == DriveKind::Pressure) {
            if (xlo->profile != boundary::BoundaryProfileKind::Constant ||
                xhi->profile != boundary::BoundaryProfileKind::Constant ||
                !xlo->hasRhoLb || !xhi->hasRhoLb) {
                errors.push_back(
                    "pressure-driven Poiseuille requires constant rho_lb on "
                    "xlo pressure_inlet and xhi pressure_outlet");
            }
        } else {
            validate_poiseuille_profile(*xlo, true);
            validate_poiseuille_profile(*xhi, true);
            if (!xlo->hasRhoLb || !xhi->hasRhoLb) {
                errors.push_back(
                    "prescribed-state Poiseuille requires rho_lb on both x faces");
            }
        }

        if (xlo->hasRhoLb && xhi->hasRhoLb) {
            rhoIn = xlo->rhoLb;
            rhoOut = xhi->rhoLb;
        }
        if (initial.linearDensity != 1) {
            errors.push_back(
                "density-driven canonical Poiseuille initialization requires "
                "explicit streamwise endpoint densities");
        } else if (
            !nearly_equal(initial.rhoXLoLb, rhoIn) ||
            !nearly_equal(initial.rhoXHiLb, rhoOut)) {
            errors.push_back(
                "Poiseuille initialization endpoint densities must equal the "
                "resolved xlo/xhi boundary densities");
        }

        const amrex::Real densityDelta = rhoIn - rhoOut;
        const amrex::Real rhoMidpoint = 0.5 * (rhoIn + rhoOut);
        if (!(rhoIn > rhoOut)) {
            errors.push_back(
                "canonical Poiseuille requires xlo rho_lb > xhi rho_lb");
        }
        if (!nearly_equal(rhoMidpoint, lbm.rhoRefLb)) {
            errors.push_back(
                "density-driven Poiseuille boundary values must be symmetric "
                "around lbm.rho_ref_lb");
        }
        if (diagnosticGeometry.streamwiseOnLatticeBoundaryNodes == 0 ||
            diagnosticGeometry.onLatticeBoundaryNodes == 0) {
            errors.push_back(
                "density-driven canonical Poiseuille requires on-lattice "
                "Guo nodes on both x and y axes");
        }

        int pressureScalingCode = 0;
        if (pressureScaling == "fixed_mach_characterization") {
            pressureScalingCode = 1;
        } else if (pressureScaling == "fixed_re_fixed_tau") {
            pressureScalingCode = 2;
        } else {
            errors.push_back(
                "poiseuille_flow.pressure_scaling='" +
                pressureScaling +
                "' is unsupported; expected fixed_mach_characterization or "
                "fixed_re_fixed_tau");
        }

        const amrex::Real lengthLb = static_cast<amrex::Real>(
            diagnosticGeometry.streamwiseLatticeIntervals);
        const amrex::Real reEffective =
            lbm.uRefLb * heightLb / lbm.nuLb;
        if (!nearly_equal(lbm.reynolds, reEffective)) {
            std::ostringstream os;
            os << "density-driven Poiseuille requires lbm.re="
               << reEffective << " from U_max_lb=" << lbm.uRefLb
               << ", H_lb(intervals)=" << heightLb
               << ", nu_lb=" << lbm.nuLb
               << "; resolved lbm.re=" << lbm.reynolds;
            errors.push_back(os.str());
        }
        const amrex::Real expectedDensityDelta =
            poiseuille_density_delta(
                lengthLb,
                heightLb,
                lbm.rhoRefLb,
                lbm.uRefLb,
                lbm.nuLb,
                Descriptor::cs2());
        if (!nearly_equal(densityDelta, expectedDensityDelta)) {
            std::ostringstream os;
            os << "rho_in-rho_out=" << densityDelta
               << " but interval-based Poiseuille target is "
               << expectedDensityDelta << " from Lx_lb=" << lengthLb
               << ", H_lb=" << heightLb
               << ", nu_lb=" << lbm.nuLb
               << ", U_max_lb=" << lbm.uRefLb
               << "; valid-node counts must not replace lattice intervals";
            errors.push_back(os.str());
        }

        PoiseuillePressureDiagnosticConfig& pressure =
            result.data.pressureDiagnostic;
        pressure.enabled = 1;
        pressure.pressureDriven =
            drive == DriveKind::Pressure ? 1 : 0;
        pressure.prescribedState =
            drive == DriveKind::PrescribedState ? 1 : 0;
        pressure.scalingCode = pressureScalingCode;
        pressure.rhoInLb = rhoIn;
        pressure.rhoOutLb = rhoOut;
        pressure.densityDeltaTargetLb = densityDelta;
        pressure.densityDeltaExpectedLb = expectedDensityDelta;
        pressure.densityDeltaOverRhoRef =
            densityDelta / lbm.rhoRefLb;
        pressure.mach =
            lbm.uRefLb / std::sqrt(Descriptor::cs2());
        pressure.reEffective = reEffective;
        pressure.streamwiseLengthLb = lengthLb;
        pressure.channelHeightLb = heightLb;
    }

    return result;
}

} // namespace lbm::cases::poiseuille_flow

#endif
