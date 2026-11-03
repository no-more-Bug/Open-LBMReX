#ifndef LBM_CASES_CHANNELCYLINDER_CASEVALIDATION_H
#define LBM_CASES_CHANNELCYLINDER_CASEVALIDATION_H

#include "channel_cylinder/config/ObservationConfig.H"
#include "lbm/boundary/core/ResolvedBoundary.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/config/LbmConfig.H"

#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace lbm::cases::channel_cylinder {

namespace channel_cylinder_benchmark_detail {

inline bool nearly_equal(amrex::Real lhs, amrex::Real rhs) noexcept
{
    const amrex::Real scale = std::max(
        amrex::Real(1.0), std::max(std::abs(lhs), std::abs(rhs)));
    return std::abs(lhs - rhs) <=
        amrex::Real(512.0) * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

inline int aligned_cell_index(
    amrex::Geometry const& geometry,
    int axis,
    amrex::Real coordinate,
    std::string const& label,
    std::vector<std::string>& errors)
{
    const amrex::Real position =
        (coordinate - geometry.ProbLo(axis)) / geometry.CellSize(axis) -
        amrex::Real(0.5);
    const int index = static_cast<int>(std::llround(position));
    if (!nearly_equal(position, static_cast<amrex::Real>(index)) ||
        index < geometry.Domain().smallEnd(axis) ||
        index > geometry.Domain().bigEnd(axis)) {
        errors.push_back(
            label + " must coincide with one valid cell center");
        return geometry.Domain().smallEnd(axis);
    }
    return index;
}

inline std::array<boundary::ResolvedDomainBoundary const*, 4>
collect_xy_faces(
    boundary::ResolvedBoundarySet const& resolved,
    std::vector<std::string>& errors)
{
    std::array<boundary::ResolvedDomainBoundary const*, 4> faces{};
    for (boundary::ResolvedDomainBoundary const& face : resolved.domain) {
        const int side = boundary::domain_side_index(face.side);
        if (side < 0 || side >= 4) {
            errors.push_back(
                "channel-cylinder must not define z-face boundaries");
            continue;
        }
        if (faces[static_cast<std::size_t>(side)] != nullptr) {
            errors.push_back(
                "duplicate " + std::string(boundary::to_string(face.side)) +
                " domain boundary");
            continue;
        }
        faces[static_cast<std::size_t>(side)] = &face;
    }
    for (int side = 0; side < 4; ++side) {
        if (faces[static_cast<std::size_t>(side)] == nullptr) {
            errors.push_back(
                "missing " +
                std::string(boundary::to_string(boundary::AllDomainSides[side])) +
                " domain boundary");
        }
    }
    return faces;
}

inline bool is_guo_face(
    boundary::ResolvedDomainBoundary const& face) noexcept
{
    return face.scheme == boundary::BoundaryScheme::GuoExtrapolation &&
           face.stage == boundary::BoundaryExecutionStage::PostStreamingValidNode;
}

} // namespace channel_cylinder_benchmark_detail

/**
 * @brief Validate the selected channel-cylinder setup.
 *
 * The validator reads already resolved host values once. It does not derive or
 * copy a Reynolds number and does not mutate LBM or boundary state.
 */
template <class Descriptor>
ChannelCylinderBenchmarkValidationResult validate_channel_cylinder_benchmark(
    config::ResolvedLbmConfig const& lbm,
    amrex::Geometry const& geometry,
    boundary::ResolvedBoundarySet const& resolved,
    boundary::PostStreamingBoundaryPlan const& plan,
    std::string const& reference,
    int forceInterval,
    int outputInterval,
    int writeHistory,
    int assertEnabled)
{
    using namespace channel_cylinder_benchmark_detail;
    ChannelCylinderBenchmarkValidationResult result;
    ChannelCylinderRuntimeConfig& data = result.data;
    data.enabled = true;
    data.referenceLabel = reference;
    data.forceInterval = forceInterval;
    data.outputInterval = outputInterval;
    data.writeHistory = writeHistory;
    data.assertEnabled = assertEnabled;

    const bool tandem = reference == "tandem_two_cylinder";
    if (reference != "dfg_2d_1_re20" &&
        reference != "dfg_2d_2_re100" && !tandem) {
        result.errors.push_back(
            "Stage 4.7 supports reference='dfg_2d_1_re20' or "
            "'dfg_2d_2_re100', or integration label="
            "'tandem_two_cylinder' only");
    }
    if (forceInterval <= 0) {
        result.errors.push_back("force_interval must be positive");
    }
    if (outputInterval < forceInterval) {
        result.errors.push_back(
            "output_interval must be greater than or equal to force_interval");
    }
    if ((writeHistory != 0 && writeHistory != 1) ||
        (assertEnabled != 0 && assertEnabled != 1)) {
        result.errors.push_back("write_history and assert must each be 0 or 1");
    }
    if (reference == "dfg_2d_2_re100") {
        if (writeHistory != 1) {
            result.errors.push_back(
                "dfg_2d_2_re100 requires write_history=1 for offline analysis");
        }
        if (assertEnabled != 0) {
            result.errors.push_back(
                "dfg_2d_2_re100 requires assert=0; its periodic gate is offline");
        }
    }
    if (tandem && (writeHistory != 1 || assertEnabled != 0)) {
        result.errors.push_back(
            "tandem_two_cylinder requires write_history=1 and assert=0");
    }

    auto const faces = collect_xy_faces(resolved, result.errors);
    if (faces[0] != nullptr) {
        auto const& inlet = *faces[0];
        if (!is_guo_face(inlet) ||
            inlet.physical != boundary::BoundaryPhysical::VelocityInlet ||
            inlet.profile != boundary::BoundaryProfileKind::Poiseuille ||
            !inlet.hasProfileAxis || inlet.profileAxis != 1 ||
            !inlet.hasUMaxLb || !(inlet.uMaxLb > 0.0)) {
            result.errors.push_back(
                "xlo must be Guo velocity_inlet with a y-axis Poiseuille profile");
        } else {
            data.uMeanLb = amrex::Real(2.0 / 3.0) * inlet.uMaxLb;
            if (!nearly_equal(data.uMeanLb, lbm.uRefLb)) {
                result.errors.push_back(
                    "inlet u_mean_lb must equal resolved lbm.u_ref_lb");
            }
        }
    }
    if (faces[1] != nullptr) {
        auto const& outlet = *faces[1];
        if (!is_guo_face(outlet) ||
            outlet.physical != boundary::BoundaryPhysical::PressureOutlet ||
            outlet.profile != boundary::BoundaryProfileKind::Constant ||
            !outlet.hasRhoLb || !nearly_equal(outlet.rhoLb, lbm.rhoRefLb)) {
            result.errors.push_back(
                "xhi must be Guo pressure_outlet with profile=constant and "
                "rho_lb=lbm.rho_ref_lb");
        }
    }
    for (int side : {2, 3}) {
        if (faces[static_cast<std::size_t>(side)] == nullptr) {
            continue;
        }
        auto const& wall = *faces[static_cast<std::size_t>(side)];
        if (!is_guo_face(wall) ||
            wall.physical != boundary::BoundaryPhysical::NoSlipWall ||
            wall.profile != boundary::BoundaryProfileKind::None) {
            result.errors.push_back(
                std::string(boundary::to_string(wall.side)) +
                " must be a Guo static no-slip wall");
        }
    }

    const bool objectCountOk = tandem ?
        (resolved.internalObjects.size() == 1 ||
         resolved.internalObjects.size() == 2) :
        resolved.internalObjects.size() == 1;
    if (resolved.internal.size() != (tandem ? 2U : 1U) || !objectCountOk) {
        result.errors.push_back(tandem ?
            "tandem_two_cylinder requires two internal boundaries grouped as one or two objects" :
            "DFG channel-cylinder requires exactly one internal boundary and one object");
    } else {
        for (auto const& cylinder : resolved.internal) {
            if (cylinder.geometry != "cylinder" || cylinder.axis != 2 ||
                cylinder.physical != boundary::BoundaryPhysical::NoSlipWall ||
                cylinder.profile != boundary::BoundaryProfileKind::None ||
                (tandem ? cylinder.scheme != boundary::BoundaryScheme::Yms :
                 (cylinder.scheme != boundary::BoundaryScheme::Yms &&
                  cylinder.scheme != boundary::BoundaryScheme::Bouzidi))) {
                result.errors.push_back(tandem ?
                    "tandem objects must be static analytic z-cylinders using YMS" :
                    "internal object must be one static analytic z-cylinder using YMS or Bouzidi");
            }
        }

        auto const& front = resolved.internal.front();
        if (!nearly_equal(front.centerPhys[0], amrex::Real(0.2)) ||
            !nearly_equal(front.centerPhys[1], amrex::Real(0.2)) ||
            !nearly_equal(front.radiusPhys, amrex::Real(0.05))) {
            result.errors.push_back(
                "front cylinder requires center=(0.2,0.2) and radius=0.05");
        }
        if (tandem) {
            auto const& rear = resolved.internal.back();
            if (!nearly_equal(rear.centerPhys[0], amrex::Real(0.4)) ||
                !nearly_equal(rear.centerPhys[1], amrex::Real(0.2)) ||
                !nearly_equal(rear.radiusPhys, amrex::Real(0.05))) {
                result.errors.push_back(
                    "rear cylinder requires center=(0.4,0.2) and radius=0.05");
            }
        }

        for (std::size_t index = 0;
             index < resolved.internalObjects.size(); ++index) {
            auto const& object = resolved.internalObjects[index];
            if (object.objectId.value != static_cast<int>(index) ||
                object.name.empty() ||
                object.name.find_first_of(",\r\n") != std::string::npos) {
                result.errors.push_back(
                    "internal object ids must be contiguous and names must be CSV-safe");
            }
            data.observedObjects.push_back({object.name, object.objectId.value});
        }

        const bool singleMapping = !tandem &&
            resolved.internalObjects.front().boundaryCount == 1 &&
            front.objectId.value == 0 &&
            front.objectName == resolved.internalObjects.front().name;
        const bool sharedMapping = tandem &&
            resolved.internalObjects.size() == 1 &&
            resolved.internalObjects.front().boundaryCount == 2 &&
            resolved.internalObjects.front().name == "cylinder_pair" &&
            front.objectId.value == 0 && resolved.internal.back().objectId.value == 0 &&
            front.objectName == "cylinder_pair" &&
            resolved.internal.back().objectName == "cylinder_pair";
        const bool distinctMapping = tandem &&
            resolved.internalObjects.size() == 2 &&
            resolved.internalObjects[0].boundaryCount == 1 &&
            resolved.internalObjects[1].boundaryCount == 1 &&
            resolved.internalObjects[0].name == "front_cylinder" &&
            resolved.internalObjects[1].name == "rear_cylinder" &&
            front.objectId.value == 0 && resolved.internal.back().objectId.value == 1 &&
            front.objectName == "front_cylinder" &&
            resolved.internal.back().objectName == "rear_cylinder";
        if (!(singleMapping || sharedMapping || distinctMapping)) {
            result.errors.push_back(
                "internal cylinder ObjectId/name mapping is inconsistent with the selected case");
        }
        data.diameterLb = amrex::Real(2.0) * front.radiusPhys /
            geometry.CellSize(0);
        const int diameterCells =
            static_cast<int>(std::llround(data.diameterLb));
        if (!nearly_equal(
                data.diameterLb,
                static_cast<amrex::Real>(diameterCells)) ||
            diameterCells <= 0) {
            result.errors.push_back(
                "cylinder diameter must contain an integer number of lattice intervals");
        }
        if (tandem && diameterCells != 20) {
            result.errors.push_back(
                "tandem_two_cylinder requires N_D=20");
        }
    }

    if (tandem) {
        if constexpr (Descriptor::d != 2) {
            result.errors.push_back(
                "tandem_two_cylinder is a D2Q9 integration case");
        }
        if (!nearly_equal(lbm.reynolds, amrex::Real(82.0)) ||
            !nearly_equal(lbm.rhoRefLb, amrex::Real(1.0)) ||
            !nearly_equal(data.uMeanLb, amrex::Real(0.035))) {
            result.errors.push_back(
                "tandem_two_cylinder requires Re_H=82, rho0=1, and u_mean_lb=0.035");
        }
    }

    if (!plan.axisGeometry[0].active || !plan.axisGeometry[1].active ||
        !nearly_equal(plan.axisGeometry[0].physicalLo, amrex::Real(0.0)) ||
        !nearly_equal(plan.axisGeometry[0].physicalHi, amrex::Real(2.2)) ||
        !nearly_equal(plan.axisGeometry[1].physicalLo, amrex::Real(0.0)) ||
        !nearly_equal(plan.axisGeometry[1].physicalHi, amrex::Real(0.41)) ||
        !nearly_equal(geometry.CellSize(0), geometry.CellSize(1))) {
        result.errors.push_back(
            "DFG channel requires on-lattice [0,2.2]x[0,0.41] geometry with dx=dy");
    }
    if (!nearly_equal(lbm.lengthRefPhys, amrex::Real(0.41)) ||
        !nearly_equal(
            lbm.lengthRefLb,
            static_cast<amrex::Real>(plan.axisGeometry[1].latticeIntervals))) {
        result.errors.push_back(
            "lbm.length_ref_phys/lb must represent the channel height");
    }

    if constexpr (Descriptor::d == 2) {
        if (geometry.Domain().length(2) != 1) {
            result.errors.push_back(
                "D2 channel-cylinder force requires one z cell for unit-span normalization");
        }
        data.spanLb = 1.0;
        data.pressureSamplesPerPoint = 1;
    } else {
        if (!geometry.isPeriodic(2) || geometry.Domain().length(2) < 2 ||
            !nearly_equal(geometry.CellSize(2), geometry.CellSize(0))) {
            result.errors.push_back(
                "D3 channel-cylinder requires periodic z, at least two layers, and dz=dx");
        }
        data.spanLb = static_cast<amrex::Real>(geometry.Domain().length(2));
        data.pressureSamplesPerPoint = geometry.Domain().length(2);
    }

    const int frontI = aligned_cell_index(
        geometry, 0, amrex::Real(0.15), "front pressure probe x=0.15", result.errors);
    const int rearI = aligned_cell_index(
        geometry, 0, amrex::Real(0.25), "rear pressure probe x=0.25", result.errors);
    const int probeJ = aligned_cell_index(
        geometry, 1, amrex::Real(0.2), "pressure probe y=0.2", result.errors);
    const int probeK = geometry.Domain().smallEnd(2);
    data.pressureFrontCell = amrex::IntVect(AMREX_D_DECL(frontI, probeJ, probeK));
    data.pressureRearCell = amrex::IntVect(AMREX_D_DECL(rearI, probeJ, probeK));
    return result;
}

} // namespace lbm::cases::channel_cylinder

#endif
