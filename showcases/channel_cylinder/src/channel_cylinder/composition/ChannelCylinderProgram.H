#ifndef LBM_CASES_CHANNELCYLINDER_PROGRAM_H
#define LBM_CASES_CHANNELCYLINDER_PROGRAM_H

#include "channel_cylinder/composition/Preparation.H"
#include "channel_cylinder/config/Controls.H"
#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <string>
#include <utility>

namespace lbm::cases::channel_cylinder {

using InitializationProfile =
    initialization::ConstantInitializationProfile;
using Initialization =
    initialization::ProfileInitializationSelection<InitializationProfile>;

template <class Descriptor, class DynamicsSelection>
using ChannelCylinderProgram = runtime::SingleLevelProgram<
    Descriptor,
    DynamicsSelection,
    Initialization,
    Preparation>;

namespace program_detail {

inline boundary::DomainBoundarySpec inlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "inlet";
    spec.location = boundary::DomainSide::XLo;
    spec.physical = boundary::BoundaryPhysical::VelocityInlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Poiseuille;
    spec.profileAxis = 1;
    spec.hasProfileAxis = true;
    spec.uMeanLb = controls.uMeanLb;
    spec.hasUMeanLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec outlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "outlet";
    spec.location = boundary::DomainSide::XHi;
    spec.physical = boundary::BoundaryPhysical::PressureOutlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Constant;
    spec.rhoLb = controls.rhoLb;
    spec.hasRhoLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec wall(boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    spec.name = side == boundary::DomainSide::YLo ?
        "bottom_wall" : "top_wall";
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::None;
    return spec;
}

inline boundary::InternalBoundarySpec cylinder(
    Controls const& controls,
    amrex::Real dx,
    bool rear)
{
    const bool tandem = is_tandem(controls.variant);
    boundary::InternalBoundarySpec spec;
    spec.name = tandem ?
        (rear ? "rear_cylinder_wall" : "front_cylinder_wall") :
        "cylinder_wall";
    spec.geometry = "cylinder";
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.hasPhysical = true;
    spec.scheme = uses_bouzidi(controls.variant) ?
        boundary::BoundaryScheme::Bouzidi :
        boundary::BoundaryScheme::Yms;
    spec.profile = boundary::BoundaryProfileKind::None;
    spec.objectName = tandem ?
        (shares_tandem_object(controls.variant) ?
            "cylinder_pair" :
            (rear ? "rear_cylinder" : "front_cylinder")) :
        "cylinder";
    spec.centerPhys = {
        rear ? amrex::Real(0.4) : amrex::Real(0.2),
        amrex::Real(0.2),
        amrex::Real(0.5) * dx *
            static_cast<amrex::Real>(controls.depthCells)};
    spec.hasCenterPhys = true;
    spec.radiusPhys = 0.05;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

} // namespace program_detail

template <class Descriptor, class DynamicsSelection>
ChannelCylinderProgram<Descriptor, DynamicsSelection>
make_channel_cylinder_program(
    Controls controls,
    DynamicsSelection dynamics)
{
    ChannelCylinderProgram<Descriptor, DynamicsSelection> program;
    const amrex::Real dx =
        amrex::Real(0.1) /
        static_cast<amrex::Real>(controls.diameterIntervals);
    const int xIntervals = 22 * controls.diameterIntervals;
    const int yIntervals = 41 * controls.diameterIntervals / 10;

    program.mesh.nCell = {
        xIntervals + 1,
        yIntervals + 1,
        controls.depthCells};
    program.mesh.envelopeLo = {-0.5 * dx, -0.5 * dx, 0.0};
    program.mesh.envelopeHi = {
        amrex::Real(2.2) + 0.5 * dx,
        amrex::Real(0.41) + 0.5 * dx,
        dx * static_cast<amrex::Real>(controls.depthCells)};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;

    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.uMeanLb;
    program.lbm.uRefPhys = controls.uMeanPhys;
    program.lbm.lengthRefLb = static_cast<amrex::Real>(yIntervals);
    program.lbm.lengthRefPhys = 0.41;
    program.lbm.reynolds = controls.reynoldsHeight;
    program.lbm.streamingBackend = std::string("baseline");
    program.dynamics = std::move(dynamics);

    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "channel_cylinder";
    program.run.variant = controls.runVariant;

    const bool tandem = is_tandem(controls.variant);
    program.initialization = initialization::from_profile(
        InitializationProfile{{
            controls.rhoLb,
            uses_uniform_initialization(controls.variant) ?
                controls.uMeanLb :
                amrex::Real(0.0),
            0.0,
            0.0}});
    program.domainBoundaries = {
        program_detail::inlet(controls),
        program_detail::outlet(controls),
        program_detail::wall(boundary::DomainSide::YLo),
        program_detail::wall(boundary::DomainSide::YHi)};
    program.internalBoundaries = {
        program_detail::cylinder(controls, dx, false)};
    if (tandem) {
        program.internalBoundaries.push_back(
            program_detail::cylinder(controls, dx, true));
    }
    program.casePreparation = Preparation{std::move(controls)};
    return program;
}

} // namespace lbm::cases::channel_cylinder

#endif
