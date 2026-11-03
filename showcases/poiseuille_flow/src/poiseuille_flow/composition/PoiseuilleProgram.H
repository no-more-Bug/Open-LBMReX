#ifndef LBM_CASES_POISEUILLEFLOW_PROGRAM_H
#define LBM_CASES_POISEUILLEFLOW_PROGRAM_H

#include "poiseuille_flow/composition/Preparation.H"
#include "poiseuille_flow/config/Controls.H"
#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "runtime/SingleLevelProgram.H"

#include <string>
#include <utility>

namespace lbm::cases::poiseuille_flow {

using Profile = initialization::PoiseuilleInitializationProfile;
using Initialization = initialization::ProfileInitializationSelection<Profile>;

template <class Descriptor, class DynamicsSelection>
using PoiseuilleProgram = runtime::SingleLevelProgram<
    Descriptor,
    DynamicsSelection,
    Initialization,
    Preparation>;

namespace program_detail {

inline boundary::DomainBoundarySpec x_face(
    Controls const& controls,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    const bool inlet = side == boundary::DomainSide::XLo;
    spec.location = side;
    spec.scheme = uses_guo(controls.variant) ?
        boundary::BoundaryScheme::GuoExtrapolation :
        boundary::BoundaryScheme::HalfwayBounceBack;

    switch (controls.variant) {
    case Variant::HbbVelocity:
    case Variant::GuoVelocity:
        spec.name = uses_guo(controls.variant) ?
            (inlet ? "guo_poiseuille_velocity_inlet" :
                     "guo_poiseuille_velocity_outlet") :
            (inlet ? "poiseuille_velocity_inlet" :
                     "poiseuille_velocity_outlet");
        spec.physical = inlet ?
            boundary::BoundaryPhysical::VelocityInlet :
            boundary::BoundaryPhysical::VelocityOutlet;
        spec.profile = boundary::BoundaryProfileKind::Poiseuille;
        spec.profileAxis = 1;
        spec.hasProfileAxis = true;
        spec.uMaxLb = controls.uMaxLb;
        spec.hasUMaxLb = true;
        break;
    case Variant::GuoPressure:
        spec.name = inlet ? "guo_pressure_inlet" : "guo_pressure_outlet";
        spec.physical = inlet ?
            boundary::BoundaryPhysical::PressureInlet :
            boundary::BoundaryPhysical::PressureOutlet;
        spec.profile = boundary::BoundaryProfileKind::Constant;
        spec.rhoLb = inlet ? controls.rhoInLb : controls.rhoOutLb;
        spec.hasRhoLb = true;
        break;
    case Variant::GuoPrescribedState:
        spec.name = inlet ? "guo_state_xlo" : "guo_state_xhi";
        spec.physical = boundary::BoundaryPhysical::PrescribedState;
        spec.profile = boundary::BoundaryProfileKind::Poiseuille;
        spec.profileAxis = 1;
        spec.hasProfileAxis = true;
        spec.rhoLb = inlet ? controls.rhoInLb : controls.rhoOutLb;
        spec.hasRhoLb = true;
        spec.prescribedULb = {controls.uMaxLb, 0.0, 0.0};
        spec.hasPrescribedULb = true;
        break;
    }
    return spec;
}

inline boundary::DomainBoundarySpec wall(
    Controls const& controls,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    spec.name = side == boundary::DomainSide::YLo ?
        (uses_guo(controls.variant) ? "guo_bottom_no_slip_wall" :
                                     "bottom_no_slip_wall") :
        (uses_guo(controls.variant) ? "guo_top_no_slip_wall" :
                                     "top_no_slip_wall");
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = uses_guo(controls.variant) ?
        boundary::BoundaryScheme::GuoExtrapolation :
        boundary::BoundaryScheme::HalfwayBounceBack;
    return spec;
}

} // namespace program_detail

template <class Descriptor, class DynamicsSelection>
PoiseuilleProgram<Descriptor, DynamicsSelection> make_poiseuille_program(
    Controls controls,
    DynamicsSelection dynamics)
{
    PoiseuilleProgram<Descriptor, DynamicsSelection> program;
    const bool guo = uses_guo(controls.variant);
    const amrex::Real dx =
        1.0 / static_cast<amrex::Real>(controls.intervals[1]);
    program.mesh.nCell = {
        controls.intervals[0] + (guo ? 1 : 0),
        controls.intervals[1] + (guo ? 1 : 0),
        controls.depthCells};
    program.mesh.envelopeLo = {
        guo ? -0.5 * dx : 0.0,
        guo ? -0.5 * dx : 0.0,
        0.0};
    program.mesh.envelopeHi = {
        guo ? 4.0 + 0.5 * dx : 4.0,
        guo ? 1.0 + 0.5 * dx : 1.0,
        dx * static_cast<amrex::Real>(controls.depthCells)};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;

    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromTau;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.uMaxLb;
    program.lbm.uRefPhys = controls.uRefPhys;
    program.lbm.lengthRefLb =
        static_cast<amrex::Real>(controls.intervals[1]);
    program.lbm.lengthRefPhys = 1.0;
    program.lbm.tau = controls.tau;
    program.lbm.streamingBackend = std::string("baseline");
    program.dynamics = std::move(dynamics);

    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "poiseuille_flow";
    program.run.variant = controls.runVariant;

    initialization::PoiseuilleInitializationParams params;
    params.rhoLb = controls.rhoLb;
    params.uMaxLb = controls.uMaxLb;
    params.linearDensity = density_driven(controls.variant) ? 1 : 0;
    params.rhoXLoLb = density_driven(controls.variant) ?
        controls.rhoInLb : controls.rhoLb;
    params.rhoXHiLb = density_driven(controls.variant) ?
        controls.rhoOutLb : controls.rhoLb;
    program.initialization = initialization::from_profile(Profile{params});
    program.domainBoundaries = {
        program_detail::x_face(controls, boundary::DomainSide::XLo),
        program_detail::x_face(controls, boundary::DomainSide::XHi),
        program_detail::wall(controls, boundary::DomainSide::YLo),
        program_detail::wall(controls, boundary::DomainSide::YHi)};
    program.casePreparation = Preparation{std::move(controls)};
    return program;
}

} // namespace lbm::cases::poiseuille_flow

#endif
