#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYPROGRAM_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYPROGRAM_H

#include "lid_driven_cavity/composition/Preparation.H"
#include "lid_driven_cavity/config/Controls.H"
#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

using RestProfile = initialization::ConstantInitializationProfile;
using RestInitialization =
    initialization::ProfileInitializationSelection<RestProfile>;

template <class Descriptor, class DynamicsSelection>
using CavityProgram = runtime::SingleLevelProgram<
    Descriptor,
    DynamicsSelection,
    RestInitialization,
    Preparation>;

namespace cavity_program_detail {

inline boundary::BoundaryScheme boundary_scheme(WallScheme scheme) noexcept
{
    return scheme == WallScheme::HalfwayBounceBack ?
        boundary::BoundaryScheme::HalfwayBounceBack :
        boundary::BoundaryScheme::GuoExtrapolation;
}

inline boundary::DomainBoundarySpec stationary_wall(
    char const* hbbName,
    char const* guoName,
    boundary::DomainSide side,
    WallScheme scheme)
{
    boundary::DomainBoundarySpec spec;
    spec.name = scheme == WallScheme::GuoExtrapolation ? guoName : hbbName;
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = boundary_scheme(scheme);
    spec.profile = boundary::BoundaryProfileKind::None;
    return spec;
}

inline boundary::DomainBoundarySpec moving_lid(
    amrex::Real lidVelocityLb,
    WallScheme scheme)
{
    boundary::DomainBoundarySpec spec;
    spec.name = scheme == WallScheme::GuoExtrapolation ?
        "guo_moving_lid" : "moving_lid";
    spec.location = boundary::DomainSide::YHi;
    spec.physical = boundary::BoundaryPhysical::MovingWall;
    spec.scheme = boundary_scheme(scheme);
    spec.profile = boundary::BoundaryProfileKind::Uniform;
    spec.prescribedULb = {lidVelocityLb, 0.0, 0.0};
    spec.hasPrescribedULb = true;
    return spec;
}

} // namespace cavity_program_detail

/**
 * Form a complete lid-driven-cavity Program from stable case controls.
 *
 * Descriptor and local Dynamics remain compile-time choices at the caller.
 * Geometry envelope, reference closure, rest initialization and physical
 * boundary records remain one coherent case implementation.
 */
template <class Descriptor, class DynamicsSelection>
CavityProgram<Descriptor, DynamicsSelection> make_cavity_program(
    Controls controls,
    DynamicsSelection dynamics,
    RestInitialization initialization)
{
    CavityProgram<Descriptor, DynamicsSelection> program;
    const amrex::Real lengthX =
        controls.physicalHi[0] - controls.physicalLo[0];
    const amrex::Real dx =
        lengthX / static_cast<amrex::Real>(controls.intervals[0]);
    const bool onLattice =
        controls.wallScheme == WallScheme::GuoExtrapolation;

    program.mesh.nCell = {
        controls.intervals[0] + (onLattice ? 1 : 0),
        controls.intervals[1] + (onLattice ? 1 : 0),
        controls.depthCells};
    program.mesh.envelopeLo = {
        controls.physicalLo[0] - (onLattice ? 0.5 * dx : 0.0),
        controls.physicalLo[1] - (onLattice ? 0.5 * dx : 0.0),
        0.0};
    program.mesh.envelopeHi = {
        controls.physicalHi[0] + (onLattice ? 0.5 * dx : 0.0),
        controls.physicalHi[1] + (onLattice ? 0.5 * dx : 0.0),
        dx * static_cast<amrex::Real>(program.mesh.nCell[2])};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;

    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.lidVelocityLb;
    program.lbm.uRefPhys = controls.velocityReferencePhys;
    program.lbm.lengthRefLb =
        static_cast<amrex::Real>(controls.intervals[0]);
    program.lbm.lengthRefPhys = lengthX;
    program.lbm.reynolds = controls.reynolds;
    program.lbm.streamingBackend = std::string("baseline");
    program.dynamics = std::move(dynamics);

    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "lid_driven_cavity";
    program.run.variant = controls.runVariant;

    program.initialization = std::move(initialization);
    program.domainBoundaries = {
        cavity_program_detail::stationary_wall(
            "left_wall",
            "guo_left_no_slip_wall",
            boundary::DomainSide::XLo,
            controls.wallScheme),
        cavity_program_detail::stationary_wall(
            "right_wall",
            "guo_right_no_slip_wall",
            boundary::DomainSide::XHi,
            controls.wallScheme),
        cavity_program_detail::stationary_wall(
            "bottom_wall",
            "guo_bottom_no_slip_wall",
            boundary::DomainSide::YLo,
            controls.wallScheme),
        cavity_program_detail::moving_lid(
            controls.lidVelocityLb,
            controls.wallScheme)};
    program.casePreparation = Preparation{std::move(controls)};
    return program;
}

} // namespace lbm::cases::lid_driven_cavity

#endif
