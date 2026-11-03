#ifndef LBM_CASES_UNIFORMFLOW_PROGRAM_H
#define LBM_CASES_UNIFORMFLOW_PROGRAM_H

#include "uniform_flow/composition/Preparation.H"
#include "uniform_flow/config/Controls.H"
#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <string>
#include <utility>

namespace lbm::cases::uniform_flow {

using UniformProfile = initialization::ConstantInitializationProfile;
using UniformInitialization =
    initialization::ProfileInitializationSelection<UniformProfile>;

template <class Descriptor, class DynamicsSelection>
using UniformFlowProgram = runtime::SingleLevelProgram<
    Descriptor,
    DynamicsSelection,
    UniformInitialization,
    Preparation>;

namespace program_detail {

inline bool uses_guo(Variant variant) noexcept
{
    return variant != Variant::HbbVelocityIo;
}

inline boundary::DomainBoundarySpec inlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = uses_guo(controls.variant) ?
        "guo_uniform_velocity_inlet" : "uniform_velocity_inlet";
    spec.location = boundary::DomainSide::XLo;
    spec.physical = boundary::BoundaryPhysical::VelocityInlet;
    spec.scheme = uses_guo(controls.variant) ?
        boundary::BoundaryScheme::GuoExtrapolation :
        boundary::BoundaryScheme::HalfwayBounceBack;
    spec.profile = boundary::BoundaryProfileKind::Uniform;
    spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
    spec.hasPrescribedULb = true;
    return spec;
}

inline boundary::DomainBoundarySpec outlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name =
        controls.variant == Variant::GuoVelocityFreeOutflow ?
            "guo_free_outflow" :
            (uses_guo(controls.variant) ?
                "guo_uniform_velocity_outlet" : "uniform_velocity_outlet");
    spec.location = boundary::DomainSide::XHi;
    spec.physical =
        controls.variant == Variant::GuoVelocityFreeOutflow ?
            boundary::BoundaryPhysical::FreeOutflow :
            boundary::BoundaryPhysical::VelocityOutlet;
    spec.scheme = uses_guo(controls.variant) ?
        boundary::BoundaryScheme::GuoExtrapolation :
        boundary::BoundaryScheme::HalfwayBounceBack;
    if (spec.physical == boundary::BoundaryPhysical::VelocityOutlet) {
        spec.profile = boundary::BoundaryProfileKind::Uniform;
        spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
        spec.hasPrescribedULb = true;
    }
    return spec;
}

} // namespace program_detail

template <class Descriptor, class DynamicsSelection>
UniformFlowProgram<Descriptor, DynamicsSelection> make_uniform_flow_program(
    Controls controls,
    DynamicsSelection dynamics)
{
    UniformFlowProgram<Descriptor, DynamicsSelection> program;
    const bool guo = program_detail::uses_guo(controls.variant);
    const amrex::Real dx =
        1.0 / static_cast<amrex::Real>(controls.intervals[0]);
    program.mesh.nCell = {
        controls.intervals[0] + (guo ? 1 : 0),
        controls.intervals[1],
        controls.depthCells};
    program.mesh.envelopeLo = {guo ? -0.5 * dx : 0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        guo ? 1.0 + 0.5 * dx : 1.0,
        1.0,
        dx * static_cast<amrex::Real>(controls.depthCells)};
    program.mesh.isPeriodic = {0, 1, 1};
    program.mesh.maxGridSize = controls.maxGridSize;

    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.velocityLb;
    program.lbm.uRefPhys = controls.velocityReferencePhys;
    program.lbm.lengthRefLb =
        static_cast<amrex::Real>(controls.intervals[0]);
    program.lbm.lengthRefPhys = 1.0;
    program.lbm.nuLb = controls.nuLb;
    program.lbm.streamingBackend = std::string("baseline");
    program.dynamics = std::move(dynamics);

    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "uniform_flow";
    program.run.variant = controls.runVariant;

    program.initialization = initialization::from_profile(
        UniformProfile{{controls.rhoLb, controls.velocityLb, 0.0, 0.0}});
    program.domainBoundaries = {
        program_detail::inlet(controls),
        program_detail::outlet(controls)};
    program.casePreparation = Preparation{std::move(controls)};
    return program;
}

} // namespace lbm::cases::uniform_flow

#endif
