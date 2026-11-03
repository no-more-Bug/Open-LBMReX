#ifndef LBM_CASES_TAYLORGREENVORTEX_PROGRAM_H
#define LBM_CASES_TAYLORGREENVORTEX_PROGRAM_H

#include "taylor_green_vortex/composition/Preparation.H"
#include "taylor_green_vortex/config/Controls.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_Array.H>

#include <string>
#include <utility>

namespace lbm::cases::taylor_green_vortex {

template <class Descriptor, class DynamicsSelection>
using TaylorGreenProgram = runtime::SingleLevelProgram<
    Descriptor,
    DynamicsSelection,
    initialization::ConfiguredInitializationSelection,
    Preparation>;

/** Form the complete periodic Taylor-Green family Program. */
template <class Descriptor, class DynamicsSelection>
TaylorGreenProgram<Descriptor, DynamicsSelection> make_taylor_green_program(
    Controls const& controls,
    DynamicsSelection dynamics)
{
    TaylorGreenProgram<Descriptor, DynamicsSelection> program;
    program.mesh.nCell = {
        controls.nCell[0], controls.nCell[1], controls.nCell[2]};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        static_cast<amrex::Real>(controls.nCell[2]) /
            static_cast<amrex::Real>(controls.nCell[0])};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = controls.maxGridSize;

    program.lbm.parameterizationKind = controls.parameterization;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefPhys = controls.uRefPhys;
    if (controls.parameterization == config::LbmParameterizationKind::FromNuLb) {
        program.lbm.uRefLb = controls.uRefLb;
        program.lbm.nuLb = controls.nuLb;
    } else {
        program.lbm.tau = controls.tau;
        program.lbm.reynolds = controls.reynolds;
    }
    program.lbm.streamingBackend = std::string("baseline");
    program.dynamics = std::move(dynamics);

    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "taylor_green_vortex";
    program.run.variant = controls.runVariant;
    program.run.convergenceEnabled = controls.convergenceEnabled;
    program.run.convergenceReset = controls.convergenceReset;

    program.initialization.intent.kind =
        initialization::InitializationKind::TaylorGreen;
    program.initialization.intent.rhoLb = controls.rhoLb;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb =
        controls.velocityAmplitudeLb;
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::cases::taylor_green_vortex

#endif
