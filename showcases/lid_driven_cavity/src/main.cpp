#ifdef LBM_DESCRIPTOR_D3Q27
#include "lid_driven_cavity/composition/CavityProgram.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#else
#include "lid_driven_cavity/composition/D2Q9GuoCavityProgram.H"
#endif
#include "lid_driven_cavity/config/Input.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

std::string format_input_errors(std::vector<std::string> const& errors)
{
    std::ostringstream os;
    os << "Direct cavity input validation failed:";
    for (std::string const& error : errors) {
        os << "\n  - " << error;
    }
    return os.str();
}

std::string format_setup_diagnostics(
    std::vector<lbm::runtime::SingleLevelSetupDiagnostic> const& diagnostics)
{
    std::ostringstream os;
    os << "Direct cavity preparation failed:";
    for (auto const& diagnostic : diagnostics) {
        os << "\n  - [" << lbm::runtime::to_string(diagnostic.stage)
           << ':' << diagnostic.code << ':' << diagnostic.subject << "] "
           << diagnostic.message;
    }
    return os.str();
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int exitStatus = 0;
    {
        auto controls =
            lbm::cases::lid_driven_cavity::read_controls();
        if (!controls.ok()) {
            amrex::Print() << format_input_errors(controls.errors) << '\n';
            exitStatus = 2;
        } else {
            using Dynamics = lbm::BGKDynamicsSelection;
            using WallScheme =
                lbm::cases::lid_driven_cavity::WallScheme;
            using InitializationProfile =
                lbm::initialization::ConstantInitializationProfile;
            auto initialization = lbm::initialization::from_profile(
                InitializationProfile{
                    {controls.controls.rhoLb, 0.0, 0.0, 0.0}});
            auto program = [&] {
                auto caseControls = std::move(controls.controls);
#ifndef LBM_DESCRIPTOR_D3Q27
                if (caseControls.wallScheme ==
                    WallScheme::GuoExtrapolation) {
                    return lbm::cases::lid_driven_cavity::
                        make_d2q9_guo_cavity_program<
                            ActiveDescriptor,
                            Dynamics>(
                                std::move(caseControls),
                                Dynamics{},
                                std::move(initialization));
                }
#endif
                return lbm::cases::lid_driven_cavity::make_cavity_program<
                    ActiveDescriptor,
                    Dynamics>(
                        std::move(caseControls),
                        Dynamics{},
                        std::move(initialization));
            }();
            auto preparation =
                lbm::runtime::prepare_single_level(std::move(program));
            if (!preparation.ok()) {
                amrex::Print() <<
                    format_setup_diagnostics(preparation.diagnostics())
                    << '\n';
                exitStatus = 3;
            } else {
                auto report = lbm::runtime::run_single_level(
                    std::move(preparation).take());
                if (!report.ok()) {
                    amrex::Print() << report.message << '\n';
                    exitStatus = 4;
                }
            }
        }
    }
    amrex::Finalize();
    return exitStatus;
}
