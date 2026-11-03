#include "channel_cylinder/composition/ChannelCylinderProgram.H"
#include "channel_cylinder/config/Input.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
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

std::string format_errors(
    char const* heading,
    std::vector<std::string> const& errors)
{
    std::ostringstream os;
    os << heading;
    for (std::string const& error : errors) {
        os << "\n  - " << error;
    }
    return os.str();
}

std::string format_setup_diagnostics(
    std::vector<lbm::runtime::SingleLevelSetupDiagnostic> const& diagnostics)
{
    std::ostringstream os;
    os << "Direct channel-cylinder preparation failed:";
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
            lbm::cases::channel_cylinder::read_controls();
        if (!controls.ok()) {
            amrex::Print() << format_errors(
                "Direct channel-cylinder input validation failed:",
                controls.errors) << '\n';
            exitStatus = 2;
        } else {
            auto program =
                lbm::cases::channel_cylinder::
                    make_channel_cylinder_program<ActiveDescriptor>(
                        std::move(controls.controls),
                        lbm::BGKDynamicsSelection{});
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
