#include "PerformanceControls.H"

#ifdef LBM_PERFORMANCE_DFG
#include "DfgPerformance.H"
#else
#include "TaylorGreenPerformance.H"
#endif

#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_Print.H>

#include <utility>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

template <class Program>
int execute(Program program)
{
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    if (!prepared.ok()) {
        for (auto const& diagnostic : prepared.diagnostics()) {
            amrex::Print()
                << '[' << lbm::runtime::to_string(diagnostic.stage) << ':'
                << diagnostic.code << ':' << diagnostic.subject << "] "
                << diagnostic.message << '\n';
        }
        return 2;
    }

    lbm::runtime::SingleLevelRunReport const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    if (report.ok()) {
        return 0;
    }
    amrex::Print() << report.message << '\n';
    return report.status == lbm::runtime::SingleLevelRunStatus::NumericalFailure
        ? 1
        : 2;
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 0;
    {
        auto controls =
            lbm::tests::performance::read_performance_controls();
#ifdef LBM_PERFORMANCE_DFG
        status = execute(
            lbm::tests::performance::make_dfg_performance_program<
                ActiveDescriptor>(std::move(controls)));
#else
        status = execute(
            lbm::tests::performance::make_taylor_green_performance_program<
                ActiveDescriptor>(std::move(controls)));
#endif
    }
    amrex::Finalize();
    return status;
}
