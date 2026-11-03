#include "unit/lbm/LbmConfigurationTests.H"
#include "unit/lbm/LbmCoreTests.H"

#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"

#include <AMReX.H>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        lbm::config::tests::run_lbm_configuration_tests<ActiveDescriptor>();
        lbm::tests::run_lbm_core_tests<ActiveDescriptor>();
    }
    amrex::Finalize();
    return 0;
}
