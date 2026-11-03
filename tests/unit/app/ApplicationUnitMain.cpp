#include "unit/app/ApplicationValueTests.H"

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
        lbm::app::tests::run_application_value_tests<ActiveDescriptor>();
    }
    amrex::Finalize();
    return 0;
}
