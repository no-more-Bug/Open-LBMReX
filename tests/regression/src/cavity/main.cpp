#include "cavity/CavityShort.H"

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
    const int status = lbm::tests::regression::execute_short_program(
        lbm::tests::regression::cavity::make_program<ActiveDescriptor>(
            lbm::tests::regression::cavity::read_direct_controls()),
        "Regression cavity preparation failed:");
    amrex::Finalize();
    return status;
}
