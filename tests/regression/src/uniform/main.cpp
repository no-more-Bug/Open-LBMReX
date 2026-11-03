#include "uniform/UniformShort.H"

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
        lbm::tests::regression::uniform::make_program<ActiveDescriptor>(
            lbm::tests::regression::uniform::read_direct_controls()),
        "Regression uniform-flow preparation failed:");
    amrex::Finalize();
    return status;
}
