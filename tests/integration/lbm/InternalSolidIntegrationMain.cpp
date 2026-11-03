#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "integration/lbm/InternalSolidForceIntegrationTests.H"
#include "integration/lbm/InternalSolidForceValidationTests.H"
#include "integration/lbm/InternalSolidMetadataIntegrationTests.H"
#include "integration/lbm/InternalSolidTrackedFixtureTests.H"

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
    lbm::tests::integration::run_internal_solid_metadata_integration_tests<
        ActiveDescriptor>();
    lbm::tests::integration::run_internal_solid_tracked_fixture_tests<
        ActiveDescriptor>();
    lbm::tests::integration::run_internal_solid_force_integration_tests<
        ActiveDescriptor>();
    lbm::tests::integration::run_internal_solid_force_validation_tests<
        ActiveDescriptor>();
    amrex::Finalize();
    return 0;
}
