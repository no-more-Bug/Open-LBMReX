#include "integration/lbm/AmrFieldIntegrationTests.H"
#include "integration/lbm/LbmDiagnosticsIntegrationTests.H"
#include "integration/lbm/LbmDomainIntegrationTests.H"
#include "integration/lbm/LbmValidBoundaryInitializationIntegrationTests.H"

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
        using namespace lbm::tests::integration;
        run_field_integration_tests<ActiveDescriptor>();
        run_amr_field_integration_tests<ActiveDescriptor>();
        run_diagnostics_primitive_integration_tests<ActiveDescriptor>();
        run_operator_integration_tests<ActiveDescriptor>();
        run_boundary_resolver_integration_tests<ActiveDescriptor>();
        run_post_stream_boundary_geometry_integration_tests<ActiveDescriptor>();
        run_valid_boundary_initialization_integration_tests<ActiveDescriptor>();
        run_guo_pressure_executor_integration_tests<ActiveDescriptor>();
        run_guo_prescribed_state_executor_integration_tests<ActiveDescriptor>();
        run_guo_moving_wall_executor_integration_tests<ActiveDescriptor>();
        run_post_stream_boundary_lifecycle_integration_tests<ActiveDescriptor>();
        run_post_stream_boundary_decomposition_integration_tests<ActiveDescriptor>();
        run_domain_boundary_fill_integration_tests<ActiveDescriptor>();
        run_hbb_multibox_read_integration_tests<ActiveDescriptor>();
    }
    amrex::Finalize();
    return 0;
}
