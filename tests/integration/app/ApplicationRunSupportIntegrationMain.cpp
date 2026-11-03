#include "integration/app/TimestepProfilerIntegrationTests.H"

#include <AMReX.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        lbm::app::run_timestep_profiler_integration_tests();
    }
    amrex::Finalize();
    return 0;
}
