#include "integration/lbm/AmrBoundaryIntegrationTests.H"
#include "integration/runtime/AmrBoundaryEvolutionTests.H"
#include "integration/runtime/AmrBoundaryForceTests.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
#ifdef LBM_DESCRIPTOR_D3Q27
        using Descriptor = lbm::D3Q27Descriptor;
#else
        using Descriptor = lbm::D2Q9Descriptor;
#endif
        std::string mode = "prepare";
        amrex::ParmParse("test").query("mode", mode);
        if (mode == "prepare") lbm::tests::integration::amr_boundary_test::run<Descriptor>();
        else if (mode == "evolve") lbm::tests::integration::amr_boundary_test::run_evolution_checks<Descriptor>();
        else if (mode == "force") lbm::tests::integration::amr_boundary_test::run_force_checks<Descriptor>();
        else amrex::Abort("unknown boundary integration test mode");
    }
    amrex::Finalize();
    return 0;
}
