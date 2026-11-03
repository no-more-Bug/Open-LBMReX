#include "integration/runtime/AmrEvolutionTests.H"
#include "integration/runtime/AmrEvolutionSnapshot.H"
#include "integration/runtime/AmrProfilingTests.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <string>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    std::string mode="lifecycle";
    amrex::ParmParse("test").query("mode",mode);
    if (mode=="lifecycle") lbm::tests::amr_evolution::lifecycle_checks();
    else if (mode=="advance") {
        bool matrix=false;
        amrex::ParmParse("test").query("matrix",matrix);
        if (matrix) lbm::tests::amr_evolution::matrix_checks();
        else lbm::tests::amr_evolution::advance_checks();
    }
    else if (mode=="output") lbm::tests::amr_evolution::output_checks();
    else if (mode=="profiling") lbm::tests::amr_evolution::profiling_checks();
    else amrex::Abort("unknown AMR evolution test mode");
    amrex::Finalize();
}
