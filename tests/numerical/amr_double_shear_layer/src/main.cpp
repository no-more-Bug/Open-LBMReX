#include "DoubleShear.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <algorithm>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor=lbm::D3Q27Descriptor;
#else
using Descriptor=lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace double_shear;
    static_assert(std::numeric_limits<Real>::digits==53,"numerical tests require FP64");
    int cells=128, box=32, limit=-1;
    std::string layout="baseline", directory, source;
    amrex::ParmParse input("case"); input.query("layout",layout); input.query("uniform_cells",cells);
    amrex::ParmParse("mesh").query("max_grid_size",box);
    amrex::ParmParse controls("run"); controls.query("max_step",limit); controls.get("source_id",source);
    amrex::ParmParse("output").get("directory",directory);
    if ((cells!=128 && cells!=256 && cells!=512) || box<8 || box%2 || limit< -1)
        amrex::Abort("invalid DSL grid/run controls");
    auto spec=geometry(Descriptor::d,layout,cells,box);
    auto prepared=lbm::amr::prepare_static_hierarchy<Descriptor>(spec,Initial{});
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto owner=prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind=lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds=reynolds; intent.uRefLb=speed; intent.uRefPhys=1; intent.rhoRefLb=1;
    auto const& finest=owner->geometry(owner->numLevels()-1);
    intent.lengthRefLb=finest.Domain().length(0);
    intent.lengthRefPhys=finest.ProbHi(0)-finest.ProbLo(0);
    auto preparation=lbm::runtime::prepare_amr_evolution(*owner,intent,lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto evolution=std::move(preparation).take();
    int const n=spec.nCell[0], fullSteps=20*n, last=limit<0 ? fullSteps : std::min(limit,fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory,speed);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id<<std::setprecision(17)<<"schema=double-shear-v1\ndescriptor="<<Descriptor::name()
          <<"\ndimensions="<<Descriptor::d<<"\nsource_id="<<source<<"\nbackend=";
#ifdef AMREX_USE_CUDA
        id<<"cuda";
#else
        id<<"cpu";
#endif
        id<<"\ncollision=BGK\nperiodic=1,1,1\ninitialization=cell-center-equilibrium\nranks="<<amrex::ParallelDescriptor::NProcs()<<"\nreal_digits=53\nlayout="<<layout
          <<"\nbase_cells="<<n<<"\nbase_z="<<spec.nCell[2]<<"\nlength_z="<<spec.probHi[2]
          <<"\nlevels="<<owner->numLevels()<<"\nmax_grid_size="<<box
          <<"\nreynolds="<<reynolds<<"\nu0_lb="<<speed<<"\nu0_phys=1\nviscosity=0.0001\nkappa=80\ndelta=0.05"
          <<"\nrho_reference=1\nvelocity_reference="<<speed
          <<"\ngap_fill_radius=4\ncoarse_nesting_margin=4\ngrid_efficiency=1"
          <<"\nfull_steps="<<fullSteps<<"\nrequested_steps="<<last<<"\nframe_stride="<<n/2<<'\n';
        for (int lev=0; lev<owner->numLevels(); ++lev)
            id<<"dt_"<<lev<<'='<<evolution.level(lev).dtPhys<<"\ntau_"<<lev<<'='<<evolution.level(lev).lbm.tau<<'\n';
        if (!id) amrex::Abort("cannot write DSL identity");
    }
    owner->observe(true,[&](auto const& view) { output.layout(view); output.synchronized(view); });
    for (int current=0; current<last;) {
        int const target=std::min(current+n/2,last);
        auto result=owner->evolve(evolution,target-current,false,[](auto const&) {});
        if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
        current=target;
        owner->observe(true,[&](auto const& view) { output.synchronized(view); });
        amrex::Print()<<"dsl_sample step="<<current<<" time="<<current*evolution.level(0).dtPhys<<'\n';
    }
    return 0;
}
}

int main(int argc,char* argv[])
{
    amrex::Initialize(argc,argv);
    int result=2;
    try { result=run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return result;
}
