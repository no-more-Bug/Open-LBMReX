#include "ShearWave.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <set>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor=lbm::D3Q27Descriptor;
#else
using Descriptor=lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace periodic_shear;
    static_assert(std::numeric_limits<Real>::digits==53,"numerical tests require FP64");
    int nc=Descriptor::d==2 ? 64 : 32, box=32, limit=-1;
    std::string mode="axis", layout="half", directory, source;
    amrex::ParmParse input("case");
    input.query("coarse_cells",nc); input.query("mode",mode); input.query("layout",layout);
    amrex::ParmParse("mesh").query("max_grid_size",box);
    amrex::ParmParse controls("run");
    controls.query("max_step",limit); controls.get("source_id",source);
    amrex::ParmParse("output").get("directory",directory);
    if (nc<32 || nc%16 || box<8 || box%2 || limit< -1) amrex::Abort("invalid shear grid/run controls");
    auto spec=geometry(Descriptor::d,nc,layout,box);
    auto prepared=lbm::amr::prepare_static_hierarchy<Descriptor>(spec,wave(Descriptor::d,mode));
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto owner=prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind=lbm::config::LbmParameterizationKind::FromNuLb;
    intent.nuLb=viscosity*Real(2*nc); intent.uRefLb=background; intent.uRefPhys=background;
    intent.rhoRefLb=1;
    auto preparation=lbm::runtime::prepare_amr_evolution(*owner,intent,lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto evolution=std::move(preparation).take();
    int const n0=spec.nCell[0], fullSteps=4*n0, last=limit<0 ? fullSteps : std::min(limit,fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory,1);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=periodic-shear-v1\ndescriptor="<<Descriptor::name()
           <<"\ndimensions="<<Descriptor::d<<"\nsource_id="<<source<<"\nbackend=";
#ifdef AMREX_USE_CUDA
        id<<"cuda";
#else
        id<<"cpu";
#endif
        id<<"\ncollision=BGK\nperiodic=1,1,1\ninitialization=cell-center-equilibrium\nranks="<<amrex::ParallelDescriptor::NProcs()<<"\nreal_digits=53\ncoarse_cells="<<nc
          <<"\nbase_cells="<<n0<<"\nmode="<<mode<<"\nlayout="<<layout
          <<"\nlevels="<<owner->numLevels()<<"\nmax_grid_size="<<box
          <<"\nviscosity="<<viscosity<<"\nbackground="<<background<<"\namplitude="<<amplitude
          <<"\nrho_reference=1\nvelocity_reference=1\ngap_fill_radius=4\ncoarse_nesting_margin=4\ngrid_efficiency=1"
          <<"\nfull_steps="<<fullSteps<<"\nrequested_steps="<<last<<'\n';
        for (int lev=0; lev<owner->numLevels(); ++lev)
            id<<"dt_"<<lev<<'='<<evolution.level(lev).dtPhys<<"\ntau_"<<lev<<'='<<evolution.level(lev).lbm.tau<<'\n';
        if (!id) amrex::Abort("cannot write shear identity");
    }
    owner->observe(true,[&](auto const& view) { output.layout(view); output.synchronized(view); });
    std::set<int> samples{n0/8,n0/4,n0/2,n0,2*n0,4*n0,last};
    if (layout=="uniform") for (int t : {1,2,4}) for (int j=0; j<4; ++j) samples.insert(t*n0-j);
    int current=0;
    for (int step : samples) {
        if (step<=current || step>last) continue;
        auto result=owner->evolve(evolution,step-current,false,[](auto const&) {},
            [&](lbm::amr::FineSubstepView<Descriptor> const& view) {
                if (fine_sample(view.fine.step,2*nc)) output.fine(view);
            });
        if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
        current=step;
        owner->observe(true,[&](auto const& view) { output.synchronized(view); });
        amrex::Print()<<"shear_sample step="<<step<<" time="<<step*evolution.level(0).dtPhys<<'\n';
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
