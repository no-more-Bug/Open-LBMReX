#include "UniformFlowObservation.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <set>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace amr_uniform_flow;
    static_assert(std::numeric_limits<Real>::digits==53, "uniform-flow test requires FP64");
    int maxBox = 16, limit = -1;
    std::string directory, source;
    amrex::ParmParse("mesh").query("max_grid_size", maxBox);
    amrex::ParmParse controls("run");
    controls.query("max_step", limit); controls.get("source_id", source);
    amrex::ParmParse("output").get("directory", directory);
    if (maxBox<8 || maxBox%2 || limit < -1) amrex::Abort("invalid uniform-flow controls");
    auto const spec = geometry(Descriptor::d, maxBox);
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(spec, Initial{}, boundaries());
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto owner = prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromNuLb;
    intent.nuLb = Real(1)/12; intent.uRefLb = speed; intent.uRefPhys = 1; intent.rhoRefLb = 1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner, intent, lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto evolution = std::move(preparation).take();
    int const fullSteps = 32000, last = limit<0 ? fullSteps : std::min(limit, fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory, speed);
    UniformFlowObservation<Descriptor> observation(directory);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=amr-uniform-v1\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=0,1,1\nboundary=guo_velocity_io\ninitialization=constant-equilibrium"
           << "\nu0_lb=0.1\nu0_phys=1\nnu_fine_lb=" << Real(1)/12
           << "\nrho_reference=1\nvelocity_reference=0.1\nlength_z=0.125"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlevels=" << owner->numLevels()
           << "\nmax_grid_size=" << maxBox << "\nfull_steps=" << fullSteps << "\nrequested_steps=" << last << '\n';
        for (int lev=0; lev<owner->numLevels(); ++lev)
            id << "dt_" << lev << '=' << evolution.level(lev).dtPhys
               << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau << '\n';
        if (!id) amrex::Abort("cannot write uniform-flow identity");
    }
    owner->observe(true, [&](auto const& view) {
        output.layout(view); output.synchronized(view); observation.record(view);
    });
    std::set<int> samples{last};
    for (int step=320; step<=last; step+=320) samples.insert(step);
    int current = 0;
    for (int step : samples) {
        if (step<=current) continue;
        auto result = owner->evolve(evolution, step-current, false, [](auto const&) {});
        if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
        current = step;
        owner->observe(true, [&](auto const& view) {
            observation.record(view);
            if (step==16000 || step==32000 || (step==last && last<fullSteps)) output.synchronized(view);
        });
    }
    return 0;
}
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 2;
    try { status = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return status;
}
