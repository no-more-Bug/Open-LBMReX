#include "Re100Case.H"
#include "Re100Observation.H"
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
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace amr_re100;
    static_assert(std::numeric_limits<Real>::digits == 53, "Re100 tests require FP64");
    auto const initializationStart = amrex::second();
    int maxBox = 64, limit = -1;
    std::string layout = "amr", scheme = "yms", directory, source;
    amrex::ParmParse caseInput("case");
    caseInput.query("layout", layout); caseInput.query("scheme", scheme);
    amrex::ParmParse("mesh").query("max_grid_size", maxBox);
    amrex::ParmParse controls("run");
    controls.query("max_step", limit); controls.get("source_id", source);
    amrex::ParmParse("output").get("directory", directory);
    if ((layout != "coarse" && layout != "fine" && layout != "amr") ||
        (scheme != "yms" && scheme != "bouzidi") || maxBox < 2 || maxBox%2 || limit < -1)
        amrex::Abort("invalid Re100 controls");
    auto const spec = geometry(Descriptor::d, layout, maxBox);
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(spec, Initial{}, boundaries(scheme));
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto owner = prepared.take();
    Real const finestH = owner->geometry(owner->numLevels()-1).CellSize(0);
    Real const diameterLb = Real(0.1)/finestH, spanLb = Real(0.01)/finestH;
    Real const forceDenominator = Real(0.5)*meanSpeed*meanSpeed*diameterLb*(Descriptor::d == 3 ? spanLb : 1);
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 100; intent.uRefLb = meanSpeed; intent.uRefPhys = 1; intent.rhoRefLb = 1;
    intent.lengthRefLb = diameterLb; intent.lengthRefPhys = 0.1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner, intent, lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto evolution = std::move(preparation).take();
    auto const& cylinder = owner->resolvedBoundaries().internal.at(0);
    int const ownerLevel = owner->internalBoundaryLevel(cylinder.boundaryId);
    int const fullSteps = layout == "fine" ? 800000 : layout == "coarse" ? 400000 : 200000;
    int const last = limit < 0 ? fullSteps : std::min(limit, fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory, meanSpeed);
    Re100Observation observation(directory, forceDenominator);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=amr-re100-v1\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=0,0,1\ninitialization=uniform-equilibrium-then-Guo"
           << "\nreynolds=100\nviscosity=0.001\nu0_lb=0.035\nu0_phys=1\nrho_reference=1\nvelocity_reference=0.035"
           << "\nlength_z=0.01\nlayout=" << layout << "\nscheme=" << (scheme == "yms" ? "YMS" : "Bouzidi")
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlevels=" << owner->numLevels()
           << "\nmax_grid_size=" << maxBox << "\nfull_steps=" << fullSteps << "\nrequested_steps=" << last
           << "\nboundary_id=" << cylinder.boundaryId.value << "\nobject_id=" << cylinder.objectId.value
           << "\nowner_level=" << ownerLevel << "\ndiameter_lb=" << diameterLb << "\nspan_lb=" << spanLb
           << "\nforce_denominator=" << forceDenominator << '\n';
        for (int lev=0; lev<owner->numLevels(); ++lev)
            id << "dt_" << lev << '=' << evolution.level(lev).dtPhys
               << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau << '\n';
        if (!id) amrex::Abort("cannot write Re100 identity");
    }
    amrex::Gpu::streamSynchronize();
    double initializationSeconds = amrex::second()-initializationStart;
    double advanceSeconds = 0, forceWriteSeconds = 0, fieldOutputSeconds = 0;
    auto sample = [&](bool initial) {
        auto const begin = amrex::second();
        owner->observe(true, [&](auto const& view) {
            if (initial) output.layout(view);
            output.synchronized(view);
        });
        fieldOutputSeconds += amrex::second()-begin;
    };
    sample(true);
    std::set<int> samples{last};
    for (int fifth=1; fifth<=5; ++fifth) samples.insert(fullSteps*fifth/5);
    int current = 0;
    for (int step : samples) {
        if (step<=current || step>last) continue;
        auto const begin = amrex::second();
        auto result = owner->evolve(evolution, step-current, false, [&](auto const& view) {
            auto const writeBegin = amrex::second();
            observation.consume(view.forceBatch);
            forceWriteSeconds += amrex::second()-writeBegin;
        });
        amrex::Gpu::streamSynchronize();
        advanceSeconds += amrex::second()-begin;
        if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
        current = step;
        sample(false);
    }
    auto const flushBegin = amrex::second();
    observation.finish();
    double const flushSeconds = amrex::second()-flushBegin;
    double timings[] = {initializationSeconds, advanceSeconds-forceWriteSeconds,
                        forceWriteSeconds+flushSeconds, fieldOutputSeconds,
                        initializationSeconds+advanceSeconds+flushSeconds+fieldOutputSeconds};
    amrex::ParallelDescriptor::ReduceRealMax(timings, 5);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream timeOutput(directory+"/run_metrics.csv");
        timeOutput << std::setprecision(17)
                   << "initialization_s,advance_excluding_force_write_s,force_write_s,field_output_s,total_s\n";
        for (int i=0; i<5; ++i) timeOutput << (i ? "," : "") << timings[i];
        timeOutput << '\n';
        if (!timeOutput) amrex::Abort("cannot write Re100 timing");
    }
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int status = 2;
    try { status = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return status;
}
