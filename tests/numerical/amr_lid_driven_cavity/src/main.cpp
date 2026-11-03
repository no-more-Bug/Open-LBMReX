#include "CavityObservation.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace amr_cavity;
    static_assert(std::numeric_limits<Real>::digits == 53, "cavity tests require FP64");
    int box = 64, limit = -1;
    std::string layout = "amr", directory, source;
    amrex::ParmParse("case").query("layout", layout);
    amrex::ParmParse("mesh").query("max_grid_size", box);
    amrex::ParmParse controls("run");
    controls.query("max_step", limit); controls.get("source_id", source);
    amrex::ParmParse("output").get("directory", directory);
    if ((layout != "coarse" && layout != "fine" && layout != "amr") || box < 8 || box%2 || limit < -1)
        amrex::Abort("invalid cavity layout/grid/run controls");
    auto const spec = geometry(Descriptor::d, layout, box);
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(spec, Initial{}, boundaries());
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print() << error << '\n'; return 2; }
    auto owner = prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 1000; intent.uRefLb = lidSpeed; intent.uRefPhys = 1; intent.rhoRefLb = 1;
    intent.lengthRefLb = owner->geometry(owner->numLevels()-1).Domain().length(0);
    intent.lengthRefPhys = 1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner, intent, lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print() << error << '\n'; return 2; }
    auto evolution = std::move(preparation).take();
    int const stride = 10*spec.nCell[0], comparisonSteps = layout == "fine" ? 2 : 1;
    int const fullSteps = 200*stride, last = limit < 0 ? fullSteps : std::min(limit, fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory, lidSpeed);
    CavityObservation<Descriptor> observation(directory);
    Real initialMass = 0;
    owner->observe(true, [&](auto const& view) {
        output.layout(view); output.synchronized(view); initialMass = observation.initialize(view);
    });
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=cavity-v1\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=0,0,1\ninitialization=cell-center-equilibrium"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlayout=" << layout
           << "\nbase_cells=" << spec.nCell[0] << "\nbase_z=" << spec.nCell[2] << "\nlength_z=" << spec.probHi[2]
           << "\nlevels=" << owner->numLevels() << "\nmax_grid_size=" << box
           << "\nreynolds=1000\nu0_lb=0.1\nu0_phys=1\nviscosity=0.001\nrho_reference=1\nvelocity_reference=0.1"
           << "\ngap_fill_radius=4\ncoarse_nesting_margin=4\ngrid_efficiency=1"
           << "\nfull_steps=" << fullSteps << "\nrequested_steps=" << last << "\ncomparison_steps=" << comparisonSteps
           << "\ninitial_mass=" << initialMass << '\n';
        for (int lev = 0; lev < owner->numLevels(); ++lev)
            id << "dt_" << lev << '=' << evolution.level(lev).dtPhys << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau << '\n';
        if (!id) amrex::Abort("cannot write cavity identity");
    }
    int current = 0;
    auto advance = [&](int target) {
        auto result = owner->evolve(evolution, target-current, false, [](auto const&) {});
        if (!result.ok()) amrex::Abort(result.error);
        current = target;
    };
    for (int check = 1; check <= 200 && check*stride <= last; ++check) {
        int const end = check*stride;
        advance(end-comparisonSteps);
        owner->observe(true, [&](auto const& view) {
            observation.save(view);
            if (check >= 198 || (end == last && last < fullSteps)) output.synchronized(view);
        });
        advance(end);
        owner->observe(true, [&](auto const& view) {
            observation.record(view);
            if (check%20 == 0 || check >= 198 || (end == last && last < fullSteps)) output.synchronized(view);
            amrex::Print() << "cavity_sample step=" << view.step << " time=" << view.physicalTime << '\n';
        });
    }
    // Short runs retain their actual final frame and are incomplete scientific evidence.
    if (current < last) {
        advance(last);
        owner->observe(true, [&](auto const& view) { output.synchronized(view); });
    }
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int result = 2;
    try { result = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return result;
}
