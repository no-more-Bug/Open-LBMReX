#include "Re20Observation.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace amr_re20;
    static_assert(std::numeric_limits<Real>::digits == 53, "Re20 tests require FP64");
    int box = 64, limit = -1;
    std::string layout = "amr", scheme = "yms", directory, source;
    amrex::ParmParse caseControls("case");
    caseControls.query("layout", layout); caseControls.query("scheme", scheme);
    amrex::ParmParse("mesh").query("max_grid_size", box);
    amrex::ParmParse controls("run");
    controls.query("max_step", limit); controls.get("source_id", source);
    amrex::ParmParse("output").get("directory", directory);
    if ((layout != "coarse" && layout != "fine" && layout != "amr") ||
        (scheme != "yms" && scheme != "bouzidi") || box < 2 || box%2 || limit < -1)
        amrex::Abort("invalid Re20 scheme/layout/grid/run controls");
    auto const spec = geometry(Descriptor::d, layout, box);
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(spec, Initial{}, boundaries(scheme));
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print() << error << '\n'; return 2; }
    auto owner = prepared.take();
    Real const finestH = owner->geometry(owner->numLevels()-1).CellSize(0);
    Real const diameterLb = Real(0.1)/finestH, spanLb = Real(0.01)/finestH;
    Real const forceDenominator = Real(0.5)*meanSpeed*meanSpeed*diameterLb*(Descriptor::d == 3 ? spanLb : 1);
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 20; intent.uRefLb = meanSpeed; intent.uRefPhys = 0.2; intent.rhoRefLb = 1;
    intent.lengthRefLb = diameterLb; intent.lengthRefPhys = 0.1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner, intent, lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print() << error << '\n'; return 2; }
    auto evolution = std::move(preparation).take();
    auto const& cylinder = owner->resolvedBoundaries().internal.at(0);
    int const ownerLevel = owner->internalBoundaryLevel(cylinder.boundaryId);
    int const comparisonSteps = layout == "fine" ? 4 : 1, stride = 100*comparisonSteps;
    int const observationCount = 800;
    int const fullSteps = observationCount*stride, last = limit < 0 ? fullSteps : std::min(limit, fullSteps);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory, meanSpeed);
    Re20Observation<Descriptor> observation(directory, forceDenominator, cylinder.objectId);
    owner->observe(true, [&](auto const& view) {
        output.layout(view); output.synchronized(view); observation.initialize(view);
    });
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << "schema=re20-v1\ndescriptor=" << Descriptor::name() << "\ndimensions=" << Descriptor::d
           << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=0,0,1\ninitialization=uniform-equilibrium-then-Guo"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlayout=" << layout
           << "\nscheme=" << (scheme == "yms" ? "YMS" : "Bouzidi")
           << "\nphysical_lo=0,0,0\nphysical_hi=2.2,0.41,0.01"
           << "\nenvelope_lo=" << spec.probLo[0] << ',' << spec.probLo[1] << ',' << spec.probLo[2]
           << "\nenvelope_hi=" << spec.probHi[0] << ',' << spec.probHi[1] << ',' << spec.probHi[2]
           << "\ncylinder_center=0.2,0.2\ncylinder_radius=0.05"
           << "\nboundary_id=" << cylinder.boundaryId.value << "\nobject_id=" << cylinder.objectId.value
           << "\nowner_level=" << ownerLevel << "\nlevels=" << owner->numLevels() << "\nmax_grid_size=" << box
           << "\nreynolds=20\nu0_lb=0.035\nu0_phys=0.2\nviscosity=0.001\nrho_reference=1\nvelocity_reference=0.035"
           << "\ngap_fill_radius=4\ncoarse_nesting_margin=4\ngrid_efficiency=1"
           << "\ndiameter_lb=" << diameterLb << "\nspan_lb=" << spanLb << "\nforce_denominator=" << forceDenominator
           << "\nfull_steps=" << fullSteps << "\nrequested_steps=" << last << "\ncomparison_steps=" << comparisonSteps
           << "\ncomparison_dt=" << comparisonDt << '\n';
        if (auto const* threads = std::getenv("OMP_NUM_THREADS")) id << "omp_threads=" << threads << '\n';
        for (int lev = 0; lev < owner->numLevels(); ++lev) {
            auto const& geom = owner->geometry(lev);
            id << "cells_" << lev << '=' << geom.Domain().length(0) << ',' << geom.Domain().length(1) << ',' << geom.Domain().length(2)
               << "\ndx_" << lev << '=' << geom.CellSize(0) << ',' << geom.CellSize(1) << ',' << geom.CellSize(2)
               << "\ndt_" << lev << '=' << evolution.level(lev).dtPhys << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau
               << "\nlattice_scale_" << lev << '=' << evolution.level(lev).dxLb() << '\n';
        }
        if (!id) amrex::Abort("cannot write Re20 identity");
    }
    int current = 0;
    auto advance = [&](int target) {
        // evolve publishes and consumes one force batch after every completed L0 step.
        auto result = owner->evolve(evolution, target-current, false,
            [&](auto const& view) { observation.consume(view.forceBatch); });
        if (!result.ok()) amrex::Abort(result.error);
        current = target;
    };
    for (int check = 1; check <= observationCount && check*stride <= last; ++check) {
        int const end = check*stride;
        advance(end-comparisonSteps);
        owner->observe(true, [&](auto const& view) {
            observation.save(view);
            if (check >= observationCount-2 || (end == last && last < fullSteps)) output.synchronized(view);
        });
        advance(end);
        owner->observe(true, [&](auto const& view) {
            observation.record(view, check*100);
            if (check == 1 || check == 10 || check%(observationCount/10) == 0 || check >= observationCount-2 || (end == last && last < fullSteps))
                output.synchronized(view);
            amrex::Print() << "re20_sample step=" << view.step << " common_step=" << check*100 << " time=" << view.physicalTime << '\n';
        });
    }
    // Short runs retain the actual endpoint; they do not claim full scientific completion.
    if (current < last) {
        advance(last);
        owner->observe(true, [&](auto const& view) { output.synchronized(view); });
    }
    owner->observe(true, [&](auto const& view) { observation.profiles(view); });
    observation.finish();
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
