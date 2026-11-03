#include "TandemCylinder.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

namespace {
using Real = amrex::Real;
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

void write_vector(std::ostream& out, amrex::GpuArray<Real,3> const& force)
{
    for (auto component : force) out << ',' << component;
    out << '\n';
}

void write_step(std::ostream& out, char const* kind, int id,
                lbm::boundary::InternalForceStep const& step, amrex::GpuArray<Real,3> const& force)
{
    out << kind << ',' << id << ',' << step.level << ',' << step.populationStep << ','
        << step.coarsestStep << ',' << step.beginPhysicalTime << ',' << step.endPhysicalTime
        << ',' << step.samplePhysicalTime();
    write_vector(out, force);
}

void write_batch(std::ostream& out, lbm::boundary::InternalForceBatch const& batch)
{
    for (auto const& sample : batch.boundarySteps)
        for (auto const& value : sample.boundaryForces)
            write_step(out, "boundary", value.boundaryId.value, sample.step, value.forceLb);
    for (auto const& sample : batch.objectSteps)
        for (auto const& value : sample.objectForces)
            write_step(out, "object_step", value.objectId.value, sample.step, value.forceLb);
    for (auto const& sample : batch.objectIntervals)
        for (auto const& value : sample.objectForces) {
            // An interval average has no level step or instantaneous sample label.
            out << "object_interval," << value.objectId.value << ",,," << sample.coarsestStep
                << ',' << sample.beginPhysicalTime << ',' << sample.endPhysicalTime << ',';
            write_vector(out, value.forceLb);
        }
}

int run()
{
    static_assert(std::numeric_limits<Real>::digits == 53, "Tandem tests require FP64");
    std::string layout = "cross", grouping = "distinct", scheme = "yms", directory, source;
    std::string mode = "short";
    int maxGridSize = 64;
    amrex::ParmParse caseControls("case");
    caseControls.query("layout", layout);
    caseControls.query("grouping", grouping);
    caseControls.query("scheme", scheme);
    amrex::ParmParse("mesh").query("max_grid_size", maxGridSize);
    amrex::ParmParse("run").get("source_id", source);
    amrex::ParmParse("run").query("mode", mode);
    amrex::ParmParse("output").get("directory", directory);
    if ((layout != "cross" && layout != "same") ||
        (grouping != "shared" && grouping != "distinct") ||
        (scheme != "yms" && scheme != "bouzidi") || maxGridSize < 2 || maxGridSize%2)
        amrex::Abort("invalid tandem scheme/layout/grouping/grid controls");
    if (mode != "short" && mode != "compare") amrex::Abort("invalid tandem run.mode");
    bool const compare = mode == "compare";
    if (compare && (Descriptor::d != 2 || scheme != "yms" || grouping != "shared" || maxGridSize != 64))
        amrex::Abort("tandem comparison requires D2Q9/YMS/shared/max_grid_size=64");
    std::vector<int> const snapshots = compare
        ? std::vector<int>{0, 100, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000}
        : std::vector<int>{8};

    auto const spec = amr_tandem::geometry(Descriptor::d, layout == "cross", maxGridSize);
    auto prepared = lbm::amr::prepare_static_hierarchy<Descriptor>(
        spec, amr_re20::Initial{}, amr_tandem::boundaries(scheme, grouping == "shared"));
    if (!prepared.ok()) {
        for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
        return 2;
    }
    auto owner = prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 20;
    intent.uRefLb = amr_re20::meanSpeed;
    intent.uRefPhys = 0.2;
    intent.rhoRefLb = 1;
    intent.lengthRefLb = Real(0.1)/owner->geometry(owner->numLevels()-1).CellSize(0);
    intent.lengthRefPhys = 0.1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner, intent, lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) {
        for (auto const& error : preparation.errors) amrex::Print() << error << '\n';
        return 2;
    }
    auto evolution = std::move(preparation).take();
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory, amr_re20::meanSpeed);
    owner->observe(false, [&](auto const& view) { output.layout(view); });
    std::ofstream forces;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << "schema=" << (compare ? "tandem-compare-v1" : "tandem-v1")
           << "\nsource_id=" << source << "\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\nlayout=" << layout << "\ngrouping=" << grouping
           << "\nscheme=" << (scheme == "yms" ? "YMS" : "Bouzidi")
           << "\ncollision=BGK\nperiodic=0,0,1\ninitialization=uniform-equilibrium-then-Guo"
           << "\nlevels=" << owner->numLevels() << "\nfull_steps=" << snapshots.back() << "\nreal_digits=53"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nmax_grid_size=" << maxGridSize
           << "\nphysical_lo=0,0,0\nphysical_hi=2.2,0.41,0.01"
           << "\nreynolds=20\nu0_lb=0.035\nu0_phys=0.2\nviscosity=0.001"
           << "\nrho_reference=1\nvelocity_reference=0.035\nforce_units=finest-lattice"
           << "\ncomparison_dt=" << amr_re20::comparisonDt << '\n';
        if (auto const* threads = std::getenv("OMP_NUM_THREADS")) id << "omp_threads=" << threads << '\n';
        for (int lev = 0; lev < owner->numLevels(); ++lev) {
            auto const& geom = owner->geometry(lev);
            id << "cells_" << lev << '=' << geom.Domain().length(0) << ',' << geom.Domain().length(1)
               << ',' << geom.Domain().length(2) << "\ndx_" << lev << '=' << geom.CellSize(0)
               << ',' << geom.CellSize(1) << ',' << geom.CellSize(2)
               << "\ndt_" << lev << '=' << evolution.level(lev).dtPhys
               << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau << '\n';
        }
        std::ofstream boundaries(directory+"/boundaries.csv");
        boundaries << std::setprecision(std::numeric_limits<Real>::max_digits10)
                   << "name,boundary_id,object_id,level,center_x,center_y,center_z,radius\n";
        for (auto const& boundary : owner->resolvedBoundaries().internal)
            boundaries << boundary.name << ',' << boundary.boundaryId.value << ',' << boundary.objectId.value
                       << ',' << owner->internalBoundaryLevel(boundary.boundaryId) << ',' << boundary.centerPhys[0]
                       << ',' << boundary.centerPhys[1] << ',' << boundary.centerPhys[2] << ',' << boundary.radiusPhys << '\n';
        forces.open(directory+"/forces.csv");
        forces << std::setprecision(std::numeric_limits<Real>::max_digits10)
               << "kind,id,level,population_step,coarsest_step,begin_time,end_time,sample_time,Fx_lb,Fy_lb,Fz_lb\n";
        if (!id || !boundaries || !forces) amrex::Abort("cannot write tandem identity/force artifacts");
    }
    int completed = 0;
    for (int step : snapshots) {
        if (step > completed) {
            auto result = owner->evolve(evolution, step-completed, false, [&](auto const& view) {
                if (amrex::ParallelDescriptor::IOProcessor()) write_batch(forces, view.forceBatch);
            });
            if (!result.ok()) amrex::Abort(result.error);
        }
        owner->observe(true, [&](auto const& view) { output.synchronized(view); });
        completed = step;
        amrex::Print() << "tandem snapshot: L0 step " << step << '\n';
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        forces.flush();
        if (!forces) amrex::Abort("cannot finish tandem force artifacts");
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
