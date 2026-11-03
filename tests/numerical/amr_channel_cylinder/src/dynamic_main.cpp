#include "DynamicCylinder.H"
#include "Re100Observation.H"
#include "numerical/common/AmrSampleOutput.H"
#include "runtime/LbmAmrPreparation.H"
#include "lbm/amr/VorticityTagging.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <algorithm>
#include <cstdlib>

namespace {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

int run()
{
    using namespace dynamic_cylinder;
    static_assert(std::numeric_limits<Real>::digits == 53,"dynamic DFG requires FP64");
    auto const initializationStart = amrex::second();
    int maxBox = 64, limit = -1;
    std::string variant = "single", scheme = "yms", directory, source;
    amrex::ParmParse controls("case");
    controls.query("variant",variant); controls.query("scheme",scheme);
    amrex::ParmParse("mesh").query("max_grid_size",maxBox);
    amrex::ParmParse runControls("run");
    runControls.query("max_step",limit); runControls.get("source_id",source);
    amrex::ParmParse("output").get("directory",directory);
    if ((variant != "single" && variant != "tandem-a" && variant != "tandem-b") ||
        (scheme != "yms" && scheme != "bouzidi") || maxBox < 2 || maxBox%2 || limit < -1)
        amrex::Abort("invalid dynamic cylinder controls");
    auto spec = geometry(Descriptor::d,variant,maxBox);
    controls.query("grid_efficiency",spec.gridEfficiency);
    controls.query("regrid_interval",spec.regridInterval);
    amrex::Vector<int> buffers;
    if (controls.queryarr("tag_buffer_cells",buffers))
        for (int width : buffers)
            spec.tagBufferCells.emplace_back(width,width,Descriptor::d == 3 ? width : 0);
    lbm::amr::VorticityTagging tagging(.1,1,1/meanSpeed);
    amrex::Vector<Real> refine{20}, keep{10};
    controls.queryarr("theta_refine",refine); controls.queryarr("theta_keep",keep);
    if ((refine.size() != 1 && refine.size() != 2) || (keep.size() != 1 && keep.size() != 2))
        amrex::Abort("thresholds need one value or one per taggable level");
    tagging.thresholds.clear();
    for (int lev = 0; lev < int(std::max(refine.size(),keep.size())); ++lev)
        tagging.thresholds.push_back({refine[refine.size() == 1 ? 0 : lev],keep[keep.size() == 1 ? 0 : lev]});
    auto prepared = lbm::amr::prepare_dynamic_hierarchy<Descriptor>(
        spec,Initial{},tagging,boundaries(scheme,variant));
    if (!prepared.ok()) { for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto owner = prepared.take();
    Real const finestH = owner->geometry(2).CellSize(0);
    Real const diameterLb = Real(.1)/finestH, spanLb = Descriptor::d == 3 ? Real(.01)/finestH : Real(1);
    Real const denominator = Real(.5)*meanSpeed*meanSpeed*diameterLb*(Descriptor::d == 3 ? spanLb : 1);
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 100; intent.uRefLb = meanSpeed; intent.uRefPhys = 1; intent.rhoRefLb = 1;
    intent.lengthRefLb = diameterLb; intent.lengthRefPhys = .1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner,intent,lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) { for (auto const& error : preparation.errors) amrex::Print()<<error<<'\n'; return 2; }
    auto evolution = std::move(preparation).take();
    int const last = limit < 0 ? fullSteps : std::min(limit,fullSteps);
    auto const steps = sample_steps(last);
    auto const* profileSetting = std::getenv("LBM_TEST_CYLINDER_PROFILE");
    bool const profile = profileSetting && std::string(profileSetting) == "1";
    lbm::profiling::AmrTimestepProfiler profiler(
        profile ? lbm::profiling::TimestepProfilingMode::StageBreakdown
                : lbm::profiling::TimestepProfilingMode::Disabled,0,last,3);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory,meanSpeed);
    amr_re100::Re100Observation observation(directory,denominator);
    std::ofstream gridHistory;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=dynamic-cylinder-v1\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=0,0,1\ninitialization=uniform-equilibrium-then-Guo"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlayout=dynamic"
           << "\nvariant=" << variant << "\nscheme=" << (scheme == "yms" ? "YMS" : "Bouzidi")
           << "\nmax_level=2\nbase_z=" << spec.nCell[2] << "\nlength_z=0.01\nmax_grid_size=" << maxBox
           << "\nreynolds=100\nu0_lb=0.035\nu0_phys=1\nviscosity=0.001"
           << "\nrho_reference=1\nvelocity_reference=0.035\nforce_units=allowed-finest-lattice"
           << "\ndiameter_lb=" << diameterLb << "\nspan_lb=" << spanLb << "\nforce_denominator=" << denominator
           << "\ngap_fill_radius=" << spec.gapFillRadiusCells << "\ncoarse_nesting_margin=" << spec.coarseNestingMarginCells
           << "\ngrid_efficiency=" << spec.gridEfficiency << "\nregrid_interval=" << spec.regridInterval
           << "\nfull_steps=" << fullSteps << "\nrequested_steps=" << last << "\nframes=" << steps.size()
           << "\noverview_stride=4000\ndetail_begin=160000\ndetail_stride=400\nprofile=" << profile << '\n';
        for (int lev = 0; lev < 3; ++lev) {
            int const width = spec.tagBufferCells.empty() ? 2 :
                spec.tagBufferCells[spec.tagBufferCells.size() == 1 ? 0 : lev][0];
            id << "dt_" << lev << '=' << evolution.level(lev).dtPhys
               << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau
               << "\ntag_buffer_" << lev << '=' << width << '\n';
            if (lev < 2) {
                auto const pair = tagging.thresholds[tagging.thresholds.size() == 1 ? 0 : lev];
                id << "theta_refine_" << lev << '=' << pair.refine << "\ntheta_keep_" << lev << '=' << pair.keep << '\n';
            }
        }
        std::ofstream boundaryOutput(directory+"/boundaries.csv");
        boundaryOutput << std::setprecision(17) << "name,boundary_id,object_id,level,center_x,center_y,center_z,radius\n";
        for (auto const& item : owner->resolvedBoundaries().internal)
            boundaryOutput << item.name << ',' << item.boundaryId.value << ',' << item.objectId.value
                << ',' << owner->internalBoundaryLevel(item.boundaryId) << ',' << item.centerPhys[0]
                << ',' << item.centerPhys[1] << ',' << item.centerPhys[2] << ',' << item.radiusPhys << '\n';
        gridHistory.open(directory+"/grid-history.csv");
        gridHistory << std::setprecision(17) << "step,time,level,valid_cells,boxes\n";
        if (!id || !boundaryOutput || !gridHistory) amrex::Abort("cannot create dynamic cylinder artifacts");
    }
    amrex::Gpu::streamSynchronize();
    double initializationSeconds = amrex::second()-initializationStart;
    double advanceSeconds = 0, forceWriteSeconds = 0, fieldSeconds = 0;
    auto sample = [&] {
        auto const begin = amrex::second();
        owner->observe(true,[&](auto const& view) {
            output.layout(view,"layout-"+std::to_string(view.step)+".csv");
            output.synchronized(view);
            amrex::Print() << "dynamic_cylinder_sample step=" << view.step << " time=" << view.physicalTime
                           << " active_levels=" << view.levels.size() << '\n';
        });
        fieldSeconds += amrex::second()-begin;
    };
    sample();
    int current = 0;
    for (int target : steps) {
        if (target == 0) continue;
        auto const begin = amrex::second();
        auto result = owner->evolve(evolution,target-current,false,[&](auto const& view) {
            auto const writeBegin = amrex::second();
            observation.consume(view.forceBatch);
            if (amrex::ParallelDescriptor::IOProcessor())
                for (int lev = 0; lev < int(view.levels.size()); ++lev)
                    gridHistory << view.step << ',' << view.physicalTime << ',' << lev << ','
                        << view.levels[lev].boxes.numPts() << ',' << view.levels[lev].boxes.size() << '\n';
            forceWriteSeconds += amrex::second()-writeBegin;
        },nullptr,&profiler);
        amrex::Gpu::streamSynchronize();
        advanceSeconds += amrex::second()-begin;
        if (!result.ok()) { amrex::Print()<<result.error<<'\n'; return 2; }
        current = target;
        sample();
    }
    auto const flushBegin = amrex::second();
    observation.finish();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        gridHistory.flush();
        if (!gridHistory) amrex::Abort("cannot finish dynamic grid history");
    }
    double const flushSeconds = amrex::second()-flushBegin;
    profiler.finalize();
    auto const timing = profiler.performanceRecord();
    if (timing && profile && amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream csv(directory+"/timing.csv");
        csv << std::setprecision(17) << "owner,level,stage,critical_s,rank_max_s,calls\n";
        auto row = [&](char const* name, int level, char const* stage, auto const& critical, auto const& maximum) {
            csv << name << ',' << level << ',' << stage << ',' << critical.seconds << ',' << maximum.seconds
                << ',' << critical.calls << '\n';
        };
        char const* levelNames[]{"collision","communication","streaming","boundary_pre","boundary_post","management"};
        char const* pairNames[]{"coarse_to_fine","fine_to_coarse","explode","coalesce"};
        for (int lev = 0; lev < 3; ++lev) for (int stage = 0; stage < 6; ++stage)
            row("level",lev,levelNames[stage],timing->criticalRank.levels[lev][stage],timing->rankMax.levels[lev][stage]);
        for (int lev = 0; lev < 2; ++lev) for (int stage = 0; stage < 4; ++stage)
            row("pair",lev,pairNames[stage],timing->criticalRank.pairs[lev][stage],timing->rankMax.pairs[lev][stage]);
        row("hierarchy",-1,"management",timing->criticalRank.hierarchy,timing->rankMax.hierarchy);
        row("hierarchy",-1,"boundary_preparation",timing->criticalRank.boundaryPreparation,timing->rankMax.boundaryPreparation);
        if (!csv) amrex::Abort("cannot write dynamic cylinder timings");
    }
    double seconds[]{initializationSeconds,advanceSeconds-forceWriteSeconds,forceWriteSeconds+flushSeconds,fieldSeconds};
    amrex::ParallelDescriptor::ReduceRealMax(seconds,4);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream metrics(directory+"/run_metrics.csv");
        metrics << std::setprecision(17) << "initialization_s,advance_excluding_history_s,history_write_s,field_output_s\n"
                << seconds[0] << ',' << seconds[1] << ',' << seconds[2] << ',' << seconds[3] << '\n';
        if (!metrics) amrex::Abort("cannot write dynamic cylinder metrics");
    }
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    int status = 2;
    try { status = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return status;
}
