#include "DynamicDoubleShear.H"
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

#ifdef LBM_DSL_FIXED_BAND
constexpr bool fixedBand = true;
constexpr char schema[] = "fixed-band-double-shear-v2";
#else
constexpr bool fixedBand = false;
constexpr char schema[] = "dynamic-double-shear-v1";
#endif

int run()
{
    using namespace double_shear;
    static_assert(std::numeric_limits<Real>::digits == 53,"DSL requires FP64");
    int maxLevel = fixedBand ? 2 : 1, maxBox = 32, limit = -1;
    std::string directory, source;
    amrex::ParmParse controls("case");
    controls.query("max_level",maxLevel);
    amrex::ParmParse("mesh").query("max_grid_size",maxBox);
    amrex::ParmParse runControls("run");
    runControls.query("max_step",limit); runControls.get("source_id",source);
    amrex::ParmParse("output").get("directory",directory);
    if (maxLevel < 1 || maxLevel > 2 || (fixedBand && maxLevel != 2) || maxBox < 8 || maxBox%2 || limit < -1)
        amrex::Abort("invalid dynamic DSL grid/run controls");
    auto spec = fixedBand ? fixed_band_geometry(Descriptor::d,maxBox)
                          : dynamic_geometry(Descriptor::d,maxLevel,maxBox);
    controls.query("grid_efficiency",spec.gridEfficiency);
    controls.query("regrid_interval",spec.regridInterval);
    amrex::Vector<int> buffers;
    if (controls.queryarr("tag_buffer_cells",buffers))
        for (int width : buffers)
            spec.tagBufferCells.emplace_back(width,width,Descriptor::d == 3 ? width : 0);
    lbm::amr::VorticityTagging tagging(1,1,1/speed);
    amrex::Vector<Real> refine{tagging.thresholds[0].refine}, keep{tagging.thresholds[0].keep};
    controls.queryarr("theta_refine",refine); controls.queryarr("theta_keep",keep);
    if ((refine.size() != 1 && refine.size() != static_cast<std::size_t>(maxLevel)) ||
        (keep.size() != 1 && keep.size() != static_cast<std::size_t>(maxLevel)))
        amrex::Abort("threshold controls need one value or one per taggable level");
    tagging.thresholds.clear();
    for (int lev = 0; lev < static_cast<int>(std::max(refine.size(),keep.size())); ++lev)
        tagging.thresholds.push_back({refine[refine.size() == 1 ? 0 : lev],keep[keep.size() == 1 ? 0 : lev]});
    auto prepared = lbm::amr::prepare_dynamic_hierarchy<Descriptor>(spec,Initial{},tagging);
    if (!prepared.ok()) {
        for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
        return 2;
    }
    auto owner = prepared.take();
    lbm::config::LbmConfigIntent intent;
    intent.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = reynolds; intent.uRefLb = speed; intent.uRefPhys = 1; intent.rhoRefLb = 1;
    intent.lengthRefLb = owner->geometry(maxLevel).Domain().length(0);
    intent.lengthRefPhys = 1;
    auto preparation = lbm::runtime::prepare_amr_evolution(*owner,intent,lbm::BGKDynamicsSelection{});
    if (!preparation.ok()) {
        for (auto const& error : preparation.errors) amrex::Print() << error << '\n';
        return 2;
    }
    auto evolution = std::move(preparation).take();
    int const n = spec.nCell[0], fullSteps = 20*n;
    int const last = limit < 0 ? fullSteps : std::min(limit,fullSteps);
    // Tests-owned stage timings exclude frame output; the wall window includes sampling gaps.
    auto const* profileSetting = std::getenv("LBM_TEST_DSL_PROFILE");
    bool const profile = profileSetting && std::string(profileSetting) == "1";
    lbm::profiling::AmrTimestepProfiler profiler(
        profile ? lbm::profiling::TimestepProfilingMode::StageBreakdown
                : lbm::profiling::TimestepProfilingMode::Disabled, 0, last, maxLevel+1);
    lbm::tests::numerical::AmrSampleOutput<Descriptor> output(directory,speed);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream id(directory+"/identity.txt");
        id << std::setprecision(17) << "schema=" << schema << "\ndescriptor=" << Descriptor::name()
           << "\ndimensions=" << Descriptor::d << "\nsource_id=" << source << "\nbackend=";
#ifdef AMREX_USE_CUDA
        id << "cuda";
#else
        id << "cpu";
#endif
        id << "\ncollision=BGK\nperiodic=1,1,1\ninitialization=cell-center-equilibrium"
           << "\nranks=" << amrex::ParallelDescriptor::NProcs() << "\nreal_digits=53\nlayout=dynamic"
           << "\nbase_cells=" << n << "\nbase_z=" << spec.nCell[2] << "\nlength_z=" << spec.probHi[2]
           << "\nmax_level=" << maxLevel << "\nmax_grid_size=" << maxBox
           << "\nreynolds=" << reynolds << "\nu0_lb=" << speed
           << "\nu0_phys=1\nviscosity=0.0001\nkappa=80\ndelta=0.05\nrho_reference=1"
           << "\nvelocity_reference=" << speed << "\ngap_fill_radius=" << spec.gapFillRadiusCells
           << "\ncoarse_nesting_margin=" << spec.coarseNestingMarginCells << "\ngrid_efficiency=" << spec.gridEfficiency
           << "\nregrid_interval=" << spec.regridInterval << "\nfull_steps=" << fullSteps
           << "\nrequested_steps=" << last << "\nprofile=" << profile << "\nframe_stride=" << n/2 << '\n';
        if (fixedBand) {
            id << "case_variant=minimum-band\nfixed_band_min_level=1\nfixed_band_max_level=2"
               << "\nfixed_band_y_lo=0.65\nfixed_band_y_hi=0.85"
               << "\nfixed_band_parent_jlo=83\nfixed_band_parent_jhi=109"
               << "\nfixed_band_aligned_y_lo=" << Real(83)/128
               << "\nfixed_band_aligned_y_hi=" << Real(109)/128 << '\n';
        }
        for (int lev = 0; lev <= maxLevel; ++lev) {
            id << "dt_" << lev << '=' << evolution.level(lev).dtPhys
               << "\ntau_" << lev << '=' << evolution.level(lev).lbm.tau << '\n';
            int const width = spec.tagBufferCells.empty() ? 2 :
                spec.tagBufferCells[spec.tagBufferCells.size() == 1 ? 0 : lev][0];
            id << "tag_buffer_" << lev << '=' << width << '\n';
            if (lev < maxLevel) {
                auto const pair = tagging.thresholds[tagging.thresholds.size() == 1 ? 0 : lev];
                id << "theta_refine_" << lev << '=' << pair.refine
                   << "\ntheta_keep_" << lev << '=' << pair.keep << '\n';
            }
        }
        if (!id) amrex::Abort("cannot write dynamic DSL identity");
    }
    auto sample = [&](auto const& view) {
        output.layout(view,"layout-"+std::to_string(view.step)+".csv");
        output.synchronized(view);
        amrex::Print() << "dynamic_dsl_sample step=" << view.step << " time=" << view.physicalTime
                       << " active_levels=" << view.levels.size() << '\n';
    };
    owner->observe(true,sample);
    std::ofstream gridHistory;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        gridHistory.open(directory+"/grid-history.csv");
        gridHistory << std::setprecision(17) << "step,time,level,valid_cells,boxes\n";
        if (!gridHistory) amrex::Abort("cannot write grid history");
    }
    for (int current = 0; current < last;) {
        int const target = std::min(current+n/2,last);
        auto result = owner->evolve(evolution,target-current,false,[&](auto const& view) {
            if (!amrex::ParallelDescriptor::IOProcessor()) return;
            for (int lev = 0; lev < static_cast<int>(view.levels.size()); ++lev)
                gridHistory << view.step << ',' << view.physicalTime << ',' << lev << ','
                            << view.levels[lev].boxes.numPts() << ',' << view.levels[lev].boxes.size() << '\n';
        },nullptr,&profiler);
        if (!result.ok()) { amrex::Print() << result.error << '\n'; return 2; }
        current = target;
        owner->observe(true,sample);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        gridHistory.flush();
        if (!gridHistory) amrex::Abort("cannot finish grid history");
    }
    profiler.finalize();
    auto const timing = profiler.performanceRecord();
    if (timing && profile && amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream csv(directory+"/timing.csv");
        csv << std::setprecision(17) << "owner,level,stage,critical_s,rank_max_s,calls\n";
        auto row = [&](char const* ownerName, int level, char const* stage,
                       auto const& critical, auto const& maximum) {
            csv << ownerName << ',' << level << ',' << stage << ',' << critical.seconds
                << ',' << maximum.seconds << ',' << critical.calls << '\n';
        };
        char const* levelNames[]{"collision","communication","streaming","boundary_pre","boundary_post","management"};
        char const* pairNames[]{"coarse_to_fine","fine_to_coarse","explode","coalesce"};
        for (int lev = 0; lev <= maxLevel; ++lev)
            for (int stage = 0; stage < 6; ++stage)
                row("level",lev,levelNames[stage],timing->criticalRank.levels[lev][stage],
                    timing->rankMax.levels[lev][stage]);
        for (int lev = 0; lev < maxLevel; ++lev)
            for (int stage = 0; stage < 4; ++stage)
                row("pair",lev,pairNames[stage],timing->criticalRank.pairs[lev][stage],
                    timing->rankMax.pairs[lev][stage]);
        row("hierarchy",-1,"management",timing->criticalRank.hierarchy,timing->rankMax.hierarchy);
        row("hierarchy",-1,"boundary_preparation",timing->criticalRank.boundaryPreparation,
            timing->rankMax.boundaryPreparation);
        csv.flush();
        if (!csv) amrex::Abort("cannot write DSL stage timings");
        amrex::Print() << "DSL_TIMING_WINDOW " << timing->window.synchronizedWindow
                       << " step_sum=" << timing->window.criticalRankWindow.total
                       << " critical_rank=" << timing->window.criticalRank << '\n';
    }
    return 0;
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    int result = 2;
    try { result = run(); }
    catch (std::exception const& error) { amrex::Abort(error.what()); }
    amrex::Finalize();
    return result;
}
