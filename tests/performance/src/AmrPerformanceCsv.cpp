#include "AmrPerformanceCsv.H"
#include "PerformanceCsv.H"
#include <sys/resource.h>
#include <sstream>
#include <stdexcept>

namespace lbm::tests::performance {
namespace {
using csv_detail::CsvRow;
using csv_detail::environment_value;

std::filesystem::path artifact(std::string const& output, std::string const& suffix)
{
    std::filesystem::path const path(output);
    return path.parent_path()/(path.stem().string()+"-"+suffix+".csv");
}

std::ofstream open_csv(std::filesystem::path const& path)
{
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    if (std::filesystem::exists(path)) throw std::ios_base::failure("refusing to overwrite "+path.string());
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream<<std::setprecision(17);
    return stream;
}

std::string cpuset()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("Cpus_allowed_list:", 0) == 0) {
            std::istringstream entry(line.substr(18));
            std::string value;
            entry>>value;
            return value;
        }
    }
    return "unknown";
}

char const* backend()
{
#ifdef AMREX_USE_CUDA
    return "cuda";
#else
    return "cpu";
#endif
}
} // namespace

std::uint64_t amr_device_used_bytes()
{
#ifdef AMREX_USE_CUDA
    std::size_t freeBytes = 0, totalBytes = 0;
    AMREX_GPU_SAFE_CALL(cudaMemGetInfo(&freeBytes, &totalBytes));
    return totalBytes-freeBytes;
#else
    return 0;
#endif
}

void write_amr_resource_csv(std::string const& outputCsv, std::uint64_t initializedUsedBytes)
{
    int const rank = amrex::ParallelDescriptor::MyProc();
    auto stream = open_csv(artifact(outputCsv, "rank-"+std::to_string(rank)));
    stream<<"rank,cpuset,omp_threads,device_visible,device_count,device_name,device_total_bytes,"
              "after_init_used_bytes,after_window_used_bytes,host_process_peak_rss_kib,device_peak_status\n";
    int devices = 0;
    std::string name = "not_applicable";
    std::uint64_t total = 0;
#ifdef AMREX_USE_CUDA
    int device = 0;
    cudaDeviceProp properties{};
    AMREX_GPU_SAFE_CALL(cudaGetDeviceCount(&devices));
    AMREX_GPU_SAFE_CALL(cudaGetDevice(&device));
    AMREX_GPU_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
    name = properties.name;
    total = properties.totalGlobalMem;
#endif
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("cannot read process peak RSS");
    CsvRow row(stream);
    row.number(rank); row.text(cpuset()); row.number(amrex::OpenMP::get_max_threads());
    row.text(environment_value("CUDA_VISIBLE_DEVICES", "not_applicable")); row.number(devices);
    row.text(name); row.number(total); row.number(initializedUsedBytes); row.number(amr_device_used_bytes());
    row.number(usage.ru_maxrss); row.text("not_measured"); row.finish();
}

void write_amr_performance_csv(
    PerformanceControls const& controls, std::string const& layout, char const* descriptor,
    std::vector<AmrPerformanceLevel> const& levels, profiling::AmrPerformanceRecord const& record,
    amrex::Real initializationSeconds)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    auto const& window = record.window;
    bool const breakdown = window.mode == profiling::TimestepProfilingMode::StageBreakdown;
    std::int64_t updates = 0;
    auto levelOutput = open_csv(artifact(controls.outputCsv, "levels"));
    auto boxOutput = open_csv(artifact(controls.outputCsv, "boxes"));
    levelOutput<<"level,nx,ny,nz,valid_cells,boxes,dt,tau,measured_steps,valid_updates\n";
    boxOutput<<"level,box,rank,lo_x,lo_y,lo_z,hi_x,hi_y,hi_z\n";
    for (int lev=0; lev<int(levels.size()); ++lev) {
        auto const& level = levels[lev];
        auto const cells = static_cast<std::int64_t>(level.boxes.numPts());
        auto const work = cells*record.levelSteps[lev];
        updates += work;
        CsvRow row(levelOutput);
        row.number(lev);
        for (int length : level.grid) row.number(length);
        row.number(cells); row.number(level.boxes.size()); row.number(level.dt); row.number(level.tau);
        row.number(record.levelSteps[lev]); row.number(work); row.finish();
        for (int box=0; box<level.boxes.size(); ++box) {
            CsvRow entry(boxOutput);
            entry.number(lev); entry.number(box); entry.number(level.mapping[box]);
            for (int d=0; d<3; ++d) entry.number(level.boxes[box].smallEnd(d));
            for (int d=0; d<3; ++d) entry.number(level.boxes[box].bigEnd(d));
            entry.finish();
        }
    }
    auto output = open_csv(controls.outputCsv);
    output<<"schema,workload,layout,descriptor,backend,run_mode,measurement_mode,repeat,git_commit,"
            "amrex_source_commit,amrex_prefix,build_type,compiler,mpi,cpu_model,mpi_ranks,omp_threads,node,partition,qos,job_id,"
            "initialization_s,warmup_steps,nsteps,sample_start,sample_end,physical_begin,physical_end,"
            "window_s,critical_rank,critical_local_s,valid_updates,mlups,finite_final_state\n";
    CsvRow row(output);
    row.text("amr-performance-v1"); row.text(controls.workload); row.text(layout); row.text(descriptor);
    row.text(backend()); row.text(controls.mode); row.text(controls.measurementMode); row.number(controls.repeat);
    row.text(build_info::gitCommit);
    row.text(environment_value("LBM_PERFORMANCE_AMREX_SOURCE_COMMIT", build_info::amrexGitVersion));
    row.text(build_info::amrexPrefix); row.text(build_info::buildType); row.text(build_info::compiler);
    row.text(environment_value("LBM_PERFORMANCE_MPI", build_info::mpi));
    row.text(environment_value("LBM_PERFORMANCE_CPU_MODEL", "unknown"));
    row.number(amrex::ParallelDescriptor::NProcs()); row.number(amrex::OpenMP::get_max_threads());
    row.text(environment_value("SLURMD_NODENAME", environment_value("HOSTNAME", "unknown")));
    row.text(environment_value("SLURM_JOB_PARTITION", "local"));
    row.text(environment_value("LBM_PERFORMANCE_QOS", "unknown"));
    row.text(environment_value("SLURM_JOB_ID", "unknown"));
    row.number(initializationSeconds); row.number(controls.warmupSteps); row.number(window.nSteps);
    row.number(window.sampleStart); row.number(window.sampleEnd);
    row.number(controls.warmupSteps*levels[0].dt); row.number(window.sampleEnd*levels[0].dt);
    row.number(window.synchronizedWindow); row.number(window.criticalRank);
    if (breakdown) row.number(window.criticalRankWindow.total); else row.text("");
    row.number(updates);
    if (breakdown) row.text(""); else row.number(updates/(1.e6*window.synchronizedWindow));
    row.number(1); row.finish();
    if (!breakdown) return;
    auto stages = open_csv(artifact(controls.outputCsv, "stages"));
    stages<<"owner,level,stage,critical_s,calls,rank_max_s,rank_max_calls,share_percent,mean_call_s,mean_level_step_s\n";
    auto writeStage = [&](char const* owner, int level, char const* stage,
                          profiling::AmrStageTiming const& critical, profiling::AmrStageTiming const& maximum,
                          int levelSteps) {
        CsvRow entry(stages);
        entry.text(owner); entry.number(level); entry.text(stage); entry.number(critical.seconds);
        entry.number(critical.calls); entry.number(maximum.seconds); entry.number(maximum.calls);
        entry.number(100*critical.seconds/window.criticalRankWindow.total);
        if (critical.calls) entry.number(critical.seconds/critical.calls); else entry.text("");
        if (levelSteps) entry.number(critical.seconds/levelSteps); else entry.text("");
        entry.finish();
    };
    constexpr std::array<char const*,6> levelNames{
        "collision","communication","streaming","boundary_pre","boundary_post","management"};
    constexpr std::array<char const*,4> pairNames{
        "coarse_to_fine_communication","fine_to_coarse_communication","explode","coalesce"};
    for (int lev=0; lev<int(record.criticalRank.levels.size()); ++lev)
        for (int stage=0; stage<6; ++stage)
            writeStage("level", lev, levelNames[stage], record.criticalRank.levels[lev][stage],
                       record.rankMax.levels[lev][stage], record.levelSteps[lev]);
    for (int lev=0; lev<int(record.criticalRank.pairs.size()); ++lev)
        for (int stage=0; stage<4; ++stage)
            writeStage("pair", lev, pairNames[stage], record.criticalRank.pairs[lev][stage],
                       record.rankMax.pairs[lev][stage], 0);
    writeStage("hierarchy", -1, "management", record.criticalRank.hierarchy, record.rankMax.hierarchy, 0);
}
} // namespace lbm::tests::performance
