#ifndef LBM_TESTS_PERFORMANCE_GPUPERFORMANCECSV_H
#define LBM_TESTS_PERFORMANCE_GPUPERFORMANCECSV_H

#include "PerformanceBuildInfo.H"
#include "PerformanceControls.H"
#include "PerformanceCsv.H"

#include "lbm/profiling/TimestepProfiler.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>

namespace lbm::tests::performance {

struct GpuPerformanceContext {
    std::string amrexConfigSha256;
    std::string nvcc;
    std::string cudaToolkit;
    std::string cudaRuntime;
    std::string cudaDriver;
    std::string mpi;
    std::string mpiCudaBuildVersion;
    std::string mpiCudaSupportReported;
    std::string mpiCudaQuerySupport;
    std::string jobId;
    std::string node;
    std::string partition;
    std::string qos;
    int mpiRanks = 0;
    int migInstanceCount = 0;
    int parentGpuCount = 0;
    int boxesPerRank = 0;
    int ompThreads = 0;
    int amrexUseGpuAwareMpi = 0;
    std::string communicationMode;
    std::string rankDeviceMap;
    std::string slurmJobGpus;
    std::string cpuCpuset;
    std::string bindingPolicy;
    std::string migProfile;
    std::string parentGpuUuids;
    int numDevicesUsed = 0;
    int numDevicePartners = 0;
};

namespace gpu_csv_detail {

inline std::ofstream open_output(std::string const& filename)
{
    std::filesystem::path const path(filename);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    if (std::filesystem::exists(path)) {
        throw std::ios_base::failure(
            "refusing to overwrite " + path.string());
    }
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream) {
        throw std::ios_base::failure("cannot open " + path.string());
    }
    stream << std::setprecision(17);
    return stream;
}

inline amrex::Real percentage(
    amrex::Real value,
    amrex::Real parent) noexcept
{
    return parent > 0.0 ? 100.0 * value / parent : 0.0;
}

inline void write_common_header(std::ostream& stream)
{
    stream
        << "schema_version,measurement_mode,run_mode,workload,descriptor,"
           "layout,repeat,"
           "git_commit,amrex_source_commit,amrex_prefix,"
           "amrex_config_sha256,amrex_build_type,build_type,compiler,nvcc,"
           "cuda_toolkit,cuda_runtime,driver,mpi,mpi_cuda_build_version,"
           "mpi_cuda_support_reported,mpi_cuda_query_support,job_id,node,"
           "partition,qos,grid_x,"
           "grid_y,grid_z,global_valid_cells,max_grid_size,box_count,"
           "mpi_ranks,mig_instance_count,parent_gpu_count,boxes_per_rank,"
           "omp_threads,amrex_use_gpu_aware_mpi,communication_mode,"
           "rank_device_map,slurm_job_gpus,cpu_cpuset,binding_policy,"
           "mig_profile,parent_gpu_uuids,num_devices_used,"
           "num_device_partners,warmup_steps,sample_start,sample_end,nsteps";
}

template <class Descriptor>
void write_common_values(
    csv_detail::CsvRow& row,
    std::string_view measurementMode,
    PerformanceIdentity const& identity,
    PerformanceControls const& controls,
    std::array<int, 3> const& grid,
    std::int64_t globalValidCells,
    GpuPerformanceContext const& context)
{
    row.number(2);
    row.text(measurementMode);
    row.text(controls.mode);
    row.text(identity.workload);
    row.text(Descriptor::name());
    row.text(context.mpiRanks == 1 ? "1mig" : "4mig");
    row.number(controls.repeat);
    row.text(csv_detail::environment_value(
        "LBM_GPU_GIT_COMMIT", build_info::gitCommit));
    row.text(csv_detail::environment_value(
        "LBM_GPU_AMREX_SOURCE_COMMIT", build_info::amrexGitVersion));
    row.text(csv_detail::environment_value(
        "LBM_GPU_AMREX_PREFIX", build_info::amrexPrefix));
    row.text(context.amrexConfigSha256);
    row.text(build_info::amrexBuildType);
    row.text(build_info::buildType);
    row.text(build_info::compiler);
    row.text(context.nvcc);
    row.text(context.cudaToolkit);
    row.text(context.cudaRuntime);
    row.text(context.cudaDriver);
    row.text(context.mpi);
    row.text(context.mpiCudaBuildVersion);
    row.text(context.mpiCudaSupportReported);
    row.text(context.mpiCudaQuerySupport);
    row.text(context.jobId);
    row.text(context.node);
    row.text(context.partition);
    row.text(context.qos);
    row.number(grid[0]);
    row.number(grid[1]);
    row.number(grid[2]);
    row.number(globalValidCells);
    row.number(identity.maxGridSize);
    row.number(identity.boxCount);
    row.number(context.mpiRanks);
    row.number(context.migInstanceCount);
    row.number(context.parentGpuCount);
    row.number(context.boxesPerRank);
    row.number(context.ompThreads);
    row.number(context.amrexUseGpuAwareMpi);
    row.text(context.communicationMode);
    row.text(context.rankDeviceMap);
    row.text(context.slurmJobGpus);
    row.text(context.cpuCpuset);
    row.text(context.bindingPolicy);
    row.text(context.migProfile);
    row.text(context.parentGpuUuids);
    row.number(context.numDevicesUsed);
    row.number(context.numDevicePartners);
    row.number(controls.warmupSteps);
}

inline void require_finite_positive(
    amrex::Real value,
    char const* description)
{
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::ios_base::failure(
            std::string(description) + " must be finite and positive");
    }
}

inline void write_duration_values(
    csv_detail::CsvRow& row,
    profiling::TimestepProfileDurations const& durations)
{
    row.number(durations.collision);
    row.number(durations.preStream);
    row.number(durations.preStreamFillBoundary);
    row.number(durations.preStreamSetup);
    row.number(durations.preStreamPhysicalGhosts);
    row.number(durations.streaming);
    row.number(durations.domainPostStream);
    row.number(durations.internalSolidReconstruction);
    row.number(durations.postStream);
    row.number(durations.diagnostics);
    row.number(durations.total);
}

template <std::size_t Size>
std::string population_component_sums(
    std::array<amrex::Real, Size> const& sums)
{
    std::ostringstream stream;
    stream << std::setprecision(17);
    for (std::size_t population = 0; population < Size; ++population) {
        if (population != 0) {
            stream << ';';
        }
        stream << 'c' << population << '=' << sums[population];
    }
    return stream.str();
}

} // namespace gpu_csv_detail

template <class Descriptor>
void write_gpu_throughput_csv(
    PerformanceIdentity const& identity,
    PerformanceControls const& controls,
    std::array<int, 3> const& grid,
    std::int64_t globalValidCells,
    GpuPerformanceContext const& context,
    profiling::TimestepPerformanceRecord const& record,
    std::array<amrex::Real, Descriptor::q> const& finalPopulationComponentSums,
    amrex::Real finalPopulationL2)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    if (record.mode != profiling::TimestepProfilingMode::Throughput ||
        !record.formalMlups) {
        throw std::ios_base::failure(
            "throughput CSV requires a formal throughput record");
    }
    gpu_csv_detail::require_finite_positive(
        record.synchronizedWindow, "throughput window");
    const amrex::Real secondsPerStep =
        record.synchronizedWindow / static_cast<amrex::Real>(record.nSteps);
    const amrex::Real mlups =
        static_cast<amrex::Real>(globalValidCells) * record.nSteps /
        record.synchronizedWindow / 1.0e6;

    std::ofstream stream = gpu_csv_detail::open_output(controls.outputCsv);
    gpu_csv_detail::write_common_header(stream);
    stream << ",synchronized_throughput_window_s,total_s_per_step,mlups,"
              "formal_mlups,final_population_component_sums,"
              "final_population_l2\n";
    csv_detail::CsvRow row(stream);
    gpu_csv_detail::write_common_values<Descriptor>(
        row,
        profiling::to_string(record.mode),
        identity,
        controls,
        grid,
        globalValidCells,
        context);
    row.number(record.sampleStart);
    row.number(record.sampleEnd);
    row.number(record.nSteps);
    row.number(record.synchronizedWindow);
    row.number(secondsPerStep);
    row.number(mlups);
    row.text("true");
    row.text(gpu_csv_detail::population_component_sums(
        finalPopulationComponentSums));
    row.number(finalPopulationL2);
    row.finish();
    if (!stream) {
        throw std::ios_base::failure(
            "write failed for " + controls.outputCsv);
    }
}

template <class Descriptor>
void write_gpu_stage_breakdown_csv(
    PerformanceIdentity const& identity,
    PerformanceControls const& controls,
    std::array<int, 3> const& grid,
    std::int64_t globalValidCells,
    GpuPerformanceContext const& context,
    profiling::TimestepPerformanceRecord const& record,
    std::array<amrex::Real, Descriptor::q> const& finalPopulationComponentSums,
    amrex::Real finalPopulationL2)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    if (record.mode != profiling::TimestepProfilingMode::StageBreakdown ||
        record.formalMlups || record.criticalRank < 0) {
        throw std::ios_base::failure(
            "stage breakdown CSV requires a diagnostic critical-rank record");
    }
    gpu_csv_detail::require_finite_positive(
        record.criticalRankWindow.total, "critical-rank total");
    gpu_csv_detail::require_finite_positive(
        record.synchronizedWindow, "diagnostic window");
    const amrex::Real inverseSteps =
        1.0 / static_cast<amrex::Real>(record.nSteps);
    auto const& critical = record.criticalRankWindow;
    auto const& maxima = record.rankMaxWindow;
    const amrex::Real mainTotal =
        critical.collision + critical.preStream + critical.streaming +
        critical.postStream + critical.diagnostics;
    const amrex::Real residual = critical.total - mainTotal;
    const amrex::Real diagnosticMlups =
        static_cast<amrex::Real>(globalValidCells) * record.nSteps /
        record.synchronizedWindow / 1.0e6;

    std::ofstream stream = gpu_csv_detail::open_output(controls.outputCsv);
    gpu_csv_detail::write_common_header(stream);
    stream
        << ",critical_rank,critical_rank_collision_window_s,"
           "critical_rank_pre_stream_window_s,"
           "critical_rank_pre_stream_fill_boundary_window_s,"
           "critical_rank_pre_stream_setup_window_s,"
           "critical_rank_pre_stream_physical_ghosts_window_s,"
           "critical_rank_streaming_window_s,"
           "critical_rank_domain_post_stream_window_s,"
           "critical_rank_internal_solid_reconstruction_window_s,"
           "critical_rank_post_stream_window_s,"
           "critical_rank_diagnostics_window_s,"
           "critical_rank_total_window_s,"
           "critical_rank_collision_s_per_step,"
           "critical_rank_pre_stream_s_per_step,"
           "critical_rank_streaming_s_per_step,"
           "critical_rank_post_stream_s_per_step,"
           "critical_rank_diagnostics_s_per_step,"
           "critical_rank_total_s_per_step,"
           "independent_rank_max_collision_window_s,"
           "independent_rank_max_pre_stream_window_s,"
           "independent_rank_max_pre_stream_fill_boundary_window_s,"
           "independent_rank_max_pre_stream_setup_window_s,"
           "independent_rank_max_pre_stream_physical_ghosts_window_s,"
           "independent_rank_max_streaming_window_s,"
           "independent_rank_max_domain_post_stream_window_s,"
           "independent_rank_max_internal_solid_reconstruction_window_s,"
           "independent_rank_max_post_stream_window_s,"
           "independent_rank_max_diagnostics_window_s,"
           "independent_rank_max_total_window_s,collision_share_pct,"
           "pre_stream_share_pct,streaming_share_pct,post_stream_share_pct,"
           "diagnostics_share_pct,other_or_timing_overhead_share_pct,"
           "fill_boundary_share_of_pre_stream_pct,"
           "domain_share_of_post_stream_pct,"
           "internal_solid_share_of_post_stream_pct,"
           "diagnostic_synchronized_window_s,diagnostic_mlups,formal_mlups,"
           "final_population_component_sums,final_population_l2\n";
    csv_detail::CsvRow row(stream);
    gpu_csv_detail::write_common_values<Descriptor>(
        row,
        profiling::to_string(record.mode),
        identity,
        controls,
        grid,
        globalValidCells,
        context);
    row.number(record.sampleStart);
    row.number(record.sampleEnd);
    row.number(record.nSteps);
    row.number(record.criticalRank);
    gpu_csv_detail::write_duration_values(row, critical);
    row.number(critical.collision * inverseSteps);
    row.number(critical.preStream * inverseSteps);
    row.number(critical.streaming * inverseSteps);
    row.number(critical.postStream * inverseSteps);
    row.number(critical.diagnostics * inverseSteps);
    row.number(critical.total * inverseSteps);
    gpu_csv_detail::write_duration_values(row, maxima);
    row.number(gpu_csv_detail::percentage(
        critical.collision, critical.total));
    row.number(gpu_csv_detail::percentage(
        critical.preStream, critical.total));
    row.number(gpu_csv_detail::percentage(
        critical.streaming, critical.total));
    row.number(gpu_csv_detail::percentage(
        critical.postStream, critical.total));
    row.number(gpu_csv_detail::percentage(
        critical.diagnostics, critical.total));
    row.number(gpu_csv_detail::percentage(residual, critical.total));
    row.number(gpu_csv_detail::percentage(
        critical.preStreamFillBoundary, critical.preStream));
    row.number(gpu_csv_detail::percentage(
        critical.domainPostStream, critical.postStream));
    row.number(gpu_csv_detail::percentage(
        critical.internalSolidReconstruction, critical.postStream));
    row.number(record.synchronizedWindow);
    row.number(diagnosticMlups);
    row.text("false");
    row.text(gpu_csv_detail::population_component_sums(
        finalPopulationComponentSums));
    row.number(finalPopulationL2);
    row.finish();
    if (!stream) {
        throw std::ios_base::failure(
            "write failed for " + controls.outputCsv);
    }
}

} // namespace lbm::tests::performance

#endif
