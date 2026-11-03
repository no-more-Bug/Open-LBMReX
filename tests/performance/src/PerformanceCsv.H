#ifndef LBM_TESTS_PERFORMANCE_PERFORMANCECSV_H
#define LBM_TESTS_PERFORMANCE_PERFORMANCECSV_H

#include "PerformanceBuildInfo.H"
#include "PerformanceControls.H"

#include "lbm/profiling/TimestepProfiler.H"

#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace lbm::tests::performance {

namespace csv_detail {

inline std::string environment_value(
    char const* name,
    std::string fallback)
{
    char const* value = std::getenv(name);
    return value != nullptr && value[0] != '\0'
        ? std::string(value)
        : std::move(fallback);
}

class CsvRow {
public:
    explicit CsvRow(std::ostream& stream) : stream_(stream) {}

    template <class Number>
    void number(Number value)
    {
        separate();
        stream_ << value;
    }

    void text(std::string_view value)
    {
        separate();
        stream_.put('"');
        for (char character : value) {
            if (character == '"') {
                stream_.put('"');
            }
            stream_.put(character);
        }
        stream_.put('"');
    }

    void finish() { stream_.put('\n'); }

private:
    void separate()
    {
        if (!first_) {
            stream_.put(',');
        }
        first_ = false;
    }

    std::ostream& stream_;
    bool first_ = true;
};

} // namespace csv_detail

template <class Descriptor>
void write_performance_csv(
    PerformanceIdentity const& identity,
    PerformanceControls const& controls,
    std::array<int, 3> const& grid,
    std::int64_t globalValidCells,
    profiling::TimestepPerformanceRecord const& record)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }

    std::filesystem::path const path(controls.outputCsv);
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
    stream << std::setprecision(17)
           << "schema_version,workload,mode,descriptor,repeat,git_commit,"
              "amrex_source_commit,amrex_prefix,amrex_build_type,build_type,"
              "compiler,mpi,ucx_version,ucx_tls,node,partition,qos,"
              "grid_x,grid_y,grid_z,"
              "global_valid_cells,max_grid_size,box_count,mpi_ranks,"
              "omp_threads,cpuset,binding_policy,sample_start,sample_end,"
              "nsteps,timing_reduction,collision_window_s,"
              "pre_stream_window_s,pre_stream_fill_boundary_window_s,"
              "pre_stream_setup_window_s,"
              "pre_stream_physical_ghosts_window_s,streaming_window_s,"
              "domain_post_stream_window_s,"
              "internal_solid_reconstruction_window_s,"
              "post_stream_window_s,diagnostics_window_s,"
              "rank_max_local_total_window_s,synchronized_window_s,"
              "total_s_per_step,mlups\n";

    std::string const gitCommit = csv_detail::environment_value(
        "LBM_PERFORMANCE_GIT_COMMIT", build_info::gitCommit);
    std::string const amrexSource = csv_detail::environment_value(
        "LBM_PERFORMANCE_AMREX_SOURCE_COMMIT",
        build_info::amrexGitVersion);
    std::string const amrexPrefix = csv_detail::environment_value(
        "LBM_PERFORMANCE_AMREX_PREFIX", build_info::amrexPrefix);
    std::string const mpi = csv_detail::environment_value(
        "LBM_PERFORMANCE_MPI", build_info::mpi);
    std::string const ucxVersion = csv_detail::environment_value(
        "LBM_PERFORMANCE_UCX_VERSION", "unknown");
    std::string const ucxTls = csv_detail::environment_value(
        "LBM_PERFORMANCE_UCX_TLS", "unknown");
    std::string const node = csv_detail::environment_value(
        "SLURMD_NODENAME",
        csv_detail::environment_value("HOSTNAME", "unknown"));
    std::string const partition = csv_detail::environment_value(
        "SLURM_JOB_PARTITION", "local");
    std::string const qos = csv_detail::environment_value(
        "LBM_PERFORMANCE_QOS",
        csv_detail::environment_value("SLURM_JOB_QOS", "local"));
    std::string const cpuset = csv_detail::environment_value(
        "LBM_PERFORMANCE_CPUSET", "unknown");
    std::string const binding = csv_detail::environment_value(
        "LBM_PERFORMANCE_BINDING_POLICY", "unknown");
    const amrex::Real secondsPerStep =
        record.synchronizedWindow / static_cast<amrex::Real>(record.nSteps);
    const amrex::Real mlups =
        static_cast<amrex::Real>(globalValidCells) * record.nSteps /
        record.synchronizedWindow / 1.0e6;

    csv_detail::CsvRow row(stream);
    row.number(2);
    row.text(identity.workload);
    row.text(controls.mode);
    row.text(Descriptor::name());
    row.number(controls.repeat);
    row.text(gitCommit);
    row.text(amrexSource);
    row.text(amrexPrefix);
    row.text(build_info::amrexBuildType);
    row.text(build_info::buildType);
    row.text(build_info::compiler);
    row.text(mpi);
    row.text(ucxVersion);
    row.text(ucxTls);
    row.text(node);
    row.text(partition);
    row.text(qos);
    row.number(grid[0]);
    row.number(grid[1]);
    row.number(grid[2]);
    row.number(globalValidCells);
    row.number(identity.maxGridSize);
    row.number(identity.boxCount);
    row.number(amrex::ParallelDescriptor::NProcs());
    row.number(amrex::OpenMP::get_max_threads());
    row.text(cpuset);
    row.text(binding);
    row.number(record.sampleStart);
    row.number(record.sampleEnd);
    row.number(record.nSteps);
    row.text("independent_rank_max_with_synchronized_window");
    auto const& window = record.rankMaxWindow;
    row.number(window.collision);
    row.number(window.preStream);
    row.number(window.preStreamFillBoundary);
    row.number(window.preStreamSetup);
    row.number(window.preStreamPhysicalGhosts);
    row.number(window.streaming);
    row.number(window.domainPostStream);
    row.number(window.internalSolidReconstruction);
    row.number(window.postStream);
    row.number(window.diagnostics);
    row.number(window.total);
    row.number(record.synchronizedWindow);
    row.number(secondsPerStep);
    row.number(mlups);
    row.finish();
    if (!stream) {
        throw std::ios_base::failure("write failed for " + path.string());
    }
}

} // namespace lbm::tests::performance

#endif
