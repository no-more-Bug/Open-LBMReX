#ifndef LBM_TESTS_PERFORMANCE_GPUPERFORMANCERUNSESSION_H
#define LBM_TESTS_PERFORMANCE_GPUPERFORMANCERUNSESSION_H

#include "GpuPerformanceCsv.H"
#include "PerformanceControls.H"
#include "PerformanceRunSession.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_GpuDevice.H>
#include <AMReX_MultiFab.H>
#include <AMReX_OpenMP.H>
#include <AMReX_ParallelDescriptor.H>
#include <mpi-ext.h>

#ifdef AMREX_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lbm::tests::performance {

namespace gpu_session_detail {

inline std::string environment_value(char const* name)
{
    char const* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

inline bool parse_positive_int(std::string const& text, int& value)
{
    if (text.empty()) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        value = std::stoi(text, &consumed);
        return consumed == text.size() && value > 0;
    } catch (...) {
        return false;
    }
}

inline std::string sanitized(std::string text)
{
    for (char& value : text) {
        if (value == ';' || value == '|' || value == '\n' || value == '\r') {
            value = '_';
        }
    }
    return text;
}

inline std::string process_cpuset()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        constexpr std::string_view prefix = "Cpus_allowed_list:";
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(prefix.size());
            value.erase(
                value.begin(),
                std::find_if(
                    value.begin(), value.end(),
                    [](unsigned char character) {
                        return !std::isspace(character);
                    }));
            return value;
        }
    }
    return {};
}

template <std::size_t Width>
std::vector<std::string> gather_fixed_strings(
    std::string const& local,
    bool& fits)
{
    std::array<char, Width> send{};
    fits = local.size() < Width;
    std::copy_n(local.data(), std::min(local.size(), Width - 1), send.data());
    int const ranks = amrex::ParallelDescriptor::NProcs();
    std::vector<char> receive(Width * static_cast<std::size_t>(ranks));
    amrex::ParallelDescriptor::Gather(
        send.data(), send.size(), receive.data(), send.size(),
        amrex::ParallelDescriptor::IOProcessorNumber());
    std::vector<std::string> result;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        result.reserve(static_cast<std::size_t>(ranks));
        for (int rank = 0; rank < ranks; ++rank) {
            result.emplace_back(
                receive.data() + static_cast<std::size_t>(rank) * Width);
        }
    }
    return result;
}

inline std::string join_rank_values(
    std::vector<std::string> const& values)
{
    std::ostringstream stream;
    for (std::size_t rank = 0; rank < values.size(); ++rank) {
        if (rank != 0) {
            stream << ';';
        }
        stream << 'r' << rank << '=' << values[rank];
    }
    return stream.str();
}

inline int unique_delimited_values(std::string const& text)
{
    std::set<std::string> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        std::size_t const end = text.find(';', begin);
        std::string const value = text.substr(begin, end - begin);
        if (!value.empty()) {
            values.insert(value);
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return static_cast<int>(values.size());
}

inline std::vector<std::string> comma_delimited_values(
    std::string const& text)
{
    std::vector<std::string> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        std::size_t const end = text.find(',', begin);
        values.push_back(text.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return values;
}

inline std::string normalized_parent_uuid(std::string text)
{
    if (text.rfind("GPU-", 0) == 0) {
        text.erase(0, 4);
    }
    text.erase(std::remove(text.begin(), text.end(), '-'), text.end());
    std::transform(
        text.begin(), text.end(), text.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (text.size() != 32 ||
        !std::all_of(
            text.begin(), text.end(),
            [](unsigned char character) {
                return std::isxdigit(character) != 0;
            })) {
        return {};
    }
    return text;
}

#ifdef AMREX_USE_CUDA
inline std::string uuid_string(cudaUUID_t const& uuid)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char byte : uuid.bytes) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

inline std::string cuda_version_string(int version)
{
    std::ostringstream stream;
    stream << version / 1000 << '.' << (version % 1000) / 10;
    return stream.str();
}
#endif

inline std::optional<profiling::TimestepProfilingMode>
profiling_mode(std::string const& name) noexcept
{
    if (name == "throughput") {
        return profiling::TimestepProfilingMode::Throughput;
    }
    if (name == "stage_breakdown") {
        return profiling::TimestepProfilingMode::StageBreakdown;
    }
    return std::nullopt;
}

} // namespace gpu_session_detail

template <class Descriptor>
class GpuPerformanceRunSession {
public:
    GpuPerformanceRunSession() = default;

    GpuPerformanceRunSession(
        PerformanceIdentity identity,
        PerformanceControls controls)
        : identity_(std::move(identity)), controls_(std::move(controls))
    {}

    runtime::SingleLevelRunSessionResult start(
        runtime::SingleLevelRunContext<Descriptor> const& context)
    {
#ifndef AMREX_USE_CUDA
        return {
            runtime::SingleLevelRunStatus::OutputFailure,
            "GPU performance session requires an AMReX CUDA build"};
#else
        for (int direction = 0; direction < 3; ++direction) {
            grid_[direction] = context.geometry.Domain().length(direction);
        }
        if (identity_.boxCount != 16 || grid_ != identity_.grid ||
            context.boxArray.size() != identity_.boxCount) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance requires its fixed grid and 16-box identity"};
        }

        int const rankCount = amrex::ParallelDescriptor::NProcs();
        if (rankCount != 1 && rankCount != 4) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance requires exactly one or four MPI ranks"};
        }
        int const expectedBoxesPerRank = identity_.boxCount / rankCount;
        std::vector<int> boxesPerRank(static_cast<std::size_t>(rankCount), 0);
        bool balanced =
            identity_.boxCount % rankCount == 0 &&
            context.distributionMapping.ProcessorMap().size() ==
                context.boxArray.size();
        for (int owner : context.distributionMapping.ProcessorMap()) {
            if (owner < 0 || owner >= rankCount) {
                balanced = false;
            } else {
                ++boxesPerRank[static_cast<std::size_t>(owner)];
            }
        }
        for (int count : boxesPerRank) {
            balanced = balanced && count == expectedBoxesPerRank;
        }
        if (!balanced) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance requires 16 or 4 boxes per rank"};
        }

        runtime::SingleLevelRunSessionResult identityResult =
            collectGpuIdentity(rankCount, expectedBoxesPerRank);
        if (!identityResult.ok()) {
            return identityResult;
        }
        globalValidCells_ = 1;
        for (int length : grid_) {
            globalValidCells_ *= static_cast<std::int64_t>(length);
        }
        started_ = true;
        return {};
#endif
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeFinal = true;
        return plan;
    }

    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const&)
    {
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& published)
    {
        amrex::MultiFab const& populations =
            published.populations().multifab();
        if (!populations.is_finite(0, Descriptor::q, 0)) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance final populations contain NaN or Inf"};
        }
        for (int population = 0; population < Descriptor::q; ++population) {
            finalPopulationComponentSums_[population] =
                populations.sum(population);
        }
        finalPopulationL2_ = populations.norm2(0, Descriptor::q);
        if (!std::all_of(
                finalPopulationComponentSums_.begin(),
                finalPopulationComponentSums_.end(),
                [](amrex::Real value) { return std::isfinite(value); }) ||
            !std::isfinite(finalPopulationL2_) || finalPopulationL2_ <= 0.0) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance final population fingerprint is invalid"};
        }
        finalStateChecked_ = true;
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        if (!started_ || !finalStateChecked_ || !record.has_value()) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance run completed without final-state or "
                "profiling validation"};
        }
        auto const expectedMode =
            gpu_session_detail::profiling_mode(controls_.measurementMode);
        if (!expectedMode.has_value() || record->mode != *expectedMode ||
            record->nSteps != controls_.measurementSteps ||
            record->sampleStart != controls_.warmupSteps + 1 ||
            record->sampleEnd != controls_.maxStep ||
            record->sampleEnd - record->sampleStart + 1 != record->nSteps ||
            !std::isfinite(record->synchronizedWindow) ||
            record->synchronizedWindow <= 0.0) {
            return {
                runtime::SingleLevelRunStatus::NumericalFailure,
                "GPU performance profiling window is incomplete or invalid"};
        }
        try {
            if (*expectedMode == profiling::TimestepProfilingMode::Throughput) {
                write_gpu_throughput_csv<Descriptor>(
                    identity_, controls_, grid_, globalValidCells_,
                    gpuContext_, *record, finalPopulationComponentSums_,
                    finalPopulationL2_);
            } else {
                write_gpu_stage_breakdown_csv<Descriptor>(
                    identity_, controls_, grid_, globalValidCells_,
                    gpuContext_, *record, finalPopulationComponentSums_,
                    finalPopulationL2_);
            }
        } catch (std::exception const& exception) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                std::string("Failed to write GPU performance CSV: ") +
                    exception.what()};
        }
        return {};
    }

private:
#ifdef AMREX_USE_CUDA
    runtime::SingleLevelRunSessionResult collectGpuIdentity(
        int rankCount,
        int boxesPerRank)
    {
        int status = 0;
        auto requireEnvironment = [&](char const* name) {
            std::string value = gpu_session_detail::environment_value(name);
            if (value.empty()) {
                amrex::AllPrint()
                    << "gpu_performance_identity_status=environment-failure "
                    << "global_rank=" << amrex::ParallelDescriptor::MyProc()
                    << " missing=" << name << '\n';
                status = std::max(status, 2);
            }
            return value;
        };
        auto checkCuda = [&](cudaError_t error, char const* operation) {
            if (error != cudaSuccess) {
                amrex::AllPrint()
                    << "gpu_performance_identity_status=environment-failure "
                    << "global_rank=" << amrex::ParallelDescriptor::MyProc()
                    << " operation=" << operation
                    << " cuda_error=" << cudaGetErrorString(error) << '\n';
                status = std::max(status, 2);
            }
        };

        std::string const localRankText = requireEnvironment("SLURM_LOCALID");
        std::string const cudaVisible =
            requireEnvironment("CUDA_VISIBLE_DEVICES");
        std::string const slurmStepGpus =
            gpu_session_detail::environment_value("SLURM_STEP_GPUS");
        std::string const allocationCudaVisible =
            requireEnvironment("LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES");
        std::string const deviceAssignmentSource =
            requireEnvironment("LBM_GPU_DEVICE_ASSIGNMENT_SOURCE");
        std::string const parentGpuUuids =
            requireEnvironment("LBM_GPU_PARENT_UUIDS");
        int parentGpuCount = 0;
        std::string const parentCountText =
            requireEnvironment("LBM_GPU_PARENT_GPU_COUNT");
        if (!gpu_session_detail::parse_positive_int(
                parentCountText, parentGpuCount) ||
            parentGpuCount != 1 ||
            gpu_session_detail::unique_delimited_values(parentGpuUuids) != 1) {
            status = std::max(status, 2);
        }
        std::string const normalizedParentUuid =
            gpu_session_detail::normalized_parent_uuid(parentGpuUuids);
        std::string const cpuset = gpu_session_detail::process_cpuset();
        if (cpuset.empty()) {
            status = std::max(status, 2);
        }

        int visibleDevices = 0;
        int currentDevice = -1;
        int runtimeVersion = 0;
        int driverVersion = 0;
        cudaDeviceProp properties{};
        std::array<char, 32> pciBus{};
        checkCuda(cudaGetDeviceCount(&visibleDevices), "cudaGetDeviceCount");
        checkCuda(cudaGetDevice(&currentDevice), "cudaGetDevice");
        if (currentDevice >= 0) {
            checkCuda(
                cudaGetDeviceProperties(&properties, currentDevice),
                "cudaGetDeviceProperties");
            checkCuda(
                cudaDeviceGetPCIBusId(
                    pciBus.data(), static_cast<int>(pciBus.size()),
                    currentDevice),
                "cudaDeviceGetPCIBusId");
        }
        checkCuda(cudaRuntimeGetVersion(&runtimeVersion), "cudaRuntimeGetVersion");
        checkCuda(cudaDriverGetVersion(&driverVersion), "cudaDriverGetVersion");

        int localRank = -1;
        try {
            std::size_t consumed = 0;
            localRank = std::stoi(localRankText, &consumed);
            if (consumed != localRankText.size()) {
                localRank = -1;
            }
        } catch (...) {
            localRank = -1;
        }
        int const globalRank = amrex::ParallelDescriptor::MyProc();
        int const numDevicesUsed = amrex::Gpu::Device::numDevicesUsed();
        int const numDevicePartners = amrex::Gpu::Device::numDevicePartners();
        int const amrexDevice = amrex::Gpu::Device::deviceId();
        int const ompThreads = amrex::OpenMP::get_max_threads();
        int mpiCudaQueryMin = MPIX_Query_cuda_support();
        int mpiCudaQueryMax = mpiCudaQueryMin;
        amrex::ParallelDescriptor::ReduceIntMin(mpiCudaQueryMin);
        amrex::ParallelDescriptor::ReduceIntMax(mpiCudaQueryMax);
        if (mpiCudaQueryMin != mpiCudaQueryMax ||
            (mpiCudaQueryMin != 0 && mpiCudaQueryMin != 1)) {
            status = std::max(status, 2);
        }
        std::vector<std::string> const allocatedDevices =
            gpu_session_detail::comma_delimited_values(
                allocationCudaVisible);
        std::string const cudaParentUuid =
            status == 0 ? gpu_session_detail::uuid_string(properties.uuid) : "invalid";
        bool const localMappingValid =
            status == 0 && localRank == globalRank && visibleDevices == 1 &&
            currentDevice == 0 && amrexDevice == 0 &&
            numDevicesUsed == 1 && numDevicePartners == rankCount &&
            ompThreads == 1 && cudaVisible.rfind("MIG-", 0) == 0 &&
            cudaVisible.find(',') == std::string::npos &&
            allocatedDevices.size() == static_cast<std::size_t>(rankCount) &&
            localRank >= 0 && localRank < rankCount &&
            std::find(
                allocatedDevices.begin(), allocatedDevices.end(),
                cudaVisible) != allocatedDevices.end() &&
            deviceAssignmentSource == "slurm-task-local-cvd" &&
            normalizedParentUuid == cudaParentUuid;
        if (!localMappingValid && status == 0) {
            status = 1;
        }

        std::ostringstream mapping;
        mapping << "global_rank=" << globalRank
                << "|local_rank=" << localRank
                << "|hostname=" << gpu_session_detail::sanitized(
                       requireEnvironment("HOSTNAME"))
                << "|cuda_visible_devices="
                << gpu_session_detail::sanitized(cudaVisible)
                << "|slurm_step_gpus="
                << gpu_session_detail::sanitized(
                       slurmStepGpus.empty() ? "unset" : slurmStepGpus)
                << "|allocation_cuda_visible_devices="
                << gpu_session_detail::sanitized(allocationCudaVisible)
                << "|device_assignment_source="
                << gpu_session_detail::sanitized(deviceAssignmentSource)
                << "|cuda_device_count=" << visibleDevices
                << "|cuda_device_id=" << currentDevice
                << "|amrex_device_id=" << amrexDevice
                << "|device_name="
                << gpu_session_detail::sanitized(properties.name)
                << "|visible_mig_uuid="
                << gpu_session_detail::sanitized(cudaVisible)
                << "|cuda_parent_uuid=" << cudaParentUuid
                << "|compute_capability=" << properties.major << '.'
                << properties.minor
                << "|device_total_memory=" << properties.totalGlobalMem
                << "|pci_bus_id=" << pciBus.data()
                << "|cpu_cpuset=" << gpu_session_detail::sanitized(cpuset)
                << "|num_devices_used=" << numDevicesUsed
                << "|num_device_partners=" << numDevicePartners;

        bool migUuidFits = false;
        bool parentUuidFits = false;
        bool mappingFits = false;
        bool cpusetFits = false;
        std::vector<std::string> migUuids =
            gpu_session_detail::gather_fixed_strings<80>(
                cudaVisible, migUuidFits);
        std::vector<std::string> parentUuids =
            gpu_session_detail::gather_fixed_strings<80>(
                cudaParentUuid, parentUuidFits);
        std::vector<std::string> mappings =
            gpu_session_detail::gather_fixed_strings<1024>(
                mapping.str(), mappingFits);
        std::vector<std::string> cpusets =
            gpu_session_detail::gather_fixed_strings<128>(cpuset, cpusetFits);
        if (!migUuidFits || !parentUuidFits || !mappingFits || !cpusetFits) {
            status = std::max(status, 2);
        }
        amrex::ParallelDescriptor::ReduceIntMax(status);

        int globalStatus = status;
        if (amrex::ParallelDescriptor::IOProcessor() && globalStatus == 0) {
            std::sort(migUuids.begin(), migUuids.end());
            auto const uniqueMigEnd =
                std::unique(migUuids.begin(), migUuids.end());
            std::sort(parentUuids.begin(), parentUuids.end());
            auto const uniqueParentEnd =
                std::unique(parentUuids.begin(), parentUuids.end());
            if (uniqueMigEnd != migUuids.end() || migUuids.size() !=
                    static_cast<std::size_t>(rankCount) ||
                uniqueParentEnd != parentUuids.begin() + 1) {
                globalStatus = 1;
            }
        }
        amrex::ParallelDescriptor::Bcast(
            &globalStatus, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        if (globalStatus != 0) {
            return {
                globalStatus == 1
                    ? runtime::SingleLevelRunStatus::NumericalFailure
                    : runtime::SingleLevelRunStatus::OutputFailure,
                globalStatus == 1
                    ? "GPU rank/device mapping identity mismatch"
                    : "GPU runtime identity is incomplete"};
        }

        gpuContext_.amrexConfigSha256 =
            requireEnvironment("LBM_GPU_AMREX_CONFIG_SHA256");
        gpuContext_.nvcc = requireEnvironment("LBM_GPU_NVCC_VERSION");
        gpuContext_.cudaToolkit =
            requireEnvironment("LBM_GPU_CUDA_TOOLKIT_VERSION");
        gpuContext_.mpi = requireEnvironment("LBM_GPU_MPI");
        gpuContext_.mpiCudaBuildVersion =
            requireEnvironment("LBM_GPU_MPI_CUDA_BUILD_VERSION");
        gpuContext_.mpiCudaSupportReported =
            requireEnvironment("LBM_GPU_MPI_CUDA_SUPPORT_REPORTED");
        gpuContext_.mpiCudaQuerySupport =
            mpiCudaQueryMin != 0 ? "true" : "false";
        gpuContext_.jobId = requireEnvironment("SLURM_JOB_ID");
        gpuContext_.node = requireEnvironment("HOSTNAME");
        gpuContext_.partition = requireEnvironment("SLURM_JOB_PARTITION");
        gpuContext_.qos = requireEnvironment("LBM_GPU_QOS");
        gpuContext_.bindingPolicy =
            requireEnvironment("LBM_GPU_BINDING_POLICY");
        gpuContext_.migProfile = requireEnvironment("LBM_GPU_MIG_PROFILE");
        gpuContext_.parentGpuUuids = parentGpuUuids;
        gpuContext_.slurmJobGpus =
            requireEnvironment("SLURM_JOB_GPUS");
        amrex::ParallelDescriptor::ReduceIntMax(status);
        if (status != 0) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                "GPU runtime identity environment is incomplete"};
        }

        gpuContext_.cudaRuntime =
            gpu_session_detail::cuda_version_string(runtimeVersion);
        gpuContext_.cudaDriver =
            gpu_session_detail::cuda_version_string(driverVersion);
        gpuContext_.mpiRanks = rankCount;
        gpuContext_.migInstanceCount = rankCount;
        gpuContext_.parentGpuCount = parentGpuCount;
        gpuContext_.boxesPerRank = boxesPerRank;
        gpuContext_.ompThreads = ompThreads;
        gpuContext_.amrexUseGpuAwareMpi =
            amrex::ParallelDescriptor::UseGpuAwareMpi() ? 1 : 0;
        if (gpuContext_.amrexUseGpuAwareMpi != 0 && mpiCudaQueryMin == 0) {
            return {
                runtime::SingleLevelRunStatus::OutputFailure,
                "GPU-aware MPI requested but MPIX_Query_cuda_support is false"};
        }
        gpuContext_.communicationMode = gpuContext_.amrexUseGpuAwareMpi != 0
            ? "device_buffer"
            : "pinned_staging";
        gpuContext_.numDevicesUsed = numDevicesUsed;
        gpuContext_.numDevicePartners = numDevicePartners;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            gpuContext_.rankDeviceMap =
                gpu_session_detail::join_rank_values(mappings);
            gpuContext_.cpuCpuset =
                gpu_session_detail::join_rank_values(cpusets);
        }
        amrex::AllPrint()
            << "gpu_performance_identity_status=pass " << mapping.str()
            << '\n';
        return {};
    }
#endif

    PerformanceIdentity identity_;
    PerformanceControls controls_;
    std::array<int, 3> grid_{};
    std::int64_t globalValidCells_ = 0;
    GpuPerformanceContext gpuContext_;
    std::array<amrex::Real, Descriptor::q> finalPopulationComponentSums_{};
    amrex::Real finalPopulationL2_ = 0.0;
    bool started_ = false;
    bool finalStateChecked_ = false;
};

struct GpuPerformancePreparation {
    PerformanceIdentity identity;
    PerformanceControls controls;

    template <class Descriptor>
    using RunSession = GpuPerformanceRunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        auto reject = [&](char const* code, char const* subject, char const* message) {
            result.diagnostics.push_back({code, subject, message});
        };

        auto const requestedMode =
            gpu_session_detail::profiling_mode(controls.measurementMode);
        if (controls.workload != identity.workload) {
            reject(
                "performance_workload_mismatch",
                "performance.workload",
                "input workload does not match the compiled GPU target");
        }
        if (!detail::valid_mode(controls.mode)) {
            reject(
                "invalid_performance_mode",
                "performance.mode",
                "GPU performance mode must be smoke, calibration or baseline");
        }
        if (!requestedMode.has_value()) {
            reject(
                "invalid_measurement_mode",
                "performance.measurement_mode",
                "measurement mode must be throughput or stage_breakdown");
        }
        if (controls.repeat <= 0 || controls.outputCsv.empty()) {
            reject(
                "invalid_performance_output",
                "performance.output",
                "repeat must be positive and output_csv must be non-empty");
        }
        if (view.profiling.enabled != 1 || !requestedMode.has_value() ||
            view.profiling.mode != *requestedMode ||
            view.profiling.measurementSteps <= 0 ||
            view.run.maxStep != view.profiling.warmupSteps +
                    view.profiling.measurementSteps) {
            reject(
                "invalid_performance_window",
                "profiling.window",
                "GPU performance requires one complete explicit timing window");
        }
        for (int direction = 0; direction < 3; ++direction) {
            if (view.mesh.nCell[direction] != identity.grid[direction]) {
                reject(
                    "performance_grid_mismatch",
                    "mesh.n_cell",
                    "compiled GPU performance grid identity was changed");
                break;
            }
        }
        if (view.mesh.maxGridSize != identity.maxGridSize) {
            reject(
                "performance_box_mismatch",
                "mesh.max_grid_size",
                "compiled GPU performance max_grid_size identity was changed");
        }
        if (result.diagnostics.empty()) {
            result.session = RunSession<Descriptor>(
                std::move(identity), std::move(controls));
        }
        return result;
    }
};

} // namespace lbm::tests::performance

#endif
