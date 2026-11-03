#include <AMReX.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#ifdef AMREX_USE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class ExitStatus : int {
    Pass = 0,
    EnvironmentFailure = 2,
    MappingFailure = 3
};

std::string environment_value(char const* name)
{
    char const* value = std::getenv(name);
    return value == nullptr ? "unset" : value;
}

#ifdef AMREX_USE_CUDA
std::string uuid_string(cudaUUID_t const& uuid)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char byte : uuid.bytes) {
        stream << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return stream.str();
}

int check_cuda_call(cudaError_t error, char const* operation)
{
    if (error == cudaSuccess) {
        return static_cast<int>(ExitStatus::Pass);
    }
    amrex::AllPrint() << "gpu_mapping_status=environment-failure operation="
                      << operation << " cuda_error="
                      << cudaGetErrorString(error) << '\n';
    return static_cast<int>(ExitStatus::EnvironmentFailure);
}
#endif

int check_mapping()
{
#ifndef AMREX_USE_CUDA
    amrex::AllPrint()
        << "gpu_mapping_status=environment-failure reason=cuda-backend-disabled\n";
    return static_cast<int>(ExitStatus::EnvironmentFailure);
#else
    int status = static_cast<int>(ExitStatus::Pass);
    int visibleDevices = 0;
    int currentDevice = -1;
    cudaDeviceProp properties{};

    status = std::max(
        status,
        check_cuda_call(cudaGetDeviceCount(&visibleDevices), "cudaGetDeviceCount"));
    if (status == static_cast<int>(ExitStatus::Pass)) {
        status = std::max(
            status,
            check_cuda_call(cudaGetDevice(&currentDevice), "cudaGetDevice"));
    }
    if (status == static_cast<int>(ExitStatus::Pass)) {
        status = std::max(
            status,
            check_cuda_call(
                cudaGetDeviceProperties(&properties, currentDevice),
                "cudaGetDeviceProperties"));
    }

    std::string const visible = environment_value("CUDA_VISIBLE_DEVICES");
    if (visible == "unset" || visible.empty()) {
        amrex::AllPrint()
            << "gpu_mapping_status=environment-failure "
               "reason=CUDA_VISIBLE_DEVICES-unset\n";
        status = std::max(
            status,
            static_cast<int>(ExitStatus::EnvironmentFailure));
    }

    int const rankCount = amrex::ParallelDescriptor::NProcs();
    // CUDA reports the parent GPU UUID on this MIG stack. Check the observed
    // parent groups instead of requiring every rank to share one parent.
    std::string const parent(properties.uuid.bytes, sizeof(properties.uuid.bytes));
    std::vector<char> parents(parent.size() * rankCount);
#ifdef AMREX_USE_MPI
    MPI_Allgather(parent.data(), static_cast<int>(parent.size()), MPI_CHAR,
                  parents.data(), static_cast<int>(parent.size()), MPI_CHAR,
                  amrex::ParallelDescriptor::Communicator());
#else
    std::copy(parent.begin(), parent.end(), parents.begin());
#endif
    std::map<std::string, int> parentCounts;
    for (int rank = 0; rank < rankCount; ++rank) {
        ++parentCounts[std::string(parents.data() + rank * parent.size(), parent.size())];
    }
    if (status == static_cast<int>(ExitStatus::Pass) &&
        (visible.rfind("MIG-", 0) != 0 || visible.find(',') != std::string::npos ||
         visibleDevices != 1 || currentDevice != 0 ||
         amrex::Gpu::Device::deviceId() != 0 ||
         amrex::Gpu::Device::numDevicesUsed() != static_cast<int>(parentCounts.size()) ||
         amrex::Gpu::Device::numDevicePartners() != parentCounts.at(parent))) {
        status = static_cast<int>(ExitStatus::MappingFailure);
    }

    std::ostringstream record;
    record << "gpu_mapping_status="
           << (status == static_cast<int>(ExitStatus::Pass)
                   ? "pass"
                   : status == static_cast<int>(ExitStatus::MappingFailure)
                       ? "mapping-failure"
                       : "environment-failure")
           << " global_rank=" << amrex::ParallelDescriptor::MyProc()
           << " local_rank=" << environment_value("SLURM_LOCALID")
           << " amrex_local_rank="
           << amrex::ParallelDescriptor::MyRankInNode()
           << " ompi_local_rank="
           << environment_value("OMPI_COMM_WORLD_LOCAL_RANK")
           << " cuda_visible_devices=" << visible
           << " slurm_job_gpus=" << environment_value("SLURM_JOB_GPUS")
           << " slurm_step_gpus=" << environment_value("SLURM_STEP_GPUS")
           << " visible_device_count=" << visibleDevices
           << " cuda_device_id=" << currentDevice
           << " amrex_device_id=" << amrex::Gpu::Device::deviceId()
           << " num_devices_used=" << amrex::Gpu::Device::numDevicesUsed()
           << " num_device_partners="
           << amrex::Gpu::Device::numDevicePartners();
    if (status != static_cast<int>(ExitStatus::EnvironmentFailure)) {
        record << " device_name=" << properties.name
               << " visible_mig_uuid=" << visible
               << " cuda_parent_uuid=" << uuid_string(properties.uuid)
               << " compute_capability=" << properties.major << '.'
               << properties.minor
               << " total_memory_bytes=" << properties.totalGlobalMem;
    }
    amrex::AllPrint() << record.str() << '\n';

    amrex::ParallelDescriptor::ReduceIntMax(status);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "gpu_mapping_global_status=" << status << '\n';
    }
    return status;
#endif
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    int const status = check_mapping();
    amrex::Finalize();
    return status;
}
