#ifndef LBM_TESTS_PERFORMANCE_AMRPERFORMANCECSV_H
#define LBM_TESTS_PERFORMANCE_AMRPERFORMANCECSV_H

#include "PerformanceControls.H"
#include "lbm/profiling/AmrTimestepProfiler.H"
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <array>
#include <cstdint>
#include <vector>

namespace lbm::tests::performance {
struct AmrPerformanceLevel {
    std::array<int,3> grid;
    amrex::BoxArray boxes;
    amrex::DistributionMapping mapping;
    amrex::Real dt;
    amrex::Real tau;
};

// These are two device snapshots, not a device peak-memory measurement.
std::uint64_t amr_device_used_bytes();
void write_amr_resource_csv(std::string const& outputCsv, std::uint64_t initializedUsedBytes);
void write_amr_performance_csv(
    PerformanceControls const& controls, std::string const& layout, char const* descriptor,
    std::vector<AmrPerformanceLevel> const& levels, profiling::AmrPerformanceRecord const& record,
    amrex::Real initializationSeconds);
} // namespace lbm::tests::performance
#endif
