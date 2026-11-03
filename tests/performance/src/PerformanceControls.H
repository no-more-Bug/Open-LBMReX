#ifndef LBM_TESTS_PERFORMANCE_PERFORMANCECONTROLS_H
#define LBM_TESTS_PERFORMANCE_PERFORMANCECONTROLS_H

#include <AMReX_ParmParse.H>

#include <array>
#include <string>

namespace lbm::tests::performance {

struct PerformanceIdentity {
    std::string workload;
    std::array<int, 3> grid{};
    int maxGridSize = 0;
    int boxCount = 0;
};

struct PerformanceControls {
    int maxStep = 0;
    int warmupSteps = 0;
    int measurementSteps = 0;
    std::string workload;
    std::string mode;
    std::string measurementMode;
    int repeat = 0;
    std::string outputCsv;
};

inline PerformanceControls read_performance_controls()
{
    PerformanceControls controls;
    amrex::ParmParse("run").get("max_step", controls.maxStep);
    amrex::ParmParse profiling("profiling");
    profiling.get("warmup_steps", controls.warmupSteps);
    profiling.get("measurement_steps", controls.measurementSteps);
    amrex::ParmParse performance("performance");
    performance.get("workload", controls.workload);
    performance.get("mode", controls.mode);
#if defined(LBM_GPU_PERFORMANCE) || defined(LBM_AMR_PERFORMANCE)
    performance.get("measurement_mode", controls.measurementMode);
#else
    controls.measurementMode = "stage_breakdown";
#endif
    performance.get("repeat", controls.repeat);
    performance.get("output_csv", controls.outputCsv);
    return controls;
}

} // namespace lbm::tests::performance

#endif
