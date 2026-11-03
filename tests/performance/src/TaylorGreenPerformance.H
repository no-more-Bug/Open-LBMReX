#ifndef LBM_TESTS_PERFORMANCE_TAYLORGREENPERFORMANCE_H
#define LBM_TESTS_PERFORMANCE_TAYLORGREENPERFORMANCE_H

#ifdef LBM_GPU_PERFORMANCE
#include "GpuPerformanceRunSession.H"
#else
#include "PerformanceRunSession.H"
#endif

#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_REAL.H>

#include <string>
#include <utility>

namespace lbm::tests::performance {

template <class Descriptor>
auto make_taylor_green_performance_program(PerformanceControls controls)
{
#ifdef LBM_GPU_PERFORMANCE
    using Preparation = GpuPerformancePreparation;
#else
    using Preparation = PerformancePreparation;
#endif
    using Program = runtime::SingleLevelProgram<
        Descriptor,
        BGKDynamicsSelection,
        initialization::ConfiguredInitializationSelection,
        Preparation>;

    PerformanceIdentity identity;
    if constexpr (Descriptor::d == 2) {
        identity = {"tgv_d2q9", {1024, 1024, 1}, 256, 16};
    } else {
        identity = {"tgv_d3q27", {512, 512, 4}, 128, 16};
    }

    Program program;
    program.mesh.nCell = identity.grid;
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        static_cast<amrex::Real>(identity.grid[2]) /
            static_cast<amrex::Real>(identity.grid[0])};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = identity.maxGridSize;
    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.01;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = 0.01;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "performance";
    program.run.name = "taylor_green_vortex";
    program.run.variant = identity.workload;
    program.profiling = {
        1,
        controls.warmupSteps,
        controls.measurementSteps,
#ifdef LBM_GPU_PERFORMANCE
        gpu_session_detail::profiling_mode(controls.measurementMode)
            .value_or(profiling::TimestepProfilingMode::Disabled)
#else
        profiling::TimestepProfilingMode::StageBreakdown
#endif
    };
    program.initialization.intent.kind =
        initialization::InitializationKind::TaylorGreen;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb = 0.01;
    program.casePreparation = {
        std::move(identity), std::move(controls)};
    return program;
}

} // namespace lbm::tests::performance

#endif
