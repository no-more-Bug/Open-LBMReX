#ifndef LBM_TESTS_PERFORMANCE_DFGPERFORMANCE_H
#define LBM_TESTS_PERFORMANCE_DFGPERFORMANCE_H

#ifdef LBM_GPU_PERFORMANCE
#include "GpuPerformanceRunSession.H"
#else
#include "PerformanceRunSession.H"
#endif

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_REAL.H>

#include <string>
#include <utility>

namespace lbm::tests::performance {

inline boundary::DomainBoundarySpec dfg_inlet(amrex::Real uMeanLb)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "inlet";
    spec.location = boundary::DomainSide::XLo;
    spec.physical = boundary::BoundaryPhysical::VelocityInlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Poiseuille;
    spec.profileAxis = 1;
    spec.hasProfileAxis = true;
    spec.uMeanLb = uMeanLb;
    spec.hasUMeanLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec dfg_outlet(amrex::Real rhoLb)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "outlet";
    spec.location = boundary::DomainSide::XHi;
    spec.physical = boundary::BoundaryPhysical::PressureOutlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Constant;
    spec.rhoLb = rhoLb;
    spec.hasRhoLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec dfg_wall(boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    spec.name = side == boundary::DomainSide::YLo
        ? "bottom_wall" : "top_wall";
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    return spec;
}

inline boundary::InternalBoundarySpec dfg_cylinder(
    amrex::Real dx,
    int depthCells)
{
    boundary::InternalBoundarySpec spec;
    spec.name = "cylinder_wall";
    spec.geometry = "cylinder";
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.hasPhysical = true;
    spec.scheme = boundary::BoundaryScheme::Yms;
    spec.profile = boundary::BoundaryProfileKind::None;
    spec.objectName = "cylinder";
    spec.centerPhys = {0.2, 0.2, 0.5 * dx * depthCells};
    spec.hasCenterPhys = true;
    spec.radiusPhys = 0.05;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

template <class Descriptor>
auto make_dfg_performance_program(PerformanceControls controls)
{
#ifdef LBM_GPU_PERFORMANCE
    using Preparation = GpuPerformancePreparation;
#else
    using Preparation = PerformancePreparation;
#endif
    using Profile = initialization::ConstantInitializationProfile;
    using Initialization =
        initialization::ProfileInitializationSelection<Profile>;
    using Program = runtime::SingleLevelProgram<
        Descriptor,
        BGKDynamicsSelection,
        Initialization,
        Preparation>;

    constexpr amrex::Real rhoLb = 1.0;
    constexpr amrex::Real uMeanLb = 0.035;
    constexpr int diameterIntervals = Descriptor::d == 2 ? 80 : 40;
    constexpr int depthCells = Descriptor::d == 2 ? 1 : 4;
    constexpr int maxGridSize = Descriptor::d == 2 ? 224 : 112;
    constexpr int boxCount = 16;
    constexpr amrex::Real dx = 0.1 / diameterIntervals;
    constexpr int xIntervals = 22 * diameterIntervals;
    constexpr int yIntervals = 41 * diameterIntervals / 10;

    PerformanceIdentity identity{
        Descriptor::d == 2 ? "dfg_d2q9" : "dfg_d3q27",
        {xIntervals + 1, yIntervals + 1, depthCells},
        maxGridSize,
        boxCount};

    Program program;
    program.mesh.nCell = identity.grid;
    program.mesh.envelopeLo = {-0.5 * dx, -0.5 * dx, 0.0};
    program.mesh.envelopeHi = {
        2.2 + 0.5 * dx,
        0.41 + 0.5 * dx,
        dx * depthCells};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = maxGridSize;
    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = rhoLb;
    program.lbm.uRefLb = uMeanLb;
    program.lbm.uRefPhys = 0.2;
    program.lbm.lengthRefLb = yIntervals;
    program.lbm.lengthRefPhys = 0.41;
    program.lbm.reynolds = 82.0;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "performance";
    program.run.name = "channel_cylinder";
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
    program.initialization = initialization::from_profile(Profile{{
        rhoLb, uMeanLb, 0.0, 0.0}});
    program.domainBoundaries = {
        dfg_inlet(uMeanLb),
        dfg_outlet(rhoLb),
        dfg_wall(boundary::DomainSide::YLo),
        dfg_wall(boundary::DomainSide::YHi)};
    program.internalBoundaries = {dfg_cylinder(dx, depthCells)};
    program.casePreparation = {
        std::move(identity), std::move(controls)};
    return program;
}

} // namespace lbm::tests::performance

#endif
