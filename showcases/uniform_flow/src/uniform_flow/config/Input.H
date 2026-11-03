#ifndef LBM_CASES_UNIFORMFLOW_INPUT_H
#define LBM_CASES_UNIFORMFLOW_INPUT_H

#include "uniform_flow/config/Controls.H"

#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::uniform_flow {

struct ControlsReadResult {
    Controls controls;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline ControlsReadResult read_controls()
{
    ControlsReadResult result;
    Controls& controls = result.controls;
    amrex::ParmParse uniform("uniform_flow");
    std::string variant = to_string(controls.variant);
    uniform.query("variant", variant);
    if (variant == "hbb_velocity_io") {
        controls.variant = Variant::HbbVelocityIo;
    } else if (variant == "guo_velocity_io") {
        controls.variant = Variant::GuoVelocityIo;
    } else if (variant == "guo_velocity_free_outflow") {
        controls.variant = Variant::GuoVelocityFreeOutflow;
    } else {
        result.errors.push_back(
            "uniform_flow.variant must be hbb_velocity_io, guo_velocity_io or guo_velocity_free_outflow");
    }
    uniform.query("intervals", controls.intervals);
    uniform.query("depth_cells", controls.depthCells);
    uniform.query("rho_lb", controls.rhoLb);
    uniform.query("velocity_lb", controls.velocityLb);
    uniform.query("velocity_reference_phys", controls.velocityReferencePhys);
    uniform.query("nu_lb", controls.nuLb);

    amrex::ParmParse mesh("mesh");
    mesh.query("max_grid_size", controls.maxGridSize);
    amrex::ParmParse run("run");
    run.query("max_step", controls.maxStep);
    run.query("final_time_phys", controls.finalTimePhys);
    run.query("kind", controls.runKind);
    run.query("variant", controls.runVariant);
    amrex::ParmParse diagnostics("diagnostics");
    diagnostics.query("interval", controls.diagnosticsInterval);
    amrex::ParmParse validation("uniform_validation");
    validation.query("write_metrics", controls.observation.writeMetrics);
    validation.query("assert", controls.observation.assertEnabled);
    validation.query(
        "min_flow_throughs", controls.observation.minFlowThroughs);
    validation.query("e2_u_tolerance", controls.observation.e2UTolerance);
    validation.query("einf_u_tolerance", controls.observation.eInfUTolerance);
    validation.query("std_ux_tolerance", controls.observation.stdUxTolerance);
    validation.query(
        "transverse_tolerance", controls.observation.transverseTolerance);
    validation.query("mass_tolerance", controls.observation.massTolerance);
    controls.observation.enabled =
        controls.observation.writeMetrics != 0 ||
        controls.observation.assertEnabled != 0;
    amrex::ParmParse output("output");
    output.query("directory", controls.outputDirectory);
    output.query("log_directory", controls.logDirectory);
    output.query("plot_interval", controls.plotInterval);
    output.query("write_final_plotfile", controls.writeFinalPlotfile);

    auto add = [&](bool invalid, std::string message) {
        if (invalid) {
            result.errors.push_back(std::move(message));
        }
    };
    add(
        controls.intervals[0] <= 0 || controls.intervals[1] <= 0 ||
            controls.depthCells <= 0 || controls.maxGridSize <= 0,
        "uniform-flow intervals, depth_cells and max_grid_size must be positive");
    add(
        !std::isfinite(controls.rhoLb) || !(controls.rhoLb > 0.0) ||
            !std::isfinite(controls.velocityLb) ||
            !(controls.velocityLb > 0.0) ||
            !std::isfinite(controls.velocityReferencePhys) ||
            !(controls.velocityReferencePhys > 0.0) ||
            !std::isfinite(controls.nuLb) || !(controls.nuLb > 0.0),
        "uniform-flow density, velocity and viscosity must be positive and finite");
    add(
        controls.maxStep <= 0 || controls.diagnosticsInterval < 0 ||
            controls.plotInterval < 0 ||
            !std::isfinite(controls.finalTimePhys) ||
            !(controls.finalTimePhys > 0.0),
        "uniform-flow run controls are invalid");
    auto validFlag = [](int value) { return value == 0 || value == 1; };
    add(
        !validFlag(controls.writeFinalPlotfile) ||
            !validFlag(controls.observation.writeMetrics) ||
            !validFlag(controls.observation.assertEnabled),
        "uniform-flow boolean controls must be 0 or 1");
    add(
        !valid_observation_config(controls.observation),
        "uniform-flow acceptance requires non-negative flow-throughs and positive tolerances");
    add(
        controls.outputDirectory.empty() || controls.logDirectory.empty() ||
            controls.runKind.empty() || controls.runVariant.empty(),
        "uniform-flow output paths and run identity must not be empty");
    return result;
}

} // namespace lbm::cases::uniform_flow

#endif
