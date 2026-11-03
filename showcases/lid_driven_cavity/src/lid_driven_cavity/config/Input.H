#ifndef LBM_CASES_LIDDRIVENCAVITY_INPUT_H
#define LBM_CASES_LIDDRIVENCAVITY_INPUT_H

#include "lid_driven_cavity/config/Controls.H"

#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::lid_driven_cavity {

struct ControlsReadResult {
    Controls controls;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline ControlsReadResult read_controls()
{
    ControlsReadResult result;
    Controls& controls = result.controls;

    amrex::ParmParse cavity("cavity");
    cavity.query("physical_lo", controls.physicalLo);
    cavity.query("physical_hi", controls.physicalHi);
    cavity.query("intervals", controls.intervals);
    cavity.query("depth_cells", controls.depthCells);
    std::string wallScheme = to_string(controls.wallScheme);
    cavity.query("wall_scheme", wallScheme);
    if (wallScheme == "halfway_bounce_back") {
        controls.wallScheme = WallScheme::HalfwayBounceBack;
    } else if (wallScheme == "guo_extrapolation") {
        controls.wallScheme = WallScheme::GuoExtrapolation;
    } else {
        result.errors.push_back(
            "cavity.wall_scheme must be halfway_bounce_back or "
            "guo_extrapolation");
    }
    cavity.query("rho_lb", controls.rhoLb);
    cavity.query("re", controls.reynolds);
    cavity.query("lid_velocity_lb", controls.lidVelocityLb);
    cavity.query("u_ref_phys", controls.velocityReferencePhys);

    amrex::ParmParse mesh("mesh");
    mesh.query("max_grid_size", controls.maxGridSize);

    amrex::ParmParse run("run");
    run.query("max_step", controls.maxStep);
    run.query("final_time_phys", controls.finalTimePhys);
    run.query("kind", controls.runKind);
    run.query("variant", controls.runVariant);

    amrex::ParmParse diagnostics("diagnostics");
    diagnostics.query("interval", controls.diagnosticsInterval);
    diagnostics.query("steady_check_interval", controls.steadyCheckInterval);
    diagnostics.query("steady_l2_tolerance", controls.steadyL2Tolerance);
    diagnostics.query("stop_on_steady", controls.stopOnSteady);
    diagnostics.query("write_centerlines", controls.writeCenterlines);
    diagnostics.query("write_metrics", controls.writeMetrics);

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
    for (int axis = 0; axis < 2; ++axis) {
        add(
            !std::isfinite(controls.physicalLo[axis]) ||
                !std::isfinite(controls.physicalHi[axis]) ||
                !(controls.physicalHi[axis] > controls.physicalLo[axis]),
            "cavity physical bounds must be finite and increasing");
        add(
            controls.intervals[axis] <= 0,
            "cavity intervals must be positive");
    }
    const amrex::Real lengthX =
        controls.physicalHi[0] - controls.physicalLo[0];
    const amrex::Real lengthY =
        controls.physicalHi[1] - controls.physicalLo[1];
    const amrex::Real dx =
        controls.intervals[0] > 0 ?
            lengthX / static_cast<amrex::Real>(controls.intervals[0]) : 0.0;
    const amrex::Real dy =
        controls.intervals[1] > 0 ?
            lengthY / static_cast<amrex::Real>(controls.intervals[1]) : 0.0;
    const amrex::Real scale = std::max(
        amrex::Real(1.0), std::max(std::abs(dx), std::abs(dy)));
    add(
        controls.intervals[0] != controls.intervals[1] ||
            std::abs(lengthX - lengthY) > 1.0e-14 *
                std::max(amrex::Real(1.0), std::max(lengthX, lengthY)) ||
            std::abs(dx - dy) > 1.0e-14 * scale,
        "canonical cavity requires a square physical domain with equal x/y intervals");
    add(
        !std::isfinite(controls.rhoLb) || !(controls.rhoLb > 0.0),
        "cavity.rho_lb must be positive and finite");
    add(
        !std::isfinite(controls.reynolds) || !(controls.reynolds > 0.0),
        "cavity.re must be positive and finite");
    add(
        !std::isfinite(controls.lidVelocityLb) ||
            !(controls.lidVelocityLb > 0.0),
        "cavity.lid_velocity_lb must be positive and finite");
    add(
        !std::isfinite(controls.velocityReferencePhys) ||
            !(controls.velocityReferencePhys > 0.0),
        "cavity.u_ref_phys must be positive and finite");
    add(
        controls.depthCells <= 0 || controls.maxGridSize <= 0 ||
            controls.maxStep <= 0,
        "cavity.depth_cells, mesh.max_grid_size and run.max_step must be positive");
    add(
        !std::isfinite(controls.finalTimePhys) ||
            !(controls.finalTimePhys > 0.0),
        "run.final_time_phys must be positive and finite");
    add(
        controls.diagnosticsInterval < 0 ||
            controls.plotInterval < 0 ||
            controls.steadyCheckInterval < 0,
        "diagnostic, plot and steady intervals must be non-negative");
    add(
        !std::isfinite(controls.steadyL2Tolerance) ||
            !(controls.steadyL2Tolerance > 0.0),
        "diagnostics.steady_l2_tolerance must be positive and finite");
    auto validFlag = [](int value) { return value == 0 || value == 1; };
    add(
        !validFlag(controls.writeFinalPlotfile) ||
            !validFlag(controls.stopOnSteady) ||
            !validFlag(controls.writeCenterlines) ||
            !validFlag(controls.writeMetrics),
        "direct cavity boolean controls must be 0 or 1");
    add(
        controls.wallScheme == WallScheme::HalfwayBounceBack &&
            controls.writeMetrics != 0,
        "HBB cavity does not expose on-lattice wall validation metrics; "
        "set diagnostics.write_metrics=0");
    add(
        controls.outputDirectory.empty() || controls.logDirectory.empty(),
        "output.directory and output.log_directory must not be empty");
    add(
        controls.runKind.empty() || controls.runVariant.empty(),
        "run.kind and run.variant must not be empty");
    return result;
}

} // namespace lbm::cases::lid_driven_cavity

#endif
