#ifndef LBM_CASES_CHANNELCYLINDER_INPUT_H
#define LBM_CASES_CHANNELCYLINDER_INPUT_H

#include "channel_cylinder/config/Controls.H"

#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::channel_cylinder {

struct ControlsReadResult {
    Controls controls;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline ControlsReadResult read_controls()
{
    ControlsReadResult result;
    Controls& controls = result.controls;

    amrex::ParmParse channel("channel_cylinder");
    std::string variant = to_string(controls.variant);
    channel.query("variant", variant);
    if (variant == "single_yms") {
        controls.variant = Variant::SingleYms;
    } else if (variant == "single_bouzidi") {
        controls.variant = Variant::SingleBouzidi;
    } else if (variant == "tandem_distinct") {
        controls.variant = Variant::TandemDistinct;
    } else if (variant == "tandem_shared") {
        controls.variant = Variant::TandemShared;
    } else if (variant == "re20_yms") {
        controls.variant = Variant::Re20Yms;
    } else if (variant == "re100_yms") {
        controls.variant = Variant::Re100Yms;
    } else {
        result.errors.push_back(
            "channel_cylinder.variant must be single_yms, single_bouzidi, "
            "tandem_distinct, tandem_shared, re20_yms or re100_yms");
    }
    channel.query("diameter_intervals", controls.diameterIntervals);
    channel.query("depth_cells", controls.depthCells);
    channel.query("rho_lb", controls.rhoLb);
    channel.query("u_mean_lb", controls.uMeanLb);
    channel.query("u_mean_phys", controls.uMeanPhys);
    channel.query("reynolds_height", controls.reynoldsHeight);

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

    amrex::ParmParse observation("channel_cylinder_observation");
    observation.query("force_interval", controls.forceInterval);
    observation.query("output_interval", controls.outputInterval);
    observation.query("write_history", controls.writeHistory);
    observation.query("assert", controls.assertEnabled);

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
    const bool supportedDiameter =
        controls.variant == Variant::Re20Yms ?
            (controls.diameterIntervals == 20 ||
             controls.diameterIntervals == 40 ||
             controls.diameterIntervals == 80) :
        controls.variant == Variant::Re100Yms ?
            (controls.diameterIntervals == 40 ||
             controls.diameterIntervals == 80) :
            controls.diameterIntervals == 20;
    add(
        !supportedDiameter,
        "channel-cylinder re20_yms supports N_D=20,40,80; re100_yms "
        "supports N_D=40,80; other variants require N_D=20");
    add(
        controls.depthCells <= 0 || controls.maxGridSize <= 0,
        "channel-cylinder depth_cells and max_grid_size must be positive");
    add(
        !std::isfinite(controls.rhoLb) || !(controls.rhoLb > 0.0) ||
            !std::isfinite(controls.uMeanLb) || !(controls.uMeanLb > 0.0) ||
            !std::isfinite(controls.uMeanPhys) ||
            !(controls.uMeanPhys > 0.0) ||
            (controls.variant == Variant::Re100Yms &&
             std::abs(controls.uMeanPhys - amrex::Real(1.0)) > 1.0e-12) ||
            !std::isfinite(controls.reynoldsHeight) ||
            std::abs(
                controls.reynoldsHeight -
                (controls.variant == Variant::Re100Yms ?
                    amrex::Real(410.0) : amrex::Real(82.0))) > 1.0e-12,
        "channel-cylinder controls do not match the selected Reynolds identity");
    add(
        controls.maxStep <= 0 ||
            !std::isfinite(controls.finalTimePhys) ||
            !(controls.finalTimePhys > 0.0) ||
            controls.diagnosticsInterval < 0 ||
            controls.steadyCheckInterval < 0 ||
            !std::isfinite(controls.steadyL2Tolerance) ||
            !(controls.steadyL2Tolerance > 0.0) ||
            controls.plotInterval < 0,
        "channel-cylinder run, diagnostics or steady controls are invalid");
    add(
        controls.forceInterval <= 0 ||
            controls.outputInterval < controls.forceInterval,
        "channel-cylinder output_interval must be at least force_interval > 0");
    auto flag = [](int value) { return value == 0 || value == 1; };
    add(
        !flag(controls.stopOnSteady) ||
            !flag(controls.writeFinalPlotfile) ||
            !flag(controls.writeHistory) ||
            !flag(controls.assertEnabled),
        "channel-cylinder boolean controls must be 0 or 1");
    add(
        controls.runKind.empty() || controls.runVariant.empty() ||
            controls.outputDirectory.empty() || controls.logDirectory.empty(),
        "channel-cylinder run identity and output paths must not be empty");
    return result;
}

} // namespace lbm::cases::channel_cylinder

#endif
