#ifndef LBM_CASES_TAYLORGREENVORTEX_INPUT_H
#define LBM_CASES_TAYLORGREENVORTEX_INPUT_H

#include "taylor_green_vortex/config/Controls.H"

#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::taylor_green_vortex {

struct ControlsReadResult {
    Controls controls;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline ControlsReadResult read_controls()
{
    ControlsReadResult result;
    Controls& controls = result.controls;

    amrex::ParmParse tgv("taylor_green");
    tgv.query("n_cell", controls.nCell);
    std::string parameterization = config::to_string(controls.parameterization);
    tgv.query("parameterization", parameterization);
    auto const parsedParameterization =
        config::parse_lbm_parameterization_kind(parameterization);
    if (!parsedParameterization.has_value() ||
        *parsedParameterization == config::LbmParameterizationKind::FromReynolds) {
        result.errors.push_back(
            "taylor_green.parameterization must be from_nu_lb or from_tau");
    } else {
        controls.parameterization = *parsedParameterization;
    }
    tgv.query("rho_lb", controls.rhoLb);
    tgv.query("u_ref_lb", controls.uRefLb);
    tgv.query("u_ref_phys", controls.uRefPhys);
    tgv.query("nu_lb", controls.nuLb);
    tgv.query("tau", controls.tau);
    tgv.query("re", controls.reynolds);
    amrex::Real amplitude = controls.uRefLb;
    if (tgv.query("velocity_amplitude_lb", amplitude)) {
        controls.velocityAmplitudeLb = amplitude;
    }

    amrex::ParmParse mesh("mesh");
    mesh.query("max_grid_size", controls.maxGridSize);

    amrex::ParmParse run("run");
    run.query("max_step", controls.maxStep);
    run.query("final_time_phys", controls.finalTimePhys);
    run.query("kind", controls.runKind);
    run.query("variant", controls.runVariant);
    run.query("convergence_enabled", controls.convergenceEnabled);
    run.query("convergence_reset", controls.convergenceReset);

    amrex::ParmParse diagnostics("diagnostics");
    diagnostics.query("interval", controls.diagnosticsInterval);

    amrex::ParmParse output("output");
    output.query("directory", controls.outputDirectory);
    output.query("log_directory", controls.logDirectory);
    output.query("convergence_directory", controls.convergenceDirectory);
    output.query("plot_interval", controls.plotInterval);
    output.query("write_final_plotfile", controls.writeFinalPlotfile);

    auto add = [&](bool invalid, std::string message) {
        if (invalid) {
            result.errors.push_back(std::move(message));
        }
    };
    add(
        controls.nCell[0] <= 0 || controls.nCell[1] <= 0 ||
            controls.nCell[2] <= 0,
        "taylor_green.n_cell must be positive in every direction");
    add(
        controls.nCell[0] != controls.nCell[1],
        "Taylor-Green requires a square x-y grid");
    add(
        controls.maxGridSize <= 0,
        "mesh.max_grid_size must be positive");
    add(
        !std::isfinite(controls.rhoLb) || !(controls.rhoLb > 0.0) ||
            !std::isfinite(controls.uRefPhys) || !(controls.uRefPhys > 0.0),
        "Taylor-Green density and physical velocity reference must be positive and finite");
    if (controls.parameterization == config::LbmParameterizationKind::FromNuLb) {
        add(
            !std::isfinite(controls.uRefLb) || !(controls.uRefLb > 0.0) ||
                !std::isfinite(controls.nuLb) || !(controls.nuLb > 0.0),
            "from_nu_lb requires positive finite u_ref_lb and nu_lb");
    } else if (controls.parameterization == config::LbmParameterizationKind::FromTau) {
        add(
            !std::isfinite(controls.tau) || !(controls.tau > 0.5) ||
                !std::isfinite(controls.reynolds) || !(controls.reynolds > 0.0),
            "from_tau requires tau greater than 0.5 and positive finite re");
    }
    add(
        controls.velocityAmplitudeLb.has_value() &&
            (!std::isfinite(*controls.velocityAmplitudeLb) ||
             *controls.velocityAmplitudeLb < 0.0),
        "Taylor-Green velocity amplitude must be finite and non-negative");
    add(
        controls.maxStep <= 0 ||
            !std::isfinite(controls.finalTimePhys) ||
            !(controls.finalTimePhys > 0.0),
        "run.max_step and run.final_time_phys must be positive");
    add(
        controls.diagnosticsInterval < 0 || controls.plotInterval < 0,
        "diagnostic and plot intervals must be non-negative");
    auto validFlag = [](int value) { return value == 0 || value == 1; };
    add(
        !validFlag(controls.convergenceEnabled) ||
            !validFlag(controls.convergenceReset) ||
            !validFlag(controls.writeFinalPlotfile),
        "direct Taylor-Green boolean controls must be 0 or 1");
    add(
        controls.convergenceEnabled != 0 &&
            controls.parameterization != config::LbmParameterizationKind::FromTau,
        "Taylor-Green convergence variants require from_tau");
    add(
        controls.outputDirectory.empty() || controls.logDirectory.empty() ||
            controls.runKind.empty() || controls.runVariant.empty(),
        "output paths, run.kind and run.variant must not be empty");
    return result;
}

} // namespace lbm::cases::taylor_green_vortex

#endif
