#ifndef LBM_CASES_POISEUILLEFLOW_INPUT_H
#define LBM_CASES_POISEUILLEFLOW_INPUT_H

#include "poiseuille_flow/config/Controls.H"

#include <AMReX_ParmParse.H>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::poiseuille_flow {

struct ControlsReadResult {
    Controls controls;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline ControlsReadResult read_controls()
{
    ControlsReadResult result;
    Controls& controls = result.controls;
    amrex::ParmParse poiseuille("poiseuille_flow");
    std::string variant = to_string(controls.variant);
    poiseuille.query("variant", variant);
    if (variant == "hbb_velocity") {
        controls.variant = Variant::HbbVelocity;
    } else if (variant == "guo_velocity") {
        controls.variant = Variant::GuoVelocity;
    } else if (variant == "guo_pressure") {
        controls.variant = Variant::GuoPressure;
    } else if (variant == "guo_prescribed_state") {
        controls.variant = Variant::GuoPrescribedState;
    } else {
        result.errors.push_back(
            "poiseuille_flow.variant must be hbb_velocity, guo_velocity, guo_pressure or guo_prescribed_state");
    }
    poiseuille.query("intervals", controls.intervals);
    poiseuille.query("depth_cells", controls.depthCells);
    poiseuille.query("rho_lb", controls.rhoLb);
    poiseuille.query("u_max_lb", controls.uMaxLb);
    poiseuille.query("u_ref_phys", controls.uRefPhys);
    poiseuille.query("tau", controls.tau);
    poiseuille.query("rho_in_lb", controls.rhoInLb);
    poiseuille.query("rho_out_lb", controls.rhoOutLb);
    poiseuille.query("pressure_scaling", controls.pressureScaling);

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
    amrex::ParmParse validation("poiseuille_validation");
    validation.query("write_profile", controls.observation.writeProfile);
    validation.query("assert", controls.observation.profileAssert);
    validation.query("mass_assert", controls.observation.massAssert);
    validation.query(
        "e2_profile_tolerance", controls.observation.e2ProfileTolerance);
    validation.query(
        "einf_profile_tolerance", controls.observation.eInfProfileTolerance);
    validation.query(
        "transverse_tolerance", controls.observation.transverseTolerance);
    validation.query("mass_tolerance", controls.observation.massTolerance);
    validation.query(
        "pressure_boundary_rho_tolerance",
        controls.observation.pressureBoundaryRhoTolerance);

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
        "Poiseuille intervals, depth_cells and max_grid_size must be positive");
    add(
        !std::isfinite(controls.rhoLb) || !(controls.rhoLb > 0.0) ||
            !std::isfinite(controls.uMaxLb) || !(controls.uMaxLb > 0.0) ||
            !std::isfinite(controls.uRefPhys) || !(controls.uRefPhys > 0.0) ||
            !std::isfinite(controls.tau) || !(controls.tau > 0.5),
        "Poiseuille rho_lb, u_max_lb, u_ref_phys and tau are invalid");
    add(
        density_driven(controls.variant) &&
            (!std::isfinite(controls.rhoInLb) ||
             !std::isfinite(controls.rhoOutLb) ||
             !(controls.rhoInLb > controls.rhoOutLb) ||
             std::abs(
                 0.5 * (controls.rhoInLb + controls.rhoOutLb) -
                 controls.rhoLb) > 1.0e-12),
        "density-driven Poiseuille requires finite rho_in_lb > rho_out_lb symmetric around rho_lb");
    add(
        controls.pressureScaling != "fixed_re_fixed_tau" &&
            controls.pressureScaling != "fixed_mach_characterization",
        "Poiseuille pressure_scaling must be fixed_re_fixed_tau or fixed_mach_characterization");
    add(
        controls.maxStep <= 0 || controls.diagnosticsInterval < 0 ||
            controls.steadyCheckInterval < 0 ||
            !std::isfinite(controls.steadyL2Tolerance) ||
            !(controls.steadyL2Tolerance > 0.0) ||
            controls.plotInterval < 0 ||
            !std::isfinite(controls.finalTimePhys) ||
            !(controls.finalTimePhys > 0.0),
        "Poiseuille run and steady controls are invalid");
    auto flag = [](int value) { return value == 0 || value == 1; };
    add(
        !flag(controls.stopOnSteady) ||
            !flag(controls.writeFinalPlotfile) ||
            !valid_observation_config(controls.observation),
        "Poiseuille boolean controls or acceptance tolerances are invalid");
    add(
        controls.outputDirectory.empty() || controls.logDirectory.empty() ||
            controls.runKind.empty() || controls.runVariant.empty(),
        "Poiseuille output paths and run identity must not be empty");
    return result;
}

} // namespace lbm::cases::poiseuille_flow

#endif
