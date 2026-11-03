#ifndef LBM_CASES_LIDDRIVENCAVITY_RUNOUTPUT_H
#define LBM_CASES_LIDDRIVENCAVITY_RUNOUTPUT_H

#include "lid_driven_cavity/config/Controls.H"
#include "lbm/config/LbmConfig.H"
#include "lbm/fields/MacroField.H"
#include "runtime/SingleLevelDiagnosticOutput.H"
#include "runtime/SingleLevelDiagnosticsConfig.H"
#include "runtime/SingleLevelPlotfile.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

template <class Descriptor>
class RunOutput {
public:
    RunOutput(Controls controls, config::ResolvedLbmConfig const& lbm)
        : controls_(std::move(controls)), lbm_(lbm)
    {}

    std::string prepare()
    {
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                std::filesystem::create_directories(controls_.outputDirectory);
                std::filesystem::create_directories(controls_.logDirectory);
                std::filesystem::remove(diagnosticsCsvPath());
            } catch (std::filesystem::filesystem_error const& exception) {
                error = "Failed to prepare direct cavity output: ";
                error += exception.what();
            }
        }
        amrex::BroadcastString(
            error,
            amrex::ParallelDescriptor::MyProc(),
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());
        return error;
    }

    std::string const& runOutputDir() const noexcept
    {
        return controls_.outputDirectory;
    }

    void printAndLog(std::string const& text)
    {
        amrex::Print() << text;
    }

    void printRunSummary(
        runtime::SingleLevelRunContext<Descriptor> const& context,
        std::string const& cavitySummary)
    {
        std::ostringstream os;
        os << std::setprecision(12)
           << "\n=== Direct lid-driven cavity ===\n"
           << "descriptor = " << Descriptor::name() << '\n'
           << "run_kind = " << controls_.runKind << '\n'
           << "case_name = lid_driven_cavity\n"
           << "case_variant = " << controls_.runVariant << '\n'
           << "wall_scheme = " << to_string(controls_.wallScheme) << '\n'
           << "valid_nodes_xy = (" << context.geometry.Domain().length(0)
           << ", " << context.geometry.Domain().length(1) << ")\n"
           << "AMReX envelope prob_lo = ("
           << context.geometry.ProbLo(0) << ", "
           << context.geometry.ProbLo(1) << ", "
           << context.geometry.ProbLo(2) << ")\n"
           << "AMReX envelope prob_hi = ("
           << context.geometry.ProbHi(0) << ", "
           << context.geometry.ProbHi(1) << ", "
           << context.geometry.ProbHi(2) << ")\n"
           << "rho_ref_lb = " << lbm_.rhoRefLb << '\n'
           << "u_ref_lb = " << lbm_.uRefLb << '\n'
           << "length_ref_lb = " << lbm_.lengthRefLb << '\n'
           << "length_ref_phys = " << lbm_.lengthRefPhys << '\n'
           << "Re = " << lbm_.reynolds << '\n'
           << "nu_lb = " << lbm_.nuLb << '\n'
           << "tau = " << lbm_.tau << '\n'
           << "omega = " << lbm_.omega << '\n'
           << "max_step = " << controls_.maxStep << '\n'
           << "diagnostics_interval = " << controls_.diagnosticsInterval
           << '\n'
           << "plot_interval = " << controls_.plotInterval << '\n'
           << "output_directory = " << controls_.outputDirectory << '\n'
           << "terminal_log_directory = " << controls_.logDirectory
           << cavitySummary
           << "\n==========================================\n\n";
        printAndLog(os.str());
    }

    void writeDiagnosticObservation(runtime::DiagnosticRow const& row)
    {
        amrex::Print() << runtime::diagnostic_console_text(row);
    }

    void writeDiagnosticsRow(
        runtime::DiagnosticRow const& row,
        bool resetFile) const
    {
        runtime::write_diagnostics_row(
            diagnosticsCsvPath(), row, resetFile);
    }

    void writeConvergenceRow(runtime::DiagnosticRow const&) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            false,
            "Direct cavity does not support convergence-series output");
    }

    void writePlotfile(
        MacroField const& macro,
        amrex::Geometry const& geometry,
        amrex::BoxArray const& boxArray,
        amrex::DistributionMapping const& distributionMapping,
        int step,
        amrex::Real physicalTime)
    {
        runtime::write_single_level_plotfile(
            plotfilePath(step),
            macro,
            geometry,
            boxArray,
            distributionMapping,
            lbm_.rhoRefLb,
            lbm_.uRefLb,
            step,
            physicalTime);
    }

    void writeFinalSummary(
        runtime::DiagnosticRow const& row,
        bool convergenceRowWritten,
        runtime::SingleLevelFinalSummaryView const& view)
    {
        std::ostringstream os;
        os << std::setprecision(12)
           << "\n=== LBM final summary ===\n"
           << "final_diagnostic_step = " << row.step << '\n'
           << "final_time_lb = " << row.timeLb << '\n'
           << "final_time_phys = " << row.timePhys << '\n'
           << "final_relative_l2_u = " << row.relativeL2U << '\n'
           << "final_relative_mass_error = " << row.relativeMassError << '\n'
           << "convergence_row_written = "
           << (convergenceRowWritten ? "true" : "false") << '\n'
           << "steady_monitor_label = " << view.steadyMonitorLabel << '\n'
           << "steady_monitor_enabled = "
           << (view.steadyMonitorEnabled ? "true" : "false") << '\n'
           << "steady_reached = "
           << (view.steadyReached ? "true" : "false") << '\n'
           << "steady_reached_step = " << view.steadyReachedStep << '\n'
           << "final_relative_l2_delta_u = "
           << view.lastSteadyResidual << '\n'
           << "run_output_dir = " << controls_.outputDirectory << '\n'
           << "terminal_log_pattern = " << controls_.logDirectory
           << "/run-YYYYMMDD-HHMMSS.log\n"
           << "configured_max_step = " << controls_.maxStep << '\n'
           << "LBM run complete: steps=" << row.step
           << ", descriptor=" << Descriptor::name()
           << ", omega=" << lbm_.omega
           << ", nu_lb=" << lbm_.nuLb << '\n'
           << "=========================\n";
        printAndLog(os.str());
    }

private:
    std::string diagnosticsCsvPath() const
    {
        return controls_.outputDirectory + "/diagnostics.csv";
    }

    std::string plotfilePath(int step) const
    {
        return controls_.outputDirectory + "/" +
            amrex::Concatenate("plt", step, 5);
    }

    Controls controls_;
    config::ResolvedLbmConfig const& lbm_;
};

} // namespace lbm::cases::lid_driven_cavity

#endif
