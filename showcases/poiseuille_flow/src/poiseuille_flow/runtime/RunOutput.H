#ifndef LBM_CASES_POISEUILLEFLOW_RUNOUTPUT_H
#define LBM_CASES_POISEUILLEFLOW_RUNOUTPUT_H

#include "poiseuille_flow/config/Controls.H"
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

namespace lbm::cases::poiseuille_flow {

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
                error = "Failed to prepare direct Poiseuille output: ";
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
        std::string const& observationSummary)
    {
        std::ostringstream os;
        os << std::setprecision(12)
           << "\n=== Direct Poiseuille flow ===\n"
           << "descriptor = " << Descriptor::name() << '\n'
           << "case_name = poiseuille_flow\n"
           << "case_variant = " << controls_.runVariant << '\n'
           << "condition = " << condition() << '\n'
           << "scheme = "
           << (uses_guo(controls_.variant) ?
                   "guo_extrapolation" : "halfway_bounce_back") << '\n'
           << "profile = " << domainProfile() << '\n'
           << "initialization = poiseuille\n"
           << "location_source = xlo/xhi drive and ylo/yhi wall case composition\n"
           << "valid_nodes = (" << context.geometry.Domain().length(0)
           << ", " << context.geometry.Domain().length(1)
           << ", " << context.geometry.Domain().length(2) << ")\n"
           << "physical_intervals = (" << controls_.intervals[0]
           << ", " << controls_.intervals[1] << ")\n"
           << "AMReX envelope prob_lo = (" << context.geometry.ProbLo(0)
           << ", " << context.geometry.ProbLo(1)
           << ", " << context.geometry.ProbLo(2) << ")\n"
           << "AMReX envelope prob_hi = (" << context.geometry.ProbHi(0)
           << ", " << context.geometry.ProbHi(1)
           << ", " << context.geometry.ProbHi(2) << ")\n"
           << "rho_ref_lb = " << lbm_.rhoRefLb << '\n'
           << "u_ref_lb = " << lbm_.uRefLb << '\n'
           << "nu_lb = " << lbm_.nuLb << '\n'
           << "tau = " << lbm_.tau << '\n'
           << "omega = " << lbm_.omega << '\n'
           << "max_step = " << controls_.maxStep << '\n'
           << "diagnostics_interval = " << controls_.diagnosticsInterval << '\n'
           << "steady_check_interval = " << controls_.steadyCheckInterval << '\n'
           << "steady_l2_tolerance = " << controls_.steadyL2Tolerance << '\n'
           << "stop_on_steady = " << controls_.stopOnSteady << '\n'
           << "plot_interval = " << controls_.plotInterval << '\n'
           << "output_directory = " << controls_.outputDirectory << '\n'
           << "terminal_log_directory = " << controls_.logDirectory
           << observationSummary
           << "\n===============================\n\n";
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
        runtime::write_diagnostics_row(diagnosticsCsvPath(), row, resetFile);
    }

    void writeConvergenceRow(runtime::DiagnosticRow const&) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            false, "Direct Poiseuille has no convergence-series output");
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
            plotfilePath(step), macro, geometry, boxArray,
            distributionMapping, lbm_.rhoRefLb, lbm_.uRefLb, step,
            physicalTime);
    }

    void writeFinalSummary(
        runtime::DiagnosticRow const& row,
        bool convergenceRowWritten,
        runtime::SingleLevelFinalSummaryView const&)
    {
        std::ostringstream os;
        os << std::setprecision(12)
           << "\n=== Poiseuille final summary ===\n"
           << "final_diagnostic_step = " << row.step << '\n'
           << "final_time_lb = " << row.timeLb << '\n'
           << "final_time_phys = " << row.timePhys << '\n'
           << "final_relative_l2_u = " << row.relativeL2U << '\n'
           << "final_relative_mass_error = " << row.relativeMassError << '\n'
           << "convergence_row_written = "
           << (convergenceRowWritten ? "true" : "false") << '\n'
           << "run_output_dir = " << controls_.outputDirectory << '\n'
           << "LBM run complete: steps=" << row.step
           << ", descriptor=" << Descriptor::name()
           << ", omega=" << lbm_.omega
           << ", nu_lb=" << lbm_.nuLb << '\n'
           << "=================================\n";
        printAndLog(os.str());
    }

private:
    char const* condition() const noexcept
    {
        switch (controls_.variant) {
        case Variant::HbbVelocity:
        case Variant::GuoVelocity:
            return "velocity_inlet -> velocity_outlet";
        case Variant::GuoPressure:
            return "pressure_inlet -> pressure_outlet";
        case Variant::GuoPrescribedState:
            return "prescribed_state -> prescribed_state";
        }
        return "unknown";
    }

    char const* domainProfile() const noexcept
    {
        return controls_.variant == Variant::GuoPressure ?
            "constant_density" : "poiseuille";
    }

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

} // namespace lbm::cases::poiseuille_flow

#endif
