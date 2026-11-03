#ifndef LBM_TESTS_REGRESSION_COMMON_SHORTRUNSUPPORT_H
#define LBM_TESTS_REGRESSION_COMMON_SHORTRUNSUPPORT_H

#include "lbm/fields/MacroField.H"
#include "runtime/SingleLevelDiagnosticOutput.H"
#include "runtime/SingleLevelDiagnosticsConfig.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::regression {

struct ShortOutputConfig {
    std::string directory;
    std::string logDirectory;
    std::string errorPrefix;
    std::string identity;
};

class ShortRunOutput {
public:
    explicit ShortRunOutput(ShortOutputConfig config)
        : config_(std::move(config))
    {}

    std::string prepare()
    {
        std::string error;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                std::filesystem::create_directories(config_.directory);
                std::filesystem::create_directories(config_.logDirectory);
                std::filesystem::remove(diagnostics_path());
            } catch (std::filesystem::filesystem_error const& exception) {
                error = config_.errorPrefix + ": " + exception.what();
            }
        }
        amrex::BroadcastString(
            error,
            amrex::ParallelDescriptor::MyProc(),
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());
        return error;
    }

    std::string const& directory() const noexcept { return config_.directory; }

    void print_identity() const
    {
        amrex::Print() << config_.identity;
    }

    void printAndLog(std::string const& text)
    {
        amrex::Print() << text;
    }

    void writeDiagnosticObservation(runtime::DiagnosticRow const& row)
    {
        lastRow_ = row;
        amrex::Print() << runtime::diagnostic_console_text(row);
    }

    runtime::DiagnosticRow const& last_row() const noexcept
    {
        return lastRow_;
    }

    void writeDiagnosticsRow(
        runtime::DiagnosticRow const& row,
        bool resetFile) const
    {
        runtime::write_diagnostics_row(diagnostics_path(), row, resetFile);
    }

    void writeConvergenceRow(runtime::DiagnosticRow const&) const noexcept {}

    void writePlotfile(
        MacroField const&,
        amrex::Geometry const&,
        amrex::BoxArray const&,
        amrex::DistributionMapping const&,
        int,
        amrex::Real) const noexcept
    {}

    void writeFinalSummary(
        runtime::DiagnosticRow const&,
        bool,
        runtime::SingleLevelFinalSummaryView const&) const noexcept
    {}

private:
    std::string diagnostics_path() const
    {
        return config_.directory + "/diagnostics.csv";
    }

    ShortOutputConfig config_;
    runtime::DiagnosticRow lastRow_;
};

inline std::string format_setup_diagnostics(
    char const* heading,
    std::vector<runtime::SingleLevelSetupDiagnostic> const& diagnostics)
{
    std::ostringstream stream;
    stream << heading;
    for (auto const& diagnostic : diagnostics) {
        stream << "\n  - [" << runtime::to_string(diagnostic.stage)
               << ':' << diagnostic.code << ':' << diagnostic.subject
               << "] " << diagnostic.message;
    }
    return stream.str();
}

template <class Program>
int execute_short_program(Program program, char const* setupHeading)
{
    auto preparation = runtime::prepare_single_level(std::move(program));
    if (!preparation.ok()) {
        amrex::Print() << format_setup_diagnostics(
            setupHeading, preparation.diagnostics()) << '\n';
        return 3;
    }
    auto report = runtime::run_single_level(std::move(preparation).take());
    if (!report.ok()) {
        amrex::Print() << report.message << '\n';
        return 4;
    }
    return 0;
}

inline std::string direct_output_directory()
{
    std::string value = "outputs";
    amrex::ParmParse("output").query("directory", value);
    return value;
}

inline std::string direct_log_directory()
{
    std::string value = "logs";
    amrex::ParmParse("output").query("log_directory", value);
    return value;
}

} // namespace lbm::tests::regression

#endif
