#include "channel_cylinder/observation/Output.H"

#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <filesystem>
#include <iomanip>
#include <limits>

namespace lbm::cases::channel_cylinder {

namespace {

constexpr char historyHeader[] =
    "population_step,force_time_lb,t_star,reference_label,object_name,"
    "object_id,Fx_lb,Fy_lb,Fz_lb,Cd,Cl\n";

constexpr char metricsHeader[] =
    "reference_label,object_name,object_id,sample_count,"
    "final_population_step,Cd_final,Cl_final,"
    "Cd_min,Cd_max,Cd_midrange,Cd_peak_to_peak,"
    "Cl_min,Cl_max,Cl_midrange,Cl_peak_to_peak,"
    "rho_front_lb,rho_rear_lb,delta_p_lb,C_delta_p,gate_status\n";

} // namespace

ChannelCylinderOutput::ChannelCylinderOutput(
    bool enabled,
    bool writeHistory) noexcept
    : enabled_(enabled), writeHistory_(writeHistory)
{}

std::string ChannelCylinderOutput::prepare(
    std::string const& runOutputDirectory)
{
    if (!enabled_) {
        return {};
    }
    historyPath_ =
        runOutputDirectory + "/channel_cylinder_force_history.csv";
    metricsPath_ =
        runOutputDirectory + "/channel_cylinder_metrics.csv";
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            std::filesystem::remove(metricsPath_);
            if (!writeHistory_) {
                std::filesystem::remove(historyPath_);
            } else {
                history_.open(historyPath_, std::ios::out | std::ios::trunc);
                if (!history_.is_open()) {
                    error_ = "Failed to open channel-cylinder history CSV";
                } else {
                    history_ << historyHeader;
                    if (!history_) {
                        error_ =
                            "Failed to write channel-cylinder history CSV";
                    }
                }
            }
        } catch (std::filesystem::filesystem_error const& exception) {
            error_ = "Failed to prepare channel-cylinder output: ";
            error_ += exception.what();
        }
    }
    amrex::BroadcastString(
        error_,
        amrex::ParallelDescriptor::MyProc(),
        amrex::ParallelDescriptor::IOProcessorNumber(),
        amrex::ParallelDescriptor::Communicator());
    return error_;
}

void ChannelCylinderOutput::append(
    std::vector<ChannelCylinderHistoryRow> const& rows)
{
    if (!enabled_ || !writeHistory_ ||
        !amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }
    if (!error_.empty()) {
        return;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        history_.is_open(),
        "Channel-cylinder output append called before prepare");
    history_ << std::setprecision(std::numeric_limits<amrex::Real>::max_digits10);
    for (ChannelCylinderHistoryRow const& row : rows) {
        history_ << row.populationStep << ','
                 << row.forceTimeLb << ','
                 << row.tStar << ','
                 << row.referenceLabel << ','
                 << row.objectName << ','
                 << row.objectId << ','
                 << row.fxLb << ','
                 << row.fyLb << ','
                 << row.fzLb << ','
                 << row.cd << ','
                 << row.cl << '\n';
    }
    if (!history_) {
        error_ = "Failed to append channel-cylinder history CSV";
    }
}

std::string ChannelCylinderOutput::finalize(
    std::vector<ChannelCylinderMetrics> const& metrics)
{
    if (!enabled_) {
        return {};
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (history_.is_open()) {
            history_.flush();
            if (!history_ && error_.empty()) {
                error_ = "Failed to flush channel-cylinder history CSV";
            }
            history_.close();
        }
        if (error_.empty()) {
            std::ofstream output(
                metricsPath_, std::ios::out | std::ios::trunc);
            if (!output.is_open()) {
                error_ = "Failed to open channel-cylinder metrics CSV";
            } else {
                output
                    << metricsHeader
                    << std::setprecision(
                           std::numeric_limits<amrex::Real>::max_digits10);
                for (ChannelCylinderMetrics const& row : metrics) {
                    output << row.referenceLabel << ','
                           << row.objectName << ','
                           << row.objectId << ','
                           << row.sampleCount << ','
                           << row.finalPopulationStep << ','
                           << row.cdFinal << ','
                           << row.clFinal << ','
                           << row.cdMin << ','
                           << row.cdMax << ','
                           << row.cdMidrange << ','
                           << row.cdPeakToPeak << ','
                           << row.clMin << ','
                           << row.clMax << ','
                           << row.clMidrange << ','
                           << row.clPeakToPeak << ','
                           << row.rhoFrontLb << ','
                           << row.rhoRearLb << ','
                           << row.deltaPLb << ','
                           << row.cDeltaP << ','
                           << row.gateStatus << '\n';
                }
                if (!output) {
                    error_ =
                        "Failed to write channel-cylinder metrics CSV";
                }
            }
        }
    }
    amrex::BroadcastString(
        error_,
        amrex::ParallelDescriptor::MyProc(),
        amrex::ParallelDescriptor::IOProcessorNumber(),
        amrex::ParallelDescriptor::Communicator());
    return error_;
}

} // namespace lbm::cases::channel_cylinder
