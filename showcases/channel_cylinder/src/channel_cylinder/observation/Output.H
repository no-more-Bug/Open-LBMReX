#ifndef LBM_CASES_CHANNELCYLINDER_OUTPUT_H
#define LBM_CASES_CHANNELCYLINDER_OUTPUT_H

#include <AMReX_REAL.H>

#include <fstream>
#include <string>
#include <vector>

namespace lbm::cases::channel_cylinder {

struct ChannelCylinderHistoryRow {
    int populationStep = 0;
    amrex::Real forceTimeLb = 0.0;
    amrex::Real tStar = 0.0;
    std::string referenceLabel;
    std::string objectName;
    int objectId = -1;
    amrex::Real fxLb = 0.0;
    amrex::Real fyLb = 0.0;
    amrex::Real fzLb = 0.0;
    amrex::Real cd = 0.0;
    amrex::Real cl = 0.0;
};

struct ChannelCylinderMetrics {
    std::string referenceLabel;
    std::string objectName;
    int objectId = -1;
    int sampleCount = 0;
    int finalPopulationStep = -1;
    amrex::Real cdFinal = 0.0;
    amrex::Real clFinal = 0.0;
    amrex::Real cdMin = 0.0;
    amrex::Real cdMax = 0.0;
    amrex::Real cdMidrange = 0.0;
    amrex::Real cdPeakToPeak = 0.0;
    amrex::Real clMin = 0.0;
    amrex::Real clMax = 0.0;
    amrex::Real clMidrange = 0.0;
    amrex::Real clPeakToPeak = 0.0;
    amrex::Real rhoFrontLb = 0.0;
    amrex::Real rhoRearLb = 0.0;
    amrex::Real deltaPLb = 0.0;
    amrex::Real cDeltaP = 0.0;
    std::string gateStatus;
};

/** @brief IO-rank serializer for the two channel-cylinder CSV products. */
class ChannelCylinderOutput {
public:
    ChannelCylinderOutput(bool enabled, bool writeHistory) noexcept;

    ChannelCylinderOutput(ChannelCylinderOutput const&) = delete;
    ChannelCylinderOutput& operator=(ChannelCylinderOutput const&) = delete;

    std::string prepare(std::string const& runOutputDirectory);
    void append(std::vector<ChannelCylinderHistoryRow> const& rows);
    std::string finalize(std::vector<ChannelCylinderMetrics> const& metrics);

    std::string const& historyPath() const noexcept { return historyPath_; }
    std::string const& metricsPath() const noexcept { return metricsPath_; }

private:
    bool enabled_ = false;
    bool writeHistory_ = false;
    std::string historyPath_;
    std::string metricsPath_;
    std::ofstream history_;
    std::string error_;
};

} // namespace lbm::cases::channel_cylinder

#endif
