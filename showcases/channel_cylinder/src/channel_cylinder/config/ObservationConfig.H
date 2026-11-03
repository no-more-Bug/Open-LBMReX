#ifndef LBM_CASES_CHANNELCYLINDER_OBSERVATIONCONFIG_H
#define LBM_CASES_CHANNELCYLINDER_OBSERVATIONCONFIG_H

#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

namespace lbm::cases::channel_cylinder {

/** @brief One resolved object consumed by the channel-cylinder observer. */
struct ChannelCylinderObservedObject {
    std::string name;
    int objectId = -1;
};

/** @brief Runtime values needed by the channel-cylinder observer. */
struct ChannelCylinderRuntimeConfig {
    bool enabled = false;
    std::string referenceLabel;
    std::vector<ChannelCylinderObservedObject> observedObjects;
    int forceInterval = 1;
    int outputInterval = 10;
    int writeHistory = 1;
    int assertEnabled = 1;
    amrex::Real diameterLb = 0.0;
    amrex::Real uMeanLb = 0.0;
    amrex::Real spanLb = 1.0;
    amrex::IntVect pressureFrontCell{AMREX_D_DECL(0, 0, 0)};
    amrex::IntVect pressureRearCell{AMREX_D_DECL(0, 0, 0)};
    int pressureSamplesPerPoint = 1;
};

struct ChannelCylinderBenchmarkValidationResult {
    ChannelCylinderRuntimeConfig data;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

} // namespace lbm::cases::channel_cylinder

#endif
