#ifndef LBM_CASES_CHANNELCYLINDER_OBSERVATIONRESULT_H
#define LBM_CASES_CHANNELCYLINDER_OBSERVATIONRESULT_H

#include <string>

namespace lbm::cases::channel_cylinder {

struct BenchmarkObservationResult {
    std::string summary;
    std::string error;
    bool outputFailure = false;

    bool ok() const noexcept { return error.empty(); }
};

} // namespace lbm::cases::channel_cylinder

#endif
