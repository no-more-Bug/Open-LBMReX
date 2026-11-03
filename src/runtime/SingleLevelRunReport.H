#ifndef LBM_RUNTIME_SINGLELEVELRUNREPORT_H
#define LBM_RUNTIME_SINGLELEVELRUNREPORT_H

#include <string>

namespace lbm::runtime {

enum class SingleLevelRunStatus {
    Success,
    NumericalFailure,
    OutputFailure
};

/** Observable outcome returned to a composition root after one consumed run. */
struct SingleLevelRunReport {
    SingleLevelRunStatus status = SingleLevelRunStatus::Success;
    int finalStep = 0;
    bool stoppedOnSteady = false;
    std::string message;

    bool ok() const noexcept { return status == SingleLevelRunStatus::Success; }
};

/** Internal fixed-session outcome folded into the public run report. */
struct SingleLevelRunSessionResult {
    SingleLevelRunStatus status = SingleLevelRunStatus::Success;
    std::string message;

    bool ok() const noexcept { return status == SingleLevelRunStatus::Success; }
};

} // namespace lbm::runtime

#endif
