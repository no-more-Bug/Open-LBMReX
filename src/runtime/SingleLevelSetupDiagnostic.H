#ifndef LBM_RUNTIME_SINGLELEVELSETUPDIAGNOSTIC_H
#define LBM_RUNTIME_SINGLELEVELSETUPDIAGNOSTIC_H

#include <string>
#include <vector>

namespace lbm::runtime {

enum class SingleLevelSetupStage {
    Intent,
    Mesh,
    LbmClosure,
    RunControl,
    Initialization,
    PhysicalBoundary,
    CaseValidation,
    RuntimePreparation
};

inline char const* to_string(SingleLevelSetupStage stage) noexcept
{
    switch (stage) {
    case SingleLevelSetupStage::Intent:
        return "intent";
    case SingleLevelSetupStage::Mesh:
        return "mesh";
    case SingleLevelSetupStage::LbmClosure:
        return "lbm_closure";
    case SingleLevelSetupStage::RunControl:
        return "run_control";
    case SingleLevelSetupStage::Initialization:
        return "initialization";
    case SingleLevelSetupStage::PhysicalBoundary:
        return "physical_boundary";
    case SingleLevelSetupStage::CaseValidation:
        return "case_validation";
    case SingleLevelSetupStage::RuntimePreparation:
        return "runtime_preparation";
    }
    return "unknown";
}

struct SingleLevelSetupDiagnostic {
    SingleLevelSetupStage stage = SingleLevelSetupStage::Intent;
    std::string code;
    std::string subject;
    std::string message;
};

/** Case-owned diagnostic payload; runtime fixes its semantic stage and order. */
struct SingleLevelCaseDiagnostic {
    std::string code;
    std::string subject;
    std::string message;
};

template <class RunSession>
struct SingleLevelCasePreparationResult {
    RunSession session;
    std::vector<SingleLevelCaseDiagnostic> diagnostics;
};

} // namespace lbm::runtime

#endif
