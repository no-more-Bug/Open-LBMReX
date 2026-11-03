#ifndef LBM_RUNTIME_SINGLELEVELRUNCONTROL_H
#define LBM_RUNTIME_SINGLELEVELRUNCONTROL_H

#include "lbm/config/LbmConfig.H"
#include "runtime/SingleLevelRunIntent.H"

#include <AMReX_REAL.H>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace lbm::runtime {

struct ResolvedSingleLevelRunControl {
    int maxStep = 100;
    amrex::Real finalTimePhys = 1.0;
    std::string kind = "manual";
    std::string name;
    std::string variant;
    int convergenceEnabled = 0;
    int convergenceReset = 0;
    int convergenceFixedTau = 0;
};

struct SingleLevelRunControlResolveResult {
    ResolvedSingleLevelRunControl resolved;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline SingleLevelRunControlResolveResult resolve_single_level_run_control(
    SingleLevelRunIntent const& intent,
    config::ResolvedLbmConfig const& lbm)
{
    SingleLevelRunControlResolveResult result;
    ResolvedSingleLevelRunControl& resolved = result.resolved;
    resolved.maxStep = intent.maxStep;
    resolved.finalTimePhys = intent.finalTimePhys;
    resolved.kind = intent.kind;
    resolved.name = intent.name;
    resolved.variant = intent.variant;
    resolved.convergenceEnabled = intent.convergenceEnabled;
    resolved.convergenceReset = intent.convergenceReset;

    if (resolved.convergenceEnabled != 0 &&
        resolved.convergenceEnabled != 1) {
        result.errors.push_back(
            "convergence enabled flag must be 0 or 1");
    }
    if (resolved.convergenceReset != 0 && resolved.convergenceReset != 1) {
        result.errors.push_back(
            "convergence reset flag must be 0 or 1");
    }
    if (!std::isfinite(resolved.finalTimePhys) ||
        !(resolved.finalTimePhys > 0.0)) {
        result.errors.push_back(
            "physical final time must be finite and positive");
    }
    if (resolved.kind.empty()) {
        result.errors.push_back("run kind must not be empty");
    }
    if (resolved.convergenceEnabled == 0) {
        if (resolved.maxStep <= 0) {
            result.errors.push_back(
                "maximum step count must be positive for a direct run");
        }
        return result;
    }

    using Kind = config::LbmParameterizationKind;
    if (lbm.parameterizationKind != Kind::FromTau &&
        lbm.parameterizationKind != Kind::FromReynolds) {
        result.errors.push_back(
            "convergence runs require lbm.parameterization.kind=from_tau "
            "or from_reynolds");
        return result;
    }
    resolved.kind = "convergence";
    resolved.convergenceFixedTau =
        lbm.parameterizationKind == Kind::FromTau ? 1 : 0;

    const amrex::Real dxPhys = lbm.lengthRefPhys / lbm.lengthRefLb;
    const amrex::Real dtPhys = lbm.uRefLb * dxPhys / lbm.uRefPhys;
    if (!std::isfinite(dtPhys) || !(dtPhys > 0.0)) {
        result.errors.push_back(
            "resolved LBM reference scales produce a non-positive "
            "physical time step");
        return result;
    }
    const amrex::Real stepCount = resolved.finalTimePhys / dtPhys;
    if (!std::isfinite(stepCount) ||
        stepCount > static_cast<amrex::Real>(std::numeric_limits<int>::max())) {
        result.errors.push_back("derived convergence max_step does not fit in int");
        return result;
    }
    resolved.maxStep = static_cast<int>(std::llround(stepCount));
    if (resolved.maxStep < 1) {
        resolved.maxStep = 1;
    }
    return result;
}

} // namespace lbm::runtime

#endif
