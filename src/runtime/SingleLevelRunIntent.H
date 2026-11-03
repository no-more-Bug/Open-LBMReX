#ifndef LBM_RUNTIME_SINGLELEVELRUNINTENT_H
#define LBM_RUNTIME_SINGLELEVELRUNINTENT_H

#include <AMReX_REAL.H>

#include <string>

namespace lbm::runtime {

/** Parser-free run-control intent. */
struct SingleLevelRunIntent {
    int maxStep = 100;
    amrex::Real finalTimePhys = 1.0;
    std::string kind = "manual";
    std::string name;
    std::string variant;
    int convergenceEnabled = 0;
    int convergenceReset = 0;
};

} // namespace lbm::runtime

#endif
