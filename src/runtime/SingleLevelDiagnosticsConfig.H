#ifndef LBM_RUNTIME_SINGLELEVELDIAGNOSTICSCONFIG_H
#define LBM_RUNTIME_SINGLELEVELDIAGNOSTICSCONFIG_H

#include <AMReX_REAL.H>

namespace lbm::runtime {

struct SingleLevelDiagnosticsConfig {
    int maxStep = 100;
    int interval = 10;
    int plotInterval = 100;
    int writeFinalPlotfile = 0;
    int convergenceEnabled = 0;
};

struct SingleLevelSteadyObservationPolicy {
    bool enabled = false;
    char const* label = "steady";
    int checkInterval = 0;
    int stopOnSteady = 0;
    amrex::Real l2Tolerance = 5.0e-8;
};

struct SingleLevelFinalSummaryView {
    char const* steadyMonitorLabel = "steady";
    bool steadyMonitorEnabled = false;
    bool steadyReached = false;
    int steadyReachedStep = -1;
    amrex::Real lastSteadyResidual = -1.0;
};

} // namespace lbm::runtime

#endif
