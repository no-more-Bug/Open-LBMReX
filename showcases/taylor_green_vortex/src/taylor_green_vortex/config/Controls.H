#ifndef LBM_CASES_TAYLORGREENVORTEX_CONTROLS_H
#define LBM_CASES_TAYLORGREENVORTEX_CONTROLS_H

#include "lbm/config/LbmConfigIntent.H"

#include <AMReX_REAL.H>

#include <array>
#include <optional>
#include <string>

namespace lbm::cases::taylor_green_vortex {

struct Controls {
    std::array<int, 3> nCell{64, 64, 1};
    int maxGridSize = 32;
    config::LbmParameterizationKind parameterization =
        config::LbmParameterizationKind::FromNuLb;
    amrex::Real rhoLb = 1.0;
    amrex::Real uRefLb = 0.01;
    amrex::Real uRefPhys = 1.0;
    amrex::Real nuLb = 0.01;
    amrex::Real tau = 0.75;
    amrex::Real reynolds = 10.0;
    std::optional<amrex::Real> velocityAmplitudeLb;
    int maxStep = 100;
    amrex::Real finalTimePhys = 1.0;
    int convergenceEnabled = 0;
    int convergenceReset = 0;
    int diagnosticsInterval = 10;
    int plotInterval = 100;
    int writeFinalPlotfile = 1;
    std::string outputDirectory =
        "showcases/taylor_green_vortex/outputs";
    std::string logDirectory =
        "showcases/taylor_green_vortex/log";
    std::string convergenceDirectory = "convergence";
    std::string runKind = "manual";
    std::string runVariant = "d2q9_periodic_2d";
};

} // namespace lbm::cases::taylor_green_vortex

#endif
