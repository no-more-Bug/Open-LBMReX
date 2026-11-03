#ifndef LBM_CASES_LIDDRIVENCAVITY_CONTROLS_H
#define LBM_CASES_LIDDRIVENCAVITY_CONTROLS_H

#include <AMReX_REAL.H>

#include <array>
#include <string>

namespace lbm::cases::lid_driven_cavity {

enum class WallScheme {
    HalfwayBounceBack,
    GuoExtrapolation
};

inline char const* to_string(WallScheme scheme) noexcept
{
    return scheme == WallScheme::HalfwayBounceBack ?
        "halfway_bounce_back" : "guo_extrapolation";
}

struct Controls {
    std::array<amrex::Real, 2> physicalLo{0.0, 0.0};
    std::array<amrex::Real, 2> physicalHi{1.0, 1.0};
    std::array<int, 2> intervals{128, 128};
    int depthCells = 1;
    WallScheme wallScheme = WallScheme::GuoExtrapolation;
    amrex::Real rhoLb = 1.0;
    amrex::Real reynolds = 1000.0;
    amrex::Real lidVelocityLb = 0.1;
    amrex::Real velocityReferencePhys = 1.0;
    int maxGridSize = 32;
    int maxStep = 400000;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 5000;
    int plotInterval = 400000;
    int writeFinalPlotfile = 1;
    int steadyCheckInterval = 5000;
    amrex::Real steadyL2Tolerance = 5.0e-8;
    int stopOnSteady = 1;
    int writeCenterlines = 1;
    int writeMetrics = 1;
    std::string outputDirectory =
        "showcases/lid_driven_cavity/outputs";
    std::string logDirectory =
        "showcases/lid_driven_cavity/log";
    std::string runKind = "validation";
    std::string runVariant = "d2q9_guo_re1000_ghia";
};


} // namespace lbm::cases::lid_driven_cavity

#endif
