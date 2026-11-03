#ifndef LBM_CASES_UNIFORMFLOW_CONTROLS_H
#define LBM_CASES_UNIFORMFLOW_CONTROLS_H

#include <AMReX_REAL.H>

#include <array>
#include <string>

namespace lbm::cases::uniform_flow {

enum class Variant : int {
    HbbVelocityIo = 0,
    GuoVelocityIo = 1,
    GuoVelocityFreeOutflow = 2
};

inline char const* to_string(Variant variant) noexcept
{
    switch (variant) {
    case Variant::HbbVelocityIo:
        return "hbb_velocity_io";
    case Variant::GuoVelocityIo:
        return "guo_velocity_io";
    case Variant::GuoVelocityFreeOutflow:
        return "guo_velocity_free_outflow";
    }
    return "unknown";
}

struct ObservationConfig {
    bool enabled = false;
    int writeMetrics = 0;
    int assertEnabled = 0;
    int minFlowThroughs = 100;
    amrex::Real e2UTolerance = 1.0e-8;
    amrex::Real eInfUTolerance = 1.0e-7;
    amrex::Real stdUxTolerance = 1.0e-8;
    amrex::Real transverseTolerance = 1.0e-8;
    amrex::Real massTolerance = 1.0e-8;
};

inline constexpr bool valid_observation_config(
    ObservationConfig const& config) noexcept
{
    return config.minFlowThroughs >= 0 &&
        config.e2UTolerance > 0.0 &&
        config.eInfUTolerance > 0.0 &&
        config.stdUxTolerance > 0.0 &&
        config.transverseTolerance > 0.0 &&
        config.massTolerance > 0.0;
}

struct Controls {
    Variant variant = Variant::HbbVelocityIo;
    std::array<int, 2> intervals{10, 10};
    int depthCells = 1;
    int maxGridSize = 10;
    amrex::Real rhoLb = 1.0;
    amrex::Real velocityLb = 0.1;
    amrex::Real velocityReferencePhys = 1.0;
    amrex::Real nuLb = 1.0 / 12.0;
    int maxStep = 10000;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 10000;
    int plotInterval = 10000;
    int writeFinalPlotfile = 0;
    ObservationConfig observation{true, 1, 1};
    std::string runKind = "long-run";
    std::string runVariant = "d2q9_hbb_velocity_io";
    std::string outputDirectory = "showcases/uniform_flow/outputs";
    std::string logDirectory = "showcases/uniform_flow/log";
};


} // namespace lbm::cases::uniform_flow

#endif
