#ifndef LBM_CASES_POISEUILLEFLOW_CONTROLS_H
#define LBM_CASES_POISEUILLEFLOW_CONTROLS_H

#include "poiseuille_flow/observation/PoiseuilleDiagnostics.H"

#include <AMReX_REAL.H>

#include <array>
#include <string>

namespace lbm::cases::poiseuille_flow {

enum class Variant : int {
    HbbVelocity = 0,
    GuoVelocity = 1,
    GuoPressure = 2,
    GuoPrescribedState = 3
};

inline char const* to_string(Variant variant) noexcept
{
    switch (variant) {
    case Variant::HbbVelocity:
        return "hbb_velocity";
    case Variant::GuoVelocity:
        return "guo_velocity";
    case Variant::GuoPressure:
        return "guo_pressure";
    case Variant::GuoPrescribedState:
        return "guo_prescribed_state";
    }
    return "unknown";
}

inline bool uses_guo(Variant variant) noexcept
{
    return variant != Variant::HbbVelocity;
}

inline bool density_driven(Variant variant) noexcept
{
    return variant == Variant::GuoPressure ||
        variant == Variant::GuoPrescribedState;
}

struct ObservationConfig {
    bool enabled = false;
    int writeProfile = 0;
    int profileAssert = 0;
    int massAssert = 1;
    amrex::Real e2ProfileTolerance = 5.0e-2;
    amrex::Real eInfProfileTolerance = 5.0e-2;
    amrex::Real transverseTolerance = 1.0e-2;
    amrex::Real massTolerance = 1.0e-5;
    amrex::Real pressureBoundaryRhoTolerance = 1.0e-10;
    PoiseuilleDiagnosticGeometry geometry;
    PoiseuillePressureDiagnosticConfig pressure;
    std::string pressureScaling = "fixed_re_fixed_tau";
};

inline bool valid_observation_config(ObservationConfig const& config) noexcept
{
    auto flag = [](int value) { return value == 0 || value == 1; };
    return flag(config.writeProfile) &&
        flag(config.profileAssert) &&
        flag(config.massAssert) &&
        config.e2ProfileTolerance > 0.0 &&
        config.eInfProfileTolerance > 0.0 &&
        config.transverseTolerance > 0.0 &&
        config.massTolerance > 0.0 &&
        config.pressureBoundaryRhoTolerance > 0.0;
}

struct Controls {
    Variant variant = Variant::HbbVelocity;
    std::array<int, 2> intervals{40, 10};
    int depthCells = 1;
    int maxGridSize = 16;
    amrex::Real rhoLb = 1.0;
    amrex::Real uMaxLb = 0.1;
    amrex::Real uRefPhys = 1.0;
    amrex::Real tau = 0.75;
    amrex::Real rhoInLb = 1.04;
    amrex::Real rhoOutLb = 0.96;
    std::string pressureScaling = "fixed_re_fixed_tau";
    int maxStep = 20000;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 100;
    int steadyCheckInterval = 100;
    amrex::Real steadyL2Tolerance = 5.0e-8;
    int stopOnSteady = 1;
    int plotInterval = 20000;
    int writeFinalPlotfile = 1;
    ObservationConfig observation{true, 1, 1};
    std::string runKind = "steady";
    std::string runVariant = "d2q9_hbb_velocity_poiseuille_N10";
    std::string outputDirectory = "showcases/poiseuille_flow/outputs";
    std::string logDirectory = "showcases/poiseuille_flow/log";
};


} // namespace lbm::cases::poiseuille_flow

#endif
