#ifndef LBM_CASES_CHANNELCYLINDER_CONTROLS_H
#define LBM_CASES_CHANNELCYLINDER_CONTROLS_H

#include <AMReX_REAL.H>

#include <string>

namespace lbm::cases::channel_cylinder {

enum class Variant : int {
    SingleYms = 0,
    SingleBouzidi = 1,
    TandemDistinct = 2,
    TandemShared = 3,
    Re20Yms = 4,
    Re100Yms = 5
};

inline char const* to_string(Variant variant) noexcept
{
    switch (variant) {
    case Variant::SingleYms:
        return "single_yms";
    case Variant::SingleBouzidi:
        return "single_bouzidi";
    case Variant::TandemDistinct:
        return "tandem_distinct";
    case Variant::TandemShared:
        return "tandem_shared";
    case Variant::Re20Yms:
        return "re20_yms";
    case Variant::Re100Yms:
        return "re100_yms";
    }
    return "unknown";
}

inline bool is_tandem(Variant variant) noexcept
{
    return variant == Variant::TandemDistinct ||
        variant == Variant::TandemShared;
}

inline bool shares_tandem_object(Variant variant) noexcept
{
    return variant == Variant::TandemShared;
}

inline bool uses_bouzidi(Variant variant) noexcept
{
    return variant == Variant::SingleBouzidi;
}

inline bool uses_uniform_initialization(Variant variant) noexcept
{
    return is_tandem(variant) || variant == Variant::Re20Yms ||
        variant == Variant::Re100Yms;
}

struct Controls {
    Variant variant = Variant::SingleYms;
    int diameterIntervals = 20;
    int depthCells = 1;
    int maxGridSize = 64;
    amrex::Real rhoLb = 1.0;
    amrex::Real uMeanLb = 0.035;
    amrex::Real uMeanPhys = 0.2;
    amrex::Real reynoldsHeight = 82.0;
    int maxStep = 12;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 12;
    int steadyCheckInterval = 0;
    amrex::Real steadyL2Tolerance = 5.0e-8;
    int stopOnSteady = 0;
    int plotInterval = 0;
    int writeFinalPlotfile = 0;
    int forceInterval = 1;
    int outputInterval = 10;
    int writeHistory = 1;
    int assertEnabled = 0;
    std::string runKind = "smoke";
    std::string runVariant = "d2q9_single_yms_nd20_short";
    std::string outputDirectory = "showcases/channel_cylinder/outputs";
    std::string logDirectory = "showcases/channel_cylinder/log";
};


} // namespace lbm::cases::channel_cylinder

#endif
