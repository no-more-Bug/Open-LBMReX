#ifndef LBM_TESTS_PERFORMANCE_AMRPERFORMANCECASES_H
#define LBM_TESTS_PERFORMANCE_AMRPERFORMANCECASES_H

#include "numerical/amr_taylor_green_vortex/src/TgvCase.H"
#include "numerical/amr_channel_cylinder/src/Re100Case.H"
#include "lbm/config/LbmConfigIntent.H"

namespace lbm::tests::performance {

// Reuse only scientific composition, without its sampling or acceptance policy.
#ifdef LBM_PERFORMANCE_DFG
inline constexpr char amrCaseName[] = "dfg";
inline constexpr int amrMaxGridSize = 64;
inline auto amr_initial() { return amr_re100::Initial{}; }
inline auto amr_geometry(int dimensions, bool uniform)
{ return amr_re100::geometry(dimensions, uniform ? "fine" : "amr", amrMaxGridSize); }
inline auto amr_boundaries()
{
    auto boundaries = amr_re100::boundaries("yms");
    boundaries.collectForces = false;
    return boundaries;
}
inline config::LbmConfigIntent amr_intent(amrex::Real finestSpacing)
{
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromReynolds;
    intent.reynolds = 100;
    intent.uRefLb = amr_re100::meanSpeed;
    intent.uRefPhys = 1;
    intent.rhoRefLb = 1;
    intent.lengthRefLb = amrex::Real(0.1)/finestSpacing;
    intent.lengthRefPhys = 0.1;
    return intent;
}
#else
inline constexpr char amrCaseName[] = "tgv";
inline constexpr int amrMaxGridSize = 16;
inline auto amr_initial() { return amr_tgv::Initial{amrex::Real(5)/(6*160)}; }
inline auto amr_geometry(int dimensions, bool uniform)
{ return amr_tgv::geometry(dimensions, 80, uniform, amrMaxGridSize); }
inline amr::AmrBoundarySpec amr_boundaries() { return {}; }
inline config::LbmConfigIntent amr_intent(amrex::Real)
{
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    intent.nuLb = amrex::Real(1)/12;
    intent.uRefLb = amrex::Real(5)/(6*160);
    intent.uRefPhys = 1;
    intent.rhoRefLb = 1;
    return intent;
}
#endif
} // namespace lbm::tests::performance
#endif
