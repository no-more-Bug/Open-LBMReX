#ifndef LBM_CONFIG_LBMCONFIG_H
#define LBM_CONFIG_LBMCONFIG_H

#include "input/ParsedInputValue.H"
#include "lbm/config/LbmConfigIntent.H"

#include <AMReX_REAL.H>

#include <string>

namespace lbm::config {

/**
 * @brief Syntax-level LBM values before numerical closure.
 *
 * Every scalar retains whether the user supplied its canonical key. No field
 * here has been inferred from case or physical-boundary actual values. The
 * selected parameterization is mandatory even though its full
 * required/forbidden matrix is enforced by the later LbmConfigResolver.
 */
struct RawLbmConfig {
    input::ParsedInputValue<LbmParameterizationKind> parameterizationKind;
    input::ParsedInputValue<amrex::Real> rhoRefLb;
    input::ParsedInputValue<amrex::Real> uRefLb;
    input::ParsedInputValue<amrex::Real> uRefPhys;
    input::ParsedInputValue<amrex::Real> lengthRefLb;
    input::ParsedInputValue<amrex::Real> lengthRefPhys;
    input::ParsedInputValue<amrex::Real> nuLb;
    input::ParsedInputValue<amrex::Real> reynolds;
    input::ParsedInputValue<amrex::Real> tau;
    input::ParsedInputValue<std::string> streamingBackend;
};

/**
 * @brief Observable provenance retained after LBM numerical closure.
 *
 * The flags distinguish explicit reference length from values derived by the
 * selected strategy. Per-key raw input sources do not outlive parsing because
 * no post-closure consumer observes them.
 */
struct LbmInputSourceSummary {
    bool lengthReferenceExplicit = false;
    bool uRefLbDerived = false;
    bool nuLbDerived = false;
    bool reynoldsDerived = false;
    bool tauDerived = false;
};

/**
 * @brief Fully closed LBM reference and BGK numerical state.
 *
 * This value is intentionally ParmParse-, application-, and boundary-free.
 * LbmConfigResolver closes the primary inputs; AMR preparation converts those
 * facts once into each level's local lattice units. Downstream consumers only
 * inspect the resulting strategy, provenance and canonical values.
 */
struct ResolvedLbmConfig {
    LbmParameterizationKind parameterizationKind =
        LbmParameterizationKind::FromNuLb;
    amrex::Real rhoRefLb = 1.0;
    amrex::Real uRefLb = 0.01;
    amrex::Real uRefPhys = 1.0;
    amrex::Real lengthRefLb = -1.0;
    amrex::Real lengthRefPhys = -1.0;
    amrex::Real nuLb = 0.01;
    amrex::Real reynolds = 10.0;
    amrex::Real tau = 0.75;
    amrex::Real omega = 1.0;
    std::string streamingBackend = "baseline";
    LbmInputSourceSummary sources;
};

} // namespace lbm::config

#endif
