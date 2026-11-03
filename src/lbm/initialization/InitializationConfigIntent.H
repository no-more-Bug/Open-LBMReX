#ifndef LBM_INITIALIZATION_INITIALIZATIONCONFIGINTENT_H
#define LBM_INITIALIZATION_INITIALIZATIONCONFIGINTENT_H

#include "lbm/initialization/InitializationTypes.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <optional>

namespace lbm::initialization {

/** Parser-free bulk-fluid initialization intent. */
struct InitializationConfigIntent {
    std::optional<InitializationKind> kind;
    std::optional<amrex::Real> rhoLb;
    std::optional<amrex::Array<amrex::Real, 3>> uniformULb;
    std::optional<amrex::Real> taylorGreenVelocityAmplitudeLb;
    std::optional<amrex::Real> poiseuilleUMaxLb;
    std::optional<amrex::Real> poiseuilleRhoXLoLb;
    std::optional<amrex::Real> poiseuilleRhoXHiLb;
};

} // namespace lbm::initialization

#endif
