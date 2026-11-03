#ifndef LBM_INITIALIZATION_POISEUILLEINITIALIZATIONPARAMS_H
#define LBM_INITIALIZATION_POISEUILLEINITIALIZATIONPARAMS_H

#include <AMReX_REAL.H>

namespace lbm::initialization {

/**
 * @brief Immutable actual values for one Poiseuille bulk initialization.
 *
 * linearDensity selects whether rho is interpolated from the two streamwise
 * endpoint values. Fixed mathematical coordinates remain profile data; this
 * parameter value owns neither AMReX Geometry nor boundary configuration.
 */
struct PoiseuilleInitializationParams {
    amrex::Real rhoLb = 1.0;
    amrex::Real uMaxLb = 0.1;
    int linearDensity = 0;
    amrex::Real rhoXLoLb = 1.0;
    amrex::Real rhoXHiLb = 1.0;
};

} // namespace lbm::initialization

#endif
