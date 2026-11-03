#ifndef LBM_INITIALIZATION_TAYLORGREENVORTEXPARAMS_H
#define LBM_INITIALIZATION_TAYLORGREENVORTEXPARAMS_H

#include <AMReX_REAL.H>

namespace lbm::initialization {

/**
 * @brief Immutable parameters of the built-in Taylor--Green field.
 *
 * The initialization resolver supplies the actual density, velocity,
 * viscosity, lattice sound speed, wave numbers, and decay factors. The same
 * device-safe value is consumed by initialization and analytical diagnostics;
 * it owns neither AMReX Geometry nor population storage.
 */
struct TaylorGreenVortexParams {
    amrex::Real rho0 = 1.0;
    amrex::Real u0 = 0.01;
    amrex::Real nu = 0.01;
    amrex::Real cs2 = 1.0 / 3.0;
    amrex::Real length = 1.0;
    amrex::Real kx = 0.0;
    amrex::Real ky = 0.0;
    amrex::Real decay = 1.0;
    amrex::Real pressureDecay = 1.0;
};

} // namespace lbm::initialization

#endif
