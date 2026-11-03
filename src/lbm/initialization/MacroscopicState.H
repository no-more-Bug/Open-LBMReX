#ifndef LBM_INITIALIZATION_MACROSCOPICSTATE_H
#define LBM_INITIALIZATION_MACROSCOPICSTATE_H

#include <AMReX_REAL.H>

namespace lbm::initialization {

struct MacroscopicState {
    amrex::Real rho = 1.0;
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
};

} // namespace lbm::initialization

#endif
