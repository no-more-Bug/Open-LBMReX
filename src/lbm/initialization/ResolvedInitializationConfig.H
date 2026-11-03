#ifndef LBM_INITIALIZATION_RESOLVEDINITIALIZATIONCONFIG_H
#define LBM_INITIALIZATION_RESOLVEDINITIALIZATIONCONFIG_H

#include "lbm/initialization/InitializationTypes.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/PoiseuilleInitializationParams.H"
#include "lbm/initialization/TaylorGreenVortexParams.H"

namespace lbm::initialization {

/**
 * @brief Fully resolved bulk-fluid initialization consumed by the solver.
 *
 * Application parsing and semantic validation are complete before this value
 * reaches the runtime. The selected branch contains actual initial rho/u and
 * all geometry/model data needed to construct a device-safe profile. It owns
 * neither boundary values nor AMReX field storage.
 */
struct ResolvedInitializationConfig {
    InitializationKind kind = InitializationKind::Rest;
    MacroscopicState constantState;
    TaylorGreenVortexParams taylorGreenParams;
    PoiseuilleInitializationParams poiseuilleParams;
};

} // namespace lbm::initialization

#endif
