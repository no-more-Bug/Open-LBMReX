#ifndef LBM_DYNAMICS_BGKDYNAMICSSELECTION_H
#define LBM_DYNAMICS_BGKDYNAMICSSELECTION_H

#include "lbm/config/LbmConfig.H"
#include "lbm/dynamics/BGKDynamics.H"
#include "lbm/dynamics/DynamicsPreparation.H"

namespace lbm {

/** Construct BGK from the common primary-relaxation facts. */
struct BGKDynamicsSelection {
    template <class Descriptor>
    using Dynamics = BGKDynamics<Descriptor>;

    template <class Descriptor>
    DynamicsPreparationResult<Dynamics<Descriptor>> prepare(
        config::ResolvedLbmConfig const& lbm) const
    {
        DynamicsPreparationResult<Dynamics<Descriptor>> result;
        result.dynamics.emplace(lbm.omega);
        return result;
    }
};

} // namespace lbm

#endif
