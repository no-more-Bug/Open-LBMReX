#ifndef LBM_RUNTIME_LBMAMRPREPARATION_H
#define LBM_RUNTIME_LBMAMRPREPARATION_H

#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/operators/level/CollisionOperator.H"

#include <optional>

namespace lbm::runtime {

template<class Descriptor, class Selection>
class LbmAmrEvolutionPreparation {
public:
    using Collision = operators::CollisionOperator<Descriptor,
        typename Selection::template Dynamics<Descriptor>>;
    using Evolution = amr::PreparedAmrEvolution<Collision>;

    std::vector<std::string> errors;
    bool ok() const noexcept { return evolution_.has_value() && errors.empty(); }
    Evolution take() &&
    {
        auto result = std::move(*evolution_);
        evolution_.reset();
        return result;
    }

private:
    std::optional<Evolution> evolution_;
    template<class D, class P, class T, class S>
    friend LbmAmrEvolutionPreparation<D, S> prepare_amr_evolution(
        amr::LbmAmrSimulation<D, P, T> const&, config::LbmConfigIntent const&, S const&);
};

} // namespace lbm::runtime

#include "runtime/LbmAmrPreparation.HH"
#endif
