#ifndef LBM_DYNAMICS_DYNAMICSPREPARATION_H
#define LBM_DYNAMICS_DYNAMICSPREPARATION_H

#include <optional>
#include <string>
#include <vector>

namespace lbm {

/** Host-side result for constructing one concrete local dynamics value. */
template <class Dynamics>
struct DynamicsPreparationResult {
    std::optional<Dynamics> dynamics;
    std::vector<std::string> errors;

    bool ok() const noexcept
    {
        return dynamics.has_value() && errors.empty();
    }
};

} // namespace lbm

#endif
