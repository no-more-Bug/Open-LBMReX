#ifndef LBM_DYNAMICS_DYNAMICSMODEL_H
#define LBM_DYNAMICS_DYNAMICSMODEL_H

#include <type_traits>

namespace lbm {

template <class Descriptor>
class LocalDynamicsModel {
public:
    using descriptor_type = Descriptor;

    static constexpr bool is_local_dynamics_model = true;

protected:
    LocalDynamicsModel() = default;
};

template <class Descriptor>
class IsoThermalDynamicsModelBase : public LocalDynamicsModel<Descriptor> {
protected:
    IsoThermalDynamicsModelBase() = default;
};

template <class Dynamics, class Descriptor, class = void>
struct is_dynamics_model_for_descriptor : std::false_type {};

template <class Dynamics, class Descriptor>
struct is_dynamics_model_for_descriptor<
    Dynamics,
    Descriptor,
    std::void_t<typename Dynamics::descriptor_type>>
    : std::integral_constant<bool,
          std::is_same<typename Dynamics::descriptor_type, Descriptor>::value &&
          std::is_base_of<LocalDynamicsModel<Descriptor>, Dynamics>::value>
{};

} // namespace lbm

#endif
