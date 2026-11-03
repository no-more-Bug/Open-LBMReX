#ifndef LBM_CASES_LIDDRIVENCAVITY_D2Q9GUOCAVITYPROGRAM_H
#define LBM_CASES_LIDDRIVENCAVITY_D2Q9GUOCAVITYPROGRAM_H

#include "lid_driven_cavity/composition/CavityProgram.H"
#include "lbm/descriptors/D2Q9Descriptor.H"

#include <type_traits>
#include <utility>

namespace lbm::cases::lid_driven_cavity {

/** Established direct D2Q9 Guo Program type. */
template <class Descriptor, class DynamicsSelection>
using D2Q9GuoCavityProgram =
    CavityProgram<Descriptor, DynamicsSelection>;

/**
 * Preserve the established direct D2Q9 Guo composition interface while the
 * family implementation remains uniquely owned by CavityProgram.H.
 */
template <class Descriptor, class DynamicsSelection>
D2Q9GuoCavityProgram<Descriptor, DynamicsSelection>
make_d2q9_guo_cavity_program(
    Controls controls,
    DynamicsSelection dynamics,
    RestInitialization initialization)
{
    static_assert(
        std::is_same_v<Descriptor, D2Q9Descriptor>,
        "D2Q9 Guo cavity Program requires the D2Q9 Descriptor");
    controls.wallScheme = WallScheme::GuoExtrapolation;
    return make_cavity_program<Descriptor>(
        std::move(controls),
        std::move(dynamics),
        std::move(initialization));
}

} // namespace lbm::cases::lid_driven_cavity

#endif
