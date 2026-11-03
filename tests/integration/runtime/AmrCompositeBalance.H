#ifndef LBM_TESTS_AMRCOMPOSITEBALANCE_H
#define LBM_TESTS_AMRCOMPOSITEBALANCE_H

#include "integration/runtime/AmrEvolutionTestCase.H"

namespace lbm::tests::amr_evolution {
struct CompositeTotals {
    long double volume=0, mass=0, referenceMass=0;
    std::array<long double,3> momentum{}, referenceMomentum{};
    std::size_t cells=0;
};

// Integrate already-global snapshots, excluding covered coarse cells.
CompositeTotals composite(EvolutionView const& view, std::vector<Frame> const& frames,
                          std::vector<amr_reference::ReferenceLevel> const& reference);
void compare_totals(CompositeTotals const& actual, Real referenceMass,
                    std::array<Real,3> const& referenceMomentum, Real initialMass,
                    std::string_view subject);
} // namespace lbm::tests::amr_evolution
#endif
