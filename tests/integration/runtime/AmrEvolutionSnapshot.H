#ifndef LBM_TESTS_AMREVOLUTIONSNAPSHOT_H
#define LBM_TESTS_AMREVOLUTIONSNAPSHOT_H

#include "integration/runtime/AmrEvolutionTestCase.H"
#include <fstream>

namespace lbm::tests::amr_evolution {

// Test-owned consumer of borrowed views. Fine rows never form a composite frame.
class EvolutionSnapshot {
public:
    EvolutionSnapshot(std::string directory, Fixture const& fixture, int coarseSteps);
    void synchronized(EvolutionView const& view, std::vector<amr_reference::ReferenceLevel> const& reference);
    void fine(amr::FineSubstepView<Descriptor> const& view, amr_reference::ReferenceSample const& reference);
private:
    void identity(EvolutionView const& view);
    void rows(std::string_view phase, int coarsestStep, Frame const& actual,
              amr_reference::SparseVolumeGrid const& reference, Snapshot const* native);
    std::string directory_;
    Fixture const& fixture_;
    int coarseSteps_, event_ = 0;
    std::ofstream states_;
    Real initialMass_ = 0;
    std::array<Real,3> initialMomentum_{};
};

void output_checks();

} // namespace lbm::tests::amr_evolution
#endif
