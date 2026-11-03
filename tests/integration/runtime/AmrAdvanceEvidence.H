#ifndef LBM_TESTS_AMRADVANCEEVIDENCE_H
#define LBM_TESTS_AMRADVANCEEVIDENCE_H

#include "integration/runtime/AmrCompositeBalance.H"

namespace lbm::tests::amr_evolution {
// Short-run snapshots for rank/backend pairing; no plotfile or fine-step I/O.
class AdvanceEvidence {
public:
    AdvanceEvidence(std::string directory, std::string sample, int steps, Real tau, bool uniform);
    void record(EvolutionView const& view, std::vector<Frame> const& frames,
                std::vector<amr_reference::ReferenceLevel> const& reference);
private:
    void identity(EvolutionView const& view);
    std::string directory_, sample_;
    int steps_;
    Real tau_, initialMass_=0;
    bool uniform_;
    std::array<Real,3> initialMomentum_{};
};
} // namespace lbm::tests::amr_evolution
#endif
