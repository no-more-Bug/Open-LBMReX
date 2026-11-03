#ifndef LBM_TESTS_AMRHIERARCHYREFERENCE_H
#define LBM_TESTS_AMRHIERARCHYREFERENCE_H

#include "common/AmrPushReference.H"

namespace lbm::tests::amr_reference {

Real equilibrium(Stencil const& stencil, int q, Real rho, std::array<Real,3> const& u);

struct ReferenceLevel {
    SparseVolumeGrid current;         // Actual valid cells only.
    std::vector<Cell> support;        // Fine valid plus four fine cells, periodically wrapped.
    Real omega, dt;
    int step = 0;
    Real time = 0;
};
struct ReferenceSample {
    int level, step, coarsestStep;
    bool second;
    Real time;
    SparseVolumeGrid populations;
};

// A short-run, test-owned volume model. Values travel by forward push; full
// subvolumes are averaged only after both fine steps. No production masks,
// pull/gather operators, buffer swaps, observer helpers or clock code are used.
class HierarchyReference {
public:
    HierarchyReference(std::vector<ReferenceLevel> levels, int dimensions);
    std::vector<ReferenceSample> advance();
    std::vector<ReferenceLevel> const& levels() const { return levels_; }
private:
    SparseVolumeGrid step(int lev, SparseVolumeGrid source, unsigned char finePhase,
                          std::vector<ReferenceSample>& samples);
    SparseVolumeGrid fine_source(int lev, SparseVolumeGrid const& collided) const;
    void return_volumes(int lev, SparseVolumeGrid const& finePushed, SparseVolumeGrid& coarsePushed) const;
    std::vector<ReferenceLevel> levels_;
    Stencil stencil_;
    Cell ratio_;
};

} // namespace lbm::tests::amr_reference
#endif
