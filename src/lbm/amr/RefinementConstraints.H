#ifndef LBM_AMR_REFINEMENTCONSTRAINTS_H
#define LBM_AMR_REFINEMENTCONSTRAINTS_H

#include "lbm/amr/HierarchyGeometry.H"

namespace lbm::amr {

struct RefinementBox {
    amrex::Array<amrex::Real, 3> lo{};
    amrex::Array<amrex::Real, 3> hi{};
};

struct RefinementConstraint {
    RefinementBox box;
    int minLevel = 0;
    int maxLevel = 0;
};

struct RefinementConstraints {
    std::vector<amrex::BoxArray> minimumCells;
    std::vector<amrex::BoxArray> allowedCells;
    // Source-level tags from raw minimumCells, before gap/nesting closure.
    std::vector<amrex::BoxArray> requiredTags;
};

struct RefinementConstraintsResult {
    std::optional<RefinementConstraints> constraints;
    std::vector<std::string> errors;
    bool ok() const noexcept { return constraints.has_value() && errors.empty(); }
};

RefinementConstraintsResult prepare_refinement_constraints(
    std::vector<RefinementConstraint> const& requests, HierarchyGeometry const& layout);

} // namespace lbm::amr
#endif
