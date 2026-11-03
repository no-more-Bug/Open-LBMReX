#ifndef LBM_AMR_STATICREFINEMENT_H
#define LBM_AMR_STATICREFINEMENT_H

#include "lbm/amr/RefinementConstraints.H"

#include <optional>
#include <string>
#include <vector>

namespace lbm::amr {

struct RefinementRegion {
    int level = 1;
    RefinementBox box;
};

/** Physical minimum-coverage requests; omitted intermediate levels are generated. */
struct StaticHierarchySpec : HierarchyGeometrySpec {
    std::vector<RefinementRegion> regions;
};

/** Initialization-only layout facts. AmrCore owns the generated mesh. */
struct StaticHierarchyLayout : HierarchyGeometry {
    std::vector<amrex::BoxArray> requestedCells;
    std::vector<amrex::BoxArray> targetCells;
    std::vector<amrex::BoxArray> tagCells;
    std::vector<unsigned> repairReasons;
};

struct StaticLayoutResult {
    std::optional<StaticHierarchyLayout> layout;
    std::vector<std::string> errors;
    bool ok() const noexcept { return layout.has_value() && errors.empty(); }
};

StaticLayoutResult prepare_static_layout(StaticHierarchySpec const& spec, int activeDimensions);

struct StaticLayoutUpdate {
    bool rebuild = false;
    std::vector<std::string> errors;
};

// Reconcile actual coverage with accumulated requirements before publishing fields.
StaticLayoutUpdate update_static_layout(
    StaticHierarchyLayout& layout, amrex::AmrCore const& actual);

void report_static_layout(StaticHierarchyLayout const& layout,
                         amrex::AmrCore const& actual, int initializationPasses);

} // namespace lbm::amr

#endif
