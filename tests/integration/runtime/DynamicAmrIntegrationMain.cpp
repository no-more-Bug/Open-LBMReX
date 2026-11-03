#include "integration/runtime/DynamicRegridTests.HH"
#include "integration/runtime/DynamicGridClusteringTests.HH"
#include "integration/runtime/DynamicConstraintTests.HH"
#include "integration/runtime/DynamicConstraintGridTests.HH"
#include "integration/runtime/DynamicConstraintLifecycleTests.HH"
#include "integration/runtime/DynamicBoundaryTests.HH"
#include "integration/runtime/PopulationRemapTests.HH"
#include <AMReX_ParmParse.H>
#include "lbm/amr/DynamicRefinement.H"
#include "lbm/amr/StaticRefinement.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <limits>
#include <string>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = lbm::D3Q27Descriptor;
#else
using Descriptor = lbm::D2Q9Descriptor;
#endif

void require(bool condition, std::string const& message)
{
    if (!condition) amrex::Abort(message);
}

lbm::amr::DynamicHierarchySpec base_spec()
{
    lbm::amr::DynamicHierarchySpec spec;
    spec.nCell = {16, 16, Descriptor::d == 3 ? 16 : 1};
    spec.probLo = {2, -1, 3};
    spec.probHi = {3, 0, Descriptor::d == 3 ? 4 : 3.125};
    spec.refRatio = amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(8, 8, Descriptor::d == 3 ? 8 : 1)};
    return spec;
}

void reject(lbm::amr::DynamicHierarchySpec const& spec, std::string const& code)
{
    auto result = lbm::amr::prepare_dynamic_layout(spec, Descriptor::d);
    require(!result.ok() && !result.layout, "invalid dynamic spec published a layout");
    bool found = false;
    for (auto const& error : result.errors) found = found || error.find(code + ":") == 0;
    require(found, "missing dynamic diagnostic: " + code);
}

void check_preparation()
{
    using namespace lbm::amr;
    for (int maxLevel : {0, 1, 2}) {
        auto spec = base_spec();
        spec.maxLevel = maxLevel;
        auto result = prepare_dynamic_layout(spec, Descriptor::d);
        require(result.ok(), "valid dynamic geometry rejected");
        auto const& layout = *result.layout;
        require(layout.mesh.max_level == maxLevel && layout.regridInterval == 8,
                "allowed level or default interval changed");
        require(layout.activeDimensions == Descriptor::d && layout.mesh.n_proper == 4 &&
                layout.gapFillRadiusCells == 4, "geometry dimensions or nesting defaults changed");
        require(layout.geometry.Domain().ixType().cellCentered(), "geometry must be cell centered");
        for (int d = 0; d < 3; ++d) {
            require(layout.geometry.Domain().length(d) == spec.nCell[d] &&
                    layout.geometry.ProbLo(d) == spec.probLo[d] &&
                    layout.geometry.ProbHi(d) == spec.probHi[d] &&
                    layout.geometry.isPeriodic(d), "physical domain changed");
            require(layout.geometry.CellSize(d) == (d < Descriptor::d ? 0.0625 : 0.125),
                    "incorrect physical spacing");
        }
        for (int lev = 0; lev <= maxLevel; ++lev) {
            require(layout.mesh.ref_ratio[lev] == spec.refRatio &&
                    layout.mesh.max_grid_size[lev] == spec.maxGridSize[0] &&
                    layout.mesh.n_error_buf[lev] == amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 0),
                    "default refinement metadata differs");
        }
        StaticHierarchySpec fixed;
        static_cast<HierarchyGeometrySpec&>(fixed) = spec;
        if (maxLevel > 0) fixed.regions.push_back({maxLevel, {spec.probLo, spec.probHi}});
        auto staticResult = prepare_static_layout(fixed, Descriptor::d);
        require(staticResult.ok() && staticResult.layout->mesh.max_level == maxLevel,
                "shared geometry broke static preparation");
        auto const& staticLayout = *staticResult.layout;
        for (int lev = 0; lev <= maxLevel; ++lev) {
            require(staticLayout.mesh.n_error_buf[lev] == amrex::IntVect(0),
                    "dynamic tag buffers leaked into static path");
        }
        amrex::Long expected = 1;
        for (int d = 0; d < Descriptor::d; ++d) expected *= 16 * (1 << maxLevel);
        require(staticLayout.targetCells[maxLevel].numPts() == expected,
                "static full-domain request lost exact coverage");
    }

    auto spec = base_spec();
    spec.maxLevel = 2;
    spec.regridInterval = 0;
    spec.coarseNestingMarginCells = 2;
    spec.tagBufferCells = {amrex::IntVect(0), amrex::IntVect(1, 1, Descriptor::d == 3 ? 1 : 0),
                          amrex::IntVect(3, 3, Descriptor::d == 3 ? 3 : 0)};
    spec.maxGridSize = {amrex::IntVect(8, 8, Descriptor::d == 3 ? 8 : 1),
                       amrex::IntVect(16, 16, Descriptor::d == 3 ? 16 : 1),
                       amrex::IntVect(32, 32, Descriptor::d == 3 ? 32 : 1)};
    auto overrides = prepare_dynamic_layout(spec, Descriptor::d);
    require(overrides.ok() && overrides.layout->regridInterval == 0 &&
            overrides.layout->mesh.max_level == 2 && overrides.layout->mesh.n_proper == 2,
            "N=0 must preserve the allowed initial hierarchy");
    for (int lev = 0; lev <= 2; ++lev)
        require(overrides.layout->mesh.n_error_buf[lev] == spec.tagBufferCells[lev] &&
                overrides.layout->mesh.max_grid_size[lev] == spec.maxGridSize[lev],
                "per-level override lost");
    spec.tagBufferCells = {amrex::IntVect(0)};
    auto broadcast = prepare_dynamic_layout(spec, Descriptor::d);
    require(broadcast.ok() && broadcast.layout->mesh.n_error_buf[2] == amrex::IntVect(0),
            "single buffer must broadcast");

    spec = base_spec(); spec.maxLevel = -1; reject(spec, "invalid_level_count");
    spec = base_spec(); spec.maxLevel = std::numeric_limits<int>::max(); reject(spec, "index_overflow");
    spec = base_spec(); spec.nCell[0] = 0; reject(spec, "invalid_mesh");
    spec = base_spec(); spec.probHi[0] += 1; reject(spec, "unequal_active_spacing");
    spec = base_spec(); spec.regridInterval = -1; reject(spec, "invalid_regrid_interval");
    spec = base_spec(); spec.isPeriodic[0] = 0; reject(spec, "invalid_refinement_constraints");
    spec = base_spec(); spec.coarseNestingMarginCells = 1; reject(spec, "invalid_nesting_controls");
    spec = base_spec(); spec.maxLevel = 2;
    spec.tagBufferCells = {amrex::IntVect(0), amrex::IntVect(0)}; reject(spec, "invalid_tag_buffer");
    spec = base_spec(); spec.tagBufferCells = {amrex::IntVect(-1)}; reject(spec, "invalid_tag_buffer");
    spec = base_spec(); spec.tagBufferCells = {amrex::IntVect(std::numeric_limits<int>::max())};
    reject(spec, "invalid_tag_buffer");
    if constexpr (Descriptor::d == 2) {
        spec = base_spec(); spec.tagBufferCells = {amrex::IntVect(2)}; reject(spec, "invalid_tag_buffer");
        spec = base_spec(); spec.nCell[2] = 2; reject(spec, "invalid_cell_count");
    } else {
        spec = base_spec(); spec.refRatio[2] = 1; reject(spec, "invalid_ref_ratio");
    }
    amrex::Print() << "dynamic_geometry_pass d=" << Descriptor::d
                   << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    std::string mode = "prepare";
    amrex::ParmParse("test").query("mode",mode);
    if (mode == "prepare") check_preparation();
    else if (mode == "boundary_force") lbm::tests::dynamic_amr::dynamic_boundary::run_force_checks<Descriptor>();
    else if (mode == "boundary_relayout") lbm::tests::dynamic_amr::dynamic_boundary::run_relayout_checks<Descriptor>();
    else if (mode == "boundary_domain") lbm::tests::dynamic_amr::dynamic_boundary::run_domain_checks<Descriptor>();
    else if (mode == "constrained_tagging") lbm::tests::dynamic_amr::constraint_lifecycle::run_tagging_checks<Descriptor>();
    else if (mode == "constrained_initialize") lbm::tests::dynamic_amr::constraint_lifecycle::run_initialization_checks<Descriptor>();
    else if (mode == "constrained_regrid") lbm::tests::dynamic_amr::constraint_lifecycle::run_regrid_checks<Descriptor>();
    else if (mode == "constraints_finalize") lbm::tests::dynamic_amr::constraint_grids::run_checks<Descriptor>();
    else if (mode == "constraints_prepare") lbm::tests::dynamic_amr::constraints::run_preparation_checks<Descriptor>();
    else if (mode == "tagging") lbm::tests::dynamic_amr::run_tagging_checks<Descriptor>();
    else if (mode == "initialize") lbm::tests::dynamic_amr::run_initialization_checks<Descriptor>();
    else if (mode == "velocity") lbm::tests::dynamic_amr::run_velocity_interface_checks<Descriptor>();
    else if (mode == "remap") lbm::tests::dynamic_amr::remap::run_checks<Descriptor>();
    else if (mode == "regrid") lbm::tests::dynamic_amr::run_regrid_checks<Descriptor>();
    else if (mode == "clustering") lbm::tests::dynamic_amr::clustering::run_checks<Descriptor>();
    else if (mode == "relayout") lbm::tests::dynamic_amr::run_relayout_checks<Descriptor>();
    else amrex::Abort("unknown dynamic test mode");
    amrex::Finalize();
}
