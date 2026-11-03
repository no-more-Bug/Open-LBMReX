#ifndef LBM_TESTS_DYNAMICCONSTRAINTTESTS_HH
#define LBM_TESTS_DYNAMICCONSTRAINTTESTS_HH

#include "integration/runtime/DynamicConstraintTests.H"
#include "integration/runtime/DynamicAmrTestSupport.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::dynamic_amr::constraints {

// Enumerate the returned boxes, then compare every cell with hand-written integer
// predicates. This oracle does not call production discretization or set algebra.
template<class Expected>
void check_cells(amrex::BoxArray const& actual, amrex::IntVect shape,
                 Expected expected, std::string const& label)
{
    std::vector<std::uint8_t> cells(std::size_t(shape[0])*shape[1]*shape[2], 0);
    auto index = [shape](amrex::IntVect const& p) {
        return (std::size_t(p[2])*shape[1]+p[1])*shape[0]+p[0];
    };
    amrex::Box const domain(amrex::IntVect(0), shape-amrex::IntVect(1));
    for (int b = 0; b < actual.size(); ++b) {
        require(domain.contains(actual[b]), label+": out-of-domain cell");
        for (amrex::BoxIterator it(actual[b]); it.ok(); ++it) {
            auto& count = cells[index(it())];
            if (count != 0) amrex::Abort(label+": duplicate cell");
            count = 1;
        }
    }
    for (amrex::BoxIterator it(domain); it.ok(); ++it) {
        if (bool(cells[index(it())]) != expected(it()))
            amrex::Abort(label+": integer coverage differs at "+
                std::to_string(it()[0])+","+std::to_string(it()[1])+","+
                std::to_string(it()[2]));
    }
}

inline amr::RefinementBox x_strip(amr::DynamicHierarchySpec const& spec,
                                  Real lo, Real hi)
{
    amr::RefinementBox box{spec.probLo, spec.probHi};
    box.lo[0] = lo;
    box.hi[0] = hi;
    return box;
}

template<class Descriptor>
amr::DynamicHierarchySpec small_spec()
{
    auto spec = base_spec<Descriptor>();
    spec.gapFillRadiusCells = 1;
    spec.nCell = {16,16,Descriptor::d == 3 ? 4 : 1};
    spec.probHi = {1,1,Descriptor::d == 3 ? 0.25 : 0.125};
    spec.maxGridSize = {amrex::IntVect(8,8,Descriptor::d == 3 ? 4 : 1)};
    return spec;
}

inline amr::DynamicHierarchyLayout prepared(amr::DynamicHierarchySpec const& spec, int dimensions)
{
    auto result = amr::prepare_dynamic_layout(spec, dimensions);
    std::string errors;
    for (auto const& error : result.errors) errors += " "+error;
    require(result.ok(), "valid constraints rejected:"+errors);
    require(result.layout->constraints.has_value(), "constraint preparation lost explicit constraints");
    return std::move(*result.layout);
}

inline void rejected(amr::DynamicHierarchySpec const& spec, int dimensions,
                     std::string const& scenario)
{
    auto result = amr::prepare_dynamic_layout(spec, dimensions);
    require(!result.ok() && !result.layout, scenario+": invalid constraints published a layout");
    bool found = false;
    for (auto const& error : result.errors)
        found = found || error.find("invalid_refinement_constraints:") == 0;
    require(found, scenario+": missing constraint diagnostic");
}

template<class Descriptor>
void check_fixed_band()
{
    auto spec = small_spec<Descriptor>();
    spec.nCell = {16,128,Descriptor::d == 3 ? 2 : 1};
    spec.probHi = {0.125,1,Descriptor::d == 3 ? 0.015625 : 0.125};
    spec.maxGridSize = {amrex::IntVect(8,32,Descriptor::d == 3 ? 2 : 1)};
    amr::RefinementBox band{spec.probLo,spec.probHi};
    band.lo[1] = 0.65;
    band.hi[1] = 0.85;
    spec.constraints = {{band,1,1}};
    auto layout = prepared(spec,Descriptor::d);
    auto const& c = *layout.constraints;
    auto const ratio = spec.refRatio;
    amrex::IntVect const n(spec.nCell[0],spec.nCell[1],spec.nCell[2]);
    require(c.minimumCells.size() == 3 && c.allowedCells.size() == 3 &&
            c.requiredTags.size() == 2,"fixed-band prepared level count");
    check_cells(c.minimumCells[1],n*ratio,
        [](auto p) { return p[1] >= 166 && p[1] <= 217; },"DSL raw H1");
    check_cells(c.minimumCells[2],n*ratio*ratio,[](auto) { return false; },"DSL raw H2");
    check_cells(c.requiredTags[0],n,
        [](auto p) { return p[1] >= 83 && p[1] <= 108; },"DSL raw S0");
    check_cells(c.requiredTags[1],n*ratio,[](auto) { return false; },"DSL raw S1");
    check_cells(c.allowedCells[1],n*ratio,[](auto) { return true; },"DSL A1");
    check_cells(c.allowedCells[2],n*ratio*ratio,
        [](auto p) { return p[1] < 332 || p[1] >= 436; },"DSL A2 exact fixed-band exclusion");
}

template<class Descriptor>
void check_minimum_and_overlap()
{
    auto spec = small_spec<Descriptor>();
    spec.constraints = {{x_strip(spec,0.25,0.75),1,2},
                        {x_strip(spec,0.375,0.625),2,2},
                        {x_strip(spec,0,0.25),0,1},
                        {x_strip(spec,0,0.125),0,0}};
    auto layout = prepared(spec,Descriptor::d);
    auto const& c = *layout.constraints;
    amrex::IntVect const n(spec.nCell[0],spec.nCell[1],spec.nCell[2]);
    auto const ratio = spec.refRatio;
    check_cells(c.minimumCells[1],n*ratio,
        [](auto p) { return p[0] >= 8 && p[0] <= 23; },"overlap maximum minimum H1");
    check_cells(c.minimumCells[2],n*ratio*ratio,
        [](auto p) { return p[0] >= 24 && p[0] <= 39; },"overlap maximum minimum H2");
    check_cells(c.allowedCells[1],n*ratio,
        [](auto p) { return p[0] >= 4; },"overlap minimum maximum A1");
    check_cells(c.allowedCells[2],n*ratio*ratio,
        [](auto p) { return p[0] >= 16; },"overlap minimum maximum A2");
    check_cells(c.requiredTags[0],n,
        [](auto p) { return p[0] >= 4 && p[0] <= 11; },"raw S0 excludes nesting additions");
    check_cells(c.requiredTags[1],n*ratio,
        [](auto p) { return p[0] >= 12 && p[0] <= 19; },"raw S1");

    // H1 initially has only four x cells; L2 requires a wider L1 support.
    // The preparation may validate that closure, but S0 remains the raw request.
    spec.constraints = {{x_strip(spec,0.5,0.625),2,2}};
    layout = prepared(spec,Descriptor::d);
    check_cells(layout.constraints->minimumCells[1],n*ratio,
        [](auto p) { return p[0] >= 16 && p[0] <= 19; },"raw H1 before support closure");
    check_cells(layout.constraints->requiredTags[0],n,
        [](auto p) { return p[0] >= 8 && p[0] <= 9; },"S0 is not closed H1");

    spec.constraints.push_back({x_strip(spec,0.375,0.4375),0,0});
    rejected(spec,Descriptor::d,"hard nesting reaches forbidden region");

    spec.constraints = {{x_strip(spec,0.25,0.5),2,2},
                        {x_strip(spec,0.375,0.625),0,1}};
    rejected(spec,Descriptor::d,"physical minimum/maximum overlap");

    // Physical intervals are disjoint; both hit L0 parent 4 after outward rounding.
    spec.constraints = {{x_strip(spec,0.25,0.30),1,2},
                        {x_strip(spec,0.31,0.35),0,0}};
    rejected(spec,Descriptor::d,"discrete minimum/maximum overlap");
}

template<class Descriptor>
void check_allowed_and_domain()
{
    auto spec = small_spec<Descriptor>();
    spec.isPeriodic[0] = 0;
    spec.constraints = {{x_strip(spec,0,0.13),0,0},
                        {x_strip(spec,0.87,1),0,0}};
    auto layout = prepared(spec,Descriptor::d);
    amrex::IntVect const n(spec.nCell[0],spec.nCell[1],spec.nCell[2]);
    auto const ratio = spec.refRatio;
    check_cells(layout.constraints->allowedCells[1],n*ratio,
        [](auto p) { return p[0] >= 6 && p[0] <= 25; },"inward A1 from outward domain guards");
    check_cells(layout.constraints->allowedCells[2],n*ratio*ratio,
        [](auto p) { return p[0] >= 12 && p[0] <= 51; },"inward A2 inherits L0 guards");
    check_cells(layout.constraints->minimumCells[1],n*ratio,
        [](auto) { return false; },"domain guard creates no minimum");

    spec.constraints.pop_back();
    rejected(spec,Descriptor::d,"unprotected high nonperiodic face");
    spec.constraints.clear();
    rejected(spec,Descriptor::d,"unconstrained nonperiodic fine envelope");

    spec.constraints = {{{spec.probLo,spec.probHi},0,0}};
    layout = prepared(spec,Descriptor::d);
    require(layout.constraints->allowedCells[1].empty() &&
            layout.constraints->allowedCells[2].empty(),"explicit empty allowance must stay empty");

    spec.constraints.clear();
    spec.maxLevel = 0;
    auto single = amr::prepare_dynamic_layout(spec,Descriptor::d);
    require(single.ok() && !single.layout->constraints,
            "nonperiodic L0-only layout needs no fine envelope");

    // A cap-only L1 region is rounded on L1 parents, not L0 parents.
    spec = small_spec<Descriptor>();
    spec.constraints = {{x_strip(spec,0.27,0.36),0,1}};
    layout = prepared(spec,Descriptor::d);
    check_cells(layout.constraints->allowedCells[2],n*ratio*ratio,
        [](auto p) { return p[0] < 16 || p[0] >= 24; },"cap-only L1 parent rounding");
}

template<class Descriptor>
void check_periodic_and_invalid()
{
    auto spec = small_spec<Descriptor>();
    spec.constraints = {{x_strip(spec,0.9,1.1),1,1}};
    auto layout = prepared(spec,Descriptor::d);
    amrex::IntVect const n(spec.nCell[0],spec.nCell[1],spec.nCell[2]);
    auto const ratio = spec.refRatio;
    check_cells(layout.constraints->minimumCells[1],n*ratio,
        [](auto p) { return p[0] <= 3 || p[0] >= 28; },"periodic wrapped minimum");
    check_cells(layout.constraints->allowedCells[2],n*ratio*ratio,
        [](auto p) { return p[0] >= 8 && p[0] < 56; },"periodic wrapped cap");
    check_cells(layout.constraints->requiredTags[0],n,
        [](auto p) { return p[0] <= 1 || p[0] >= 14; },"periodic wrapped source tags");

    spec.constraints[0].box.lo[0] += 2;
    spec.constraints[0].box.hi[0] += 2;
    layout = prepared(spec,Descriptor::d);
    check_cells(layout.constraints->minimumCells[1],n*ratio,
        [](auto p) { return p[0] <= 3 || p[0] >= 28; },"period-shift invariant");

    spec.constraints = {{x_strip(spec,0.25,0.5),2,1}};
    rejected(spec,Descriptor::d,"inverted level interval");
    spec.constraints[0].minLevel = -1;
    rejected(spec,Descriptor::d,"negative minimum");
    spec.constraints[0].minLevel = 0;
    spec.constraints[0].maxLevel = 3;
    rejected(spec,Descriptor::d,"maximum beyond allowed hierarchy");

    spec.constraints = {{x_strip(spec,0.25,0.25),1,1}};
    rejected(spec,Descriptor::d,"empty physical interval");
    if constexpr (Descriptor::d == 2) {
        spec.constraints = {{x_strip(spec,0.25,0.5),1,1}};
        spec.constraints[0].box.hi[2] *= 0.5;
        rejected(spec,Descriptor::d,"partial inactive thickness");
        spec.constraints[0].box.hi[2] = spec.probHi[2];
        spec.isPeriodic[2] = 0;
        layout = prepared(spec,Descriptor::d);
        require(layout.constraints->minimumCells[1].minimalBox().length(2) == 1,
                "inactive nonperiodic direction is not a domain-face restriction");
    }
}

template<class Descriptor>
void run_preparation_checks()
{
    auto spec = small_spec<Descriptor>();
    auto unconstrained = amr::prepare_dynamic_layout(spec,Descriptor::d);
    require(unconstrained.ok() && !unconstrained.layout->constraints,
            "empty periodic constraints changed the legacy layout path");
    check_fixed_band<Descriptor>();
    check_minimum_and_overlap<Descriptor>();
    check_allowed_and_domain<Descriptor>();
    check_periodic_and_invalid<Descriptor>();
    amrex::Print() << "dynamic_constraints_prepare_pass d=" << Descriptor::d
                   << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
}

} // namespace lbm::tests::dynamic_amr::constraints

#endif
