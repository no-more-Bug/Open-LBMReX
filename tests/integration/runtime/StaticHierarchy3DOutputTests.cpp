#include "integration/runtime/StaticHierarchy3DTests.H"
#include "integration/runtime/StaticHierarchy3DFixtures.H"
#include "integration/runtime/StaticHierarchyOutputSupport.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"

#include <AMReX_ParmParse.H>
#include <set>

namespace lbm::tests::static_hierarchy::d3 {
namespace {

constexpr std::array<Real,2> references{1.0,0.02};

std::vector<topology::Fixture> output_fixtures()
{
    bool supplementOnly = false;
    amrex::ParmParse("static_test").query("supplement_only",supplementOnly);
    if (supplementOnly)
        return topology::supplement_output_fixtures(amrex::ParallelDescriptor::NProcs() > 1);
    const std::set<std::string> names{"D3-G2-P","D3-G3-P","D3-Vertex-P","D3-T2-z-P",
        "D3-T3-z-P","D3-Slab-z-P","D3-Full-P"};
    std::vector<topology::Fixture> result;
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        for (auto const& fixture : topology::output_fixtures(3,true))
            result.push_back({fixture.name,fixture.spec});
        return result;
    }
    for (auto& fixture : layout_fixtures(false))
        if (names.count(fixture.name) && fixture.spec.maxGridSize.front()[0] == 16)
            result.push_back({fixture.name,std::move(fixture.spec)});
    result.push_back({"D3-S0-step2",base_spec()});
    require(result.size() == 8,"D3 legacy output sample set");
    for (auto const& fixture : topology::output_fixtures(3))
        result.push_back({fixture.name,fixture.spec});
    // Preserve the original N0=40 geometry that exposed composite summation drift.
    auto nonbinary = base_spec(64);
    nonbinary.nCell = {40,40,40};
    nonbinary.gridEfficiency = 0.7;
    constexpr Real h = 1.0/80.0; // L2 requests on adjacent L1 cell faces.
    for (auto const& b : std::vector<std::array<int,4>>{{32,38,42,58},{20,30,30,44},
            {40,54,26,42},{20,36,42,62}})
        nonbinary.regions.push_back({2,{{b[0]*h,b[2]*h,28*h},{b[1]*h,b[3]*h,44*h}}});
    result.push_back({"D3-Topo-NonbinaryVolume",std::move(nonbinary)});
    return result;
}

} // namespace

void output_checks()
{
    std::ofstream summary;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::create_directories("artifacts/output");
        summary.open("artifacts/output/composite.csv");
        summary << std::setprecision(17) << "sample,profile,volume,mass\n";
    }
    for (auto const& fixture : output_fixtures()) {
        const int step = fixture.name == "D3-S0-step2" ? 2 : 0;
        auto prepared = amr::prepare_static_hierarchy<Descriptor>(fixture.spec,make_profile(fixture.spec,step != 0));
        require(prepared.ok(),fixture.name+": D3 output preparation");
        auto simulation = prepared.take();
        if (step) {
            auto parameters = runtime::prepare_amr_evolution(*simulation,intent(),BGKDynamicsSelection{});
            require(parameters.ok(),"D3 output evolution preparation");
            auto evolution = std::move(parameters).take();
            require(simulation->evolve(evolution,step,false,[](View const&) {}).ok(),
                "D3 real output advancement");
        }
        const auto path = "artifacts/output/"+fixture.name;
        if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::create_directories(path);
        amrex::ParallelDescriptor::Barrier();
        auto frames = observe(*simulation,step != 0);
        topology::validate_coverage(fixture,frames,3);
        simulation->observe(true,[&](View const& view) {
            require(runtime::write_amr_plotfile(path+"/plot",view,references[0],references[1]).ok(),"D3 writer");
        });
        const auto after = observe(*simulation,step != 0);
        for (std::size_t lev = 0; lev < frames.size(); ++lev) {
            compare(frames[lev].populations,after[lev].populations,"D3 writer changed populations",true);
            compare(frames[lev].macro,after[lev].macro,"D3 writer changed macro",true);
            require(frames[lev].step == after[lev].step && frames[lev].physicalTime == after[lev].physicalTime,
                "D3 writer changed time/step");
        }
        const auto expected = [&](int lev, amrex::IntVect const& cell) {
            return expected_macros(output_detail::level_domain(fixture.spec,lev),cell,step != 0);
        };
        output_detail::write_expectations(path,fixture.name,fixture.spec,frames,
            step ? "d3-output-uniform-v1" : "d3-periodic-xyz-v1",references,expected);
        const auto totals = output_detail::readback(path+"/plot",fixture.spec,frames,references,expected);
        if (summary.is_open()) summary << fixture.name << ',' << (step ? "uniform" : "periodic") << ','
            << totals[0] << ',' << totals[1] << '\n';
        if (fixture.name == "D3-G2-P" || fixture.name == "D3-G3-P") {
            auto uniform = amr::prepare_static_hierarchy<Descriptor>(fixture.spec,make_profile(fixture.spec,true));
            require(uniform.ok(),"D3 uniform composite preparation");
            std::vector<ValidSnapshot> macros;
            for (auto& frame : observe(*uniform.simulation,true)) macros.push_back(std::move(frame.macro));
            const auto uniformExpected = [&](int lev, amrex::IntVect const& cell) {
                return expected_macros(output_detail::level_domain(fixture.spec,lev),cell,true);
            };
            const auto uniformTotals = output_detail::composite(fixture.spec,macros,1.0,uniformExpected);
            if (summary.is_open()) summary << fixture.name << ",uniform," << uniformTotals[0] << ',' << uniformTotals[1] << '\n';
        }
        amrex::Print() << "d3_output_pass sample=" << fixture.name << " volume=" << totals[0] << " mass=" << totals[1] << '\n';
    }
    if (summary.is_open()) require(summary.good(),"D3 composite summary write");
}

} // namespace lbm::tests::static_hierarchy::d3
