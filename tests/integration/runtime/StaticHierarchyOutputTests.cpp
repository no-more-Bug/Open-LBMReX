#include "integration/runtime/StaticHierarchyOutputTests.H"
#include "integration/runtime/StaticHierarchyOutputSupport.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "runtime/LbmAmrPreparation.H"

#include <limits>

namespace lbm::tests::static_hierarchy {
namespace {

struct OutputProfile {
    bool uniform = false;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<Real, AMREX_SPACEDIM> const& position, Real) const noexcept
    {
        if (uniform) return {1.0,speed,0.0,0.0};
        constexpr Real twoPi = 6.283185307179586476925286766559;
        const Real x = twoPi*position[0], y = twoPi*position[1];
        return {1.0 + 0.001*std::sin(x)*std::cos(y),
            0.01*std::cos(x),0.005*std::sin(y),0.0};
    }
};

struct Fixture {
    std::string name;
    amr::StaticHierarchySpec spec;
    int step = 0;
    std::array<Real,2> references{2.0,0.5};
};

amr::RefinementBox rectangle(Real xlo, Real xhi, Real ylo, Real yhi)
{ return {{xlo,ylo,0.0},{xhi,yhi,1.0/16.0}}; }

std::vector<Fixture> fixtures()
{
    std::vector<Fixture> result;
    const bool mpi = amrex::ParallelDescriptor::NProcs() > 1;
    for (auto const& fixture : topology::output_fixtures(2,mpi))
        result.push_back({fixture.name,fixture.spec,0,{1.0,0.02}});
    if (mpi) return result;
    amr::StaticHierarchySpec spec;
    spec.gapFillRadiusCells = spec.coarseNestingMarginCells = 1;
    spec.regions = {{1,rectangle(0.25,0.75,0.25,0.75)}};
    result.push_back({"G2",spec});
    spec.regions.push_back({2,rectangle(0.375,0.625,0.375,0.625)});
    result.push_back({"G3-P",spec});
    spec.regions = {{1,rectangle(0,1,0.25,0.75)}};
    result.push_back({"P-through-2",spec});
    spec.regions.push_back({2,rectangle(0,1,0.375,0.625)});
    result.push_back({"P-through-3",spec});
    spec.maxGridSize = {amrex::IntVect(8,8,1)};
    spec.regions = {{1,rectangle(0.75,11.0/8.0,0.25,0.75)},
        {2,rectangle(7.0/8.0,1.25,0.375,0.625)}};
    result.push_back({"P3-P",spec});
    spec.regions = {{1,rectangle(7.0/8.0,1.25,15.0/16.0,9.0/8.0)}};
    result.push_back({"P-corner",spec});
    spec.regions.clear();
    result.push_back({"S0-step2",spec,2});
    return result;
}

std::array<Real,4> expected_values(amrex::Box const& domain, amrex::IntVect const& cell, bool uniform)
{
    if (uniform) return {1.0,speed,0.0,0.0};
    const Real pi = std::acos(Real(-1.0));
    const Real x = 2*pi*(cell[0]+0.5)/domain.length(0), y = 2*pi*(cell[1]+0.5)/domain.length(1);
    return {1.0+0.001*std::sin(x)*std::cos(y),0.01*std::cos(x),0.005*std::sin(y),0.0};
}

std::vector<Frame> capture_levels(amr::LbmAmrView<D2Q9Descriptor> const& view)
{
    std::vector<Frame> result;
    for (auto const& level : view.levels) {
        require(level.macro != nullptr, "output observation is missing macro");
        auto frame = capture(level.step,level.physicalTime,level.geometry,
            level.populations.multifab(), &level.macro->multifab());
        frame.level = static_cast<int>(result.size());
        result.push_back(std::move(frame));
    }
    return result;
}

void require_no_output(std::string const& path, runtime::AmrPlotfileResult const& result, std::string const& code)
{
    require(!result.ok() && result.error.find(code) != std::string::npos, "writer missed validation: " + code);
    if (amrex::ParallelDescriptor::IOProcessor())
        require(!std::filesystem::exists(path), "rejected writer created output");
}

void reject_invalid_requests(amr::LbmAmrSimulation<D2Q9Descriptor, OutputProfile>& simulation)
{
    const std::string root = "artifacts/output/rejected";
    simulation.observe(false, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        require_no_output(root + "-macro", runtime::write_amr_plotfile(root + "-macro",view,2.0,0.5),
            "missing_plotfile_macro");
    });
    simulation.observe(true, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        int index = 0;
        for (Real invalid : {Real(0),Real(-1),std::numeric_limits<Real>::infinity(),
            std::numeric_limits<Real>::quiet_NaN(),std::numeric_limits<Real>::denorm_min()}) {
            const std::string rhoPath = root + "-rho" + std::to_string(index);
            const std::string velocityPath = root + "-velocity" + std::to_string(index++);
            require_no_output(rhoPath,runtime::write_amr_plotfile(rhoPath,view,invalid,0.5),"invalid_plotfile_reference");
            require_no_output(velocityPath,runtime::write_amr_plotfile(velocityPath,view,2.0,invalid),"invalid_plotfile_reference");
        }
        auto broken = view;
        broken.refRatios.clear();
        require_no_output(root + "-ratios",runtime::write_amr_plotfile(root + "-ratios",broken,2.0,0.5),
            "invalid_plotfile_view");
        broken.levels.clear();
        require_no_output(root + "-empty",runtime::write_amr_plotfile(root + "-empty",broken,2.0,0.5),
            "invalid_plotfile_view");
    });
}

} // namespace

void run_static_output_checks()
{
    const std::string root = "artifacts/output";
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
    }
    amrex::ParallelDescriptor::Barrier();
    std::ofstream summary;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        summary.open(root+"/composite.csv");
        summary << std::setprecision(17) << "sample,profile,volume,mass\n";
    }
    for (auto const& fixture : fixtures()) {
        auto prepared = amr::prepare_static_hierarchy<D2Q9Descriptor>(fixture.spec,OutputProfile{fixture.step != 0});
        for (auto const& error : prepared.errors) amrex::Print() << fixture.name << ": " << error << '\n';
        require(prepared.ok(), fixture.name + ": output fixture preparation failed");
        auto simulation = prepared.take();
        if (fixture.name == "G2") reject_invalid_requests(*simulation);
        if (fixture.step != 0) {
            config::LbmConfigIntent intent;
            intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
            intent.tau = 0.65; intent.uRefLb = speed; intent.uRefPhys = 1.0;
            auto parameters = runtime::prepare_amr_evolution(*simulation,intent,BGKDynamicsSelection{});
            require(parameters.ok(),"single-level output evolution preparation");
            auto evolution = std::move(parameters).take();
            close(evolution.level(0).dtPhys,dtPhys,"single-level output timestep");
            int calls = 0;
            auto result = simulation->evolve(evolution,fixture.step,false,
                [&](amr::LbmAmrView<D2Q9Descriptor> const&) { ++calls; });
            require(result.ok() && calls == fixture.step, "single-level output fixture did not advance");
        }
        const std::string directory = root + "/" + fixture.name;
        if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::create_directories(directory);
        amrex::ParallelDescriptor::Barrier();
        std::vector<Frame> source;
        simulation->observe(true, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
            source = capture_levels(view);
            for (auto const& frame : source) check_moments(frame);
            topology::validate_coverage(fixture.spec,source,2);
            const auto result = runtime::write_amr_plotfile(directory + "/plot",view,
                fixture.references[0],fixture.references[1]);
            require(result.ok(), result.error);
            auto const after = capture_levels(view);
            for (std::size_t lev = 0; lev < source.size(); ++lev) {
                compare(source[lev].populations,after[lev].populations,"writer changed populations",true);
                compare(source[lev].macro,after[lev].macro,"writer changed macro",true);
                require(after[lev].step == source[lev].step && after[lev].physicalTime == source[lev].physicalTime,
                    "writer changed time or step");
            }
        });
        const auto expected = [&](int lev, amrex::IntVect const& cell) {
            return expected_values(output_detail::level_domain(fixture.spec,lev),cell,fixture.step != 0);
        };
        const auto totals = output_detail::readback(directory+"/plot",fixture.spec,source,fixture.references,expected);
        output_detail::write_expectations(directory,fixture.name,fixture.spec,source,
            fixture.step ? "output-uniform-v1" : "output-periodic-v1",fixture.references,expected);
        if (summary.is_open()) summary << fixture.name << ',' << (fixture.step ? "uniform" : "periodic")
            << ',' << totals[0] << ',' << totals[1] << '\n';
        amrex::Print() << "static_output_pass sample=" << fixture.name << " levels=" << source.size()
            << " step=" << fixture.step << " volume=" << totals[0] << " mass=" << totals[1]
            << " plot=" << directory << "/plot\n";
    }
    if (summary.is_open()) require(summary.good(),"D2 composite summary write");
}

} // namespace lbm::tests::static_hierarchy
