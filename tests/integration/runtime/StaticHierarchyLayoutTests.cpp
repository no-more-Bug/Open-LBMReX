#include "integration/runtime/StaticHierarchyLayoutTests.H"
#include "integration/runtime/StaticHierarchy2DFixtures.H"
#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "integration/runtime/StaticHierarchySnapshot.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"

#include <AMReX_ParmParse.H>

#include <limits>
#include <utility>

namespace lbm::tests::static_hierarchy {
namespace {

using Spec = amr::StaticHierarchySpec;
using Rectangle = amr::RefinementBox;

using namespace d2_layout;

struct LayoutProfile {
    Real xOrigin = 0.0, yOrigin = 0.0;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<Real, AMREX_SPACEDIM> const& position, Real) const noexcept
    {
        constexpr Real twoPi = 6.283185307179586476925286766559;
        const Real x = twoPi*(position[0]-xOrigin), y = twoPi*(position[1]-yOrigin);
        return {1.0 + 0.001*std::sin(x)*std::cos(y),
            0.01*std::cos(x), 0.005*std::sin(y), 0.0};
    }
};

void check_analytic(Frame const& frame)
{
    check_moments(frame);
    const auto& f = frame.populations;
    const int n = f.domain.length(0);
    const Real pi = std::acos(Real(-1.0));
    for (std::size_t cell = 0; cell < f.owners.size(); ++cell) {
        const auto index = f.cell_at(cell);
        const int i = index[0], j = index[1];
        // Oracle cell centers use the prescribed level dimensions, not the
        // production coordinate conversion or its profile evaluation.
        const Real x = 2*pi*(i+0.5)/n, y = 2*pi*(j+0.5)/n;
        const Real rho = 1.0 + 0.001*std::sin(x)*std::cos(y);
        const Real ux = 0.01*std::cos(x), uy = 0.005*std::sin(y);
        close(frame.macro.values[cell*4], rho, "layout analytic density");
        close(frame.macro.values[cell*4+1], ux, "layout analytic ux");
        close(frame.macro.values[cell*4+2], uy, "layout analytic uy");
        for (int q = 0; q < 9; ++q) {
            const Real cu = cx[q]*ux+cy[q]*uy;
            close(f.values[cell*9+q], weights[q]*rho*(1+3*cu+4.5*cu*cu-1.5*(ux*ux+uy*uy)),
                "independent layout equilibrium");
        }
    }
}

std::vector<Frame> observe_layout(amr::LbmAmrSimulation<D2Q9Descriptor, LayoutProfile>& simulation, Sample const& fixture)
{
    std::vector<Frame> frames;
    simulation.observe(true, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        require(view.levels.size() == std::size_t(fixture.finestLevel+1) && view.step == 0 && view.physicalTime == 0.0,
            fixture.name + ": missing levels or nonzero initialization time");
        require(view.refRatios.size()+1 == view.levels.size(), "refinement ratio count mismatch");
        for (auto const& ratio : view.refRatios)
            require(ratio == amrex::IntVect(2,2,1), "actual refinement ratio is not (2,2,1)");
        for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
            auto const& level = view.levels[lev];
            const int n = 32*(1 << lev);
            auto const expectedDomain = amrex::Box(amrex::IntVect(0), amrex::IntVect(n-1,n-1,0));
            require(level.geometry.Domain() == expectedDomain && level.step == 0 &&
                level.physicalTime == 0.0 && level.macro, "invalid published level geometry/state");
            auto const& f = level.populations.multifab();
            auto const& m = level.macro->multifab();
            require(f.boxArray() == level.boxes && m.boxArray() == level.boxes &&
                f.DistributionMap() == level.mapping && m.DistributionMap() == level.mapping,
                "final BA/DM is not shared by populations and macro");
            for (int d = 0; d < 3; ++d) {
                close(level.geometry.ProbLo(d), fixture.spec.probLo[d], "level physical lower bound");
                close(level.geometry.ProbHi(d), fixture.spec.probHi[d], "level physical upper bound");
                close(level.geometry.CellSize(d), d == 2 ? 1.0/16.0 : 1.0/n, "directional level spacing");
                require(level.geometry.isPeriodic(d), "level lost periodic topology");
            }
            auto const& maxGrid = fixture.spec.maxGridSize.size() == 1
                ? fixture.spec.maxGridSize.front() : fixture.spec.maxGridSize[lev];
            for (int b = 0; b < level.boxes.size(); ++b) {
                auto const& box = level.boxes[b];
                require(box.ixType().cellCentered() && expectedDomain.contains(box) &&
                    box.length(2) == 1 && box.smallEnd(0)%2 == 0 && box.smallEnd(1)%2 == 0 &&
                    box.length(0)%2 == 0 && box.length(1)%2 == 0 &&
                    box.length(0) <= maxGrid[0] && box.length(1) <= maxGrid[1],
                    "Box violates cell centering, thickness, coarse-face alignment or max size");
            }
            auto frame = capture(0, 0.0, level.geometry, f, &m);
            frame.level = static_cast<int>(lev);
            check_analytic(frame);
            frames.push_back(std::move(frame));
            if (lev == 1 && fixture.name.rfind("G2-", 0) == 0 &&
                amrex::ParallelDescriptor::NProcs() == 2 && maxGrid[0] == 32) {
                int emptyRanks = 1;
                for (amrex::MFIter mfi(f); mfi.isValid(); ++mfi) emptyRanks = 0;
                amrex::ParallelDescriptor::ReduceIntSum(emptyRanks);
                require(emptyRanks == 1 && level.boxes.size() == 1, "G2 did not exercise an empty fine rank");
            }
        }
    });
    topology::validate_coverage(fixture.spec,frames,2);
    return frames;
}

void check_through_x_ghosts(amr::LbmAmrSimulation<D2Q9Descriptor, LayoutProfile>& simulation,
    std::vector<Frame> const& frames, std::string const& name)
{
    std::vector<int> lower(frames.size()), upper(frames.size()), crossRank(frames.size());
    simulation.observe(false, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        for (std::size_t lev = 1; lev < view.levels.size(); ++lev) {
            auto const& level = view.levels[lev];
            auto const& reference = frames[lev].populations;
            auto const& domain = level.geometry.Domain();
            auto host = host_field_snapshot(level.populations.multifab());
            for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
                auto const data = host.const_array(mfi);
                auto const& box = mfi.validbox();
                for (bool high : {false,true}) {
                    if ((high ? box.bigEnd(0) : box.smallEnd(0)) !=
                        (high ? domain.bigEnd(0) : domain.smallEnd(0))) continue;
                    const int gx = high ? domain.bigEnd(0)+1 : domain.smallEnd(0)-1;
                    const int dx = high ? domain.smallEnd(0) : domain.bigEnd(0);
                    // Only face ghosts with valid y and k=0: stripe corners
                    // outside the fine band may correctly need a coarse donor.
                    for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                        const amrex::IntVect donor(dx,j,0);
                        const auto index = reference.offset(donor);
                        require(index < reference.owners.size() && reference.owners[index] == 1,
                            name + ": full-x face lost its periodic same-level donor");
                        for (int q = 0; q < 9; ++q)
                            close(data(gx,j,0,q), reference.values[index*9+q],
                                name + ": full-x periodic population copy");
                        ++(high ? upper[lev] : lower[lev]);
                        for (int b = 0; b < level.boxes.size(); ++b)
                            if (level.boxes[b].contains(donor) &&
                                level.mapping[b] != amrex::ParallelDescriptor::MyProc())
                                ++crossRank[lev];
                    }
                }
            }
        }
    });
    for (std::size_t lev = 1; lev < frames.size(); ++lev) {
        for (int* count : {&lower[lev],&upper[lev],&crossRank[lev]})
            amrex::ParallelDescriptor::ReduceIntSum(*count);
        require(lower[lev] > 0 && upper[lev] > 0, name + ": missing a periodic x face");
        if (amrex::ParallelDescriptor::NProcs() > 1)
            require(crossRank[lev] > 0, name + ": no periodic x face crossed ranks");
        amrex::Print() << "through_x_ghost_pass sample=" << name << " level=" << lev
            << " lower=" << lower[lev] << " upper=" << upper[lev]
            << " cross_rank=" << crossRank[lev] << '\n';
    }
}

std::vector<Frame> run_sample(Sample const& fixture)
{
    auto prepared = amr::prepare_static_hierarchy<D2Q9Descriptor>(fixture.spec,
        LayoutProfile{fixture.spec.probLo[0],fixture.spec.probLo[1]});
    for (auto const& error : prepared.errors) amrex::Print() << fixture.name << ": " << error << '\n';
    require(prepared.ok(), fixture.name + ": real AmrCore preparation failed");
    auto simulation = prepared.take();
    auto frames = observe_layout(*simulation, fixture);
    if (fixture.checkThroughX) check_through_x_ghosts(*simulation,frames,fixture.name);
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
    intent.tau = 0.65; intent.uRefLb = speed; intent.uRefPhys = 1.0;
    auto parameters = runtime::prepare_amr_evolution(*simulation,intent,BGKDynamicsSelection{});
    require(parameters.ok(),"multilevel layout evolution preparation");
    auto evolution = std::move(parameters).take();
    require(evolution.numLevels() == simulation->numLevels(),"layout prepared level count");
    int observations = 0;
    auto result = simulation->evolve(evolution, -1, true,
        [&](amr::LbmAmrView<D2Q9Descriptor> const&) { ++observations; });
    require(!result.ok() && result.error.find("negative_steps") != std::string::npos &&
        observations == 0, "negative multilevel step did not reject before advancing");
    auto after = observe_layout(*simulation, fixture);
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        compare(frames[lev].populations, after[lev].populations, "rejected multilevel evolve", true);
        compare(frames[lev].macro, after[lev].macro, "rejected multilevel macro", true);
    }
    write_static_snapshot("artifacts/layout/" + fixture.name + "/box" +
        std::to_string(fixture.spec.maxGridSize.front()[0]), fixture.name, fixture.spec,
        frames, "layout-periodic-v1");
    amrex::Print() << "static_layout_pass sample=" << fixture.name << " levels=" << frames.size()
        << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
    return frames;
}

void compare_selected(std::vector<Frame> const& a, std::vector<Frame> const& b, Sample const& fixture)
{
    require(a.size() == b.size(), "layout comparison lost a level");
    for (std::size_t lev = 0; lev < a.size(); ++lev) {
        compare(a[lev], b[lev], fixture.name + " equivalent layout");
    }
}

void reject(Spec const& spec, std::string const& code, int level)
{
    auto prepared = amr::prepare_static_hierarchy<D2Q9Descriptor>(spec, LayoutProfile{});
    require(!prepared.ok() && !prepared.simulation, "invalid hierarchy published a simulation: " + code);
    bool matched = false;
    for (auto const& error : prepared.errors)
        matched = matched || (error.find(code + ":") != std::string::npos &&
            error.find("level=" + std::to_string(level)) != std::string::npos);
    if (!matched) for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(matched, "missing diagnostic code/level: " + code + " level=" + std::to_string(level));
}

void negative_checks()
{
    Spec spec;
    spec.probHi[2] = 0.0; reject(spec, "invalid_mesh", 0);
    spec = {}; spec.probLo[0] = std::numeric_limits<Real>::quiet_NaN(); reject(spec, "invalid_mesh", 0);
    spec = {}; spec.probHi[2] = std::numeric_limits<Real>::infinity(); reject(spec, "invalid_mesh", 0);
    spec = {}; spec.nCell[0] = 0; reject(spec, "invalid_mesh", 0);
    spec = {}; spec.nCell[0] = 31; reject(spec, "unequal_active_spacing", 0);
    spec = {}; spec.nCell[2] = 2; reject(spec, "invalid_cell_count", 0);
    spec = {}; spec.refRatio = amrex::IntVect(2); reject(spec, "invalid_ref_ratio", 0);
    spec = {}; spec.isPeriodic[0] = 0; reject(spec, "boundary_preparation", 0);
    spec = {}; spec.probHi[1] = 2.0; reject(spec, "unequal_active_spacing", 0);
    spec = {}; spec.probLo = {1.0e16,1.0e16,0.0}; spec.probHi = {1.0e16+2.0,1.0e16+2.0,1.0};
    reject(spec, "unrepresentable_geometry", 0);
    spec = {}; spec.probHi[2] = 1.0e-320; reject(spec, "unrepresentable_geometry", 0);
    spec = {}; spec.maxGridSize.clear(); reject(spec, "invalid_max_grid_size", 0);
    spec = {}; spec.maxGridSize = {amrex::IntVect(7,8,1)}; reject(spec, "invalid_max_grid_size", 0);
    spec = {}; spec.maxGridSize = {amrex::IntVect(8,8,2)}; reject(spec, "invalid_max_grid_size", 0);
    spec = g3(true).spec; spec.maxGridSize.resize(2, amrex::IntVect(8,8,1)); reject(spec, "invalid_max_grid_size", 0);
    spec = {}; spec.coarseNestingMarginCells = 0; reject(spec, "invalid_nesting_controls", 0);
    spec = {}; spec.gapFillRadiusCells = -1; reject(spec, "invalid_nesting_controls", 0);
    for (Real eff : {Real(0),Real(1.1),std::numeric_limits<Real>::quiet_NaN()}) {
        spec = {}; spec.gridEfficiency = eff; reject(spec,"invalid_grid_efficiency",0);
    }
    for (int level : {0,-1}) {
        spec = g2(false).spec; spec.regions.front().level = level;
        reject(spec, "invalid_level_count", 0);
    }
    spec = g2(false).spec; spec.regions.front().level = 30;
    reject(spec, "index_overflow", 26);
    for (int level : {1,2}) {
        spec = g3(true).spec; spec.regions[level-1].box = {};
        reject(spec, "invalid_region", level);
    }
    const Real nan = std::numeric_limits<Real>::quiet_NaN(), inf = std::numeric_limits<Real>::infinity();
    for (auto r : {rectangle(nan,0.75,0.25,0.75), rectangle(0.25,inf,0.25,0.75),
                   rectangle(0.25,0.25,0.25,0.75), rectangle(0.75,0.25,0.25,0.75),
                   rectangle(0.25,1.5,0.25,0.75)}) {
        spec = g2(false).spec; spec.regions.front().box = r;
        reject(spec, "invalid_region", 1);
    }
    spec = g2(false).spec;
    spec.regions.front().box.hi[2] *= 0.5;
    reject(spec, "inactive_region_extent", 1);
    spec = g2(false).spec;
    spec.regions.front().box.lo[0] += 1.0/64.0;
    reject(spec, "unaligned_region", 1);
    for (auto const& bounds : {rectangle(0.125,0.375,0.375,0.625),
                               rectangle(0.25,0.5,0.375,0.625)}) {
        auto expanded = g3(true);
        expanded.name = bounds.lo[0] < 0.25 ? "Expanded-outside" : "Expanded-margin";
        expanded.spec.regions[1].box = bounds;
        run_sample(expanded);
    }
    run_sample(sample("Expanded-periodic",true,
        {rectangle(0.0,0.5,0.25,0.75),rectangle(0.0,0.25,0.375,0.625)}));
    amrex::Print() << "static_layout_negative_checks: PASS\n";
}

} // namespace

void run_static_layout_checks()
{
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts/layout");
    amrex::ParallelDescriptor::Barrier();
    const bool mpi = amrex::ParallelDescriptor::NProcs() > 1;
    if (!mpi) run_sample(sample("P3-wide-P", true,
        {rectangle(3.0/32.0,33.0/32.0,0.25,0.75), rectangle(7.0/64.0,65.0/64.0,17.0/64.0,47.0/64.0)}));
    for (bool perLevel : {false,true}) {
        run_sample(g2(perLevel));
        run_sample(corner(perLevel));
        run_sample(p3(perLevel));
        for (int levels : {2,3}) {
            auto fixture = through_x(perLevel,levels);
            auto reference = run_sample(fixture);
            if (!mpi) {
                translate_regions(fixture.spec,2.0,-3.0);
                fixture.name += "-period-shift";
                compare_selected(reference,run_sample(fixture),fixture);
            }
        }
        if (mpi) continue;
        run_sample(g3(perLevel));
        run_sample(sample(perLevel ? "P-wide-P" : "P-wide-A", perLevel,
            {rectangle(0.5,1.25,0.25,0.75)}));
        auto throughY = sample(std::string("P-through-y-") + (perLevel ? "P" : "A"),
            perLevel, {rectangle(0.25,0.75,0,1)});
        run_sample(throughY);
        translate_regions(throughY.spec,2.0,-3.0);
        throughY.name += "-period-shift";
        run_sample(throughY);
        auto wrapped = sample(perLevel ? "P-wrap-P" : "P-wrap-A", perLevel,
            {rectangle(7.0/8.0,1.25,0.25,0.75)});
        auto reference = run_sample(wrapped);
        translate_regions(wrapped.spec, -2.0, 3.0);
        wrapped.name += "-period-shift";
        compare_selected(reference, run_sample(wrapped), wrapped);
        for (bool upper : {false,true})
            run_sample(sample(std::string("P-side-") + (upper ? "upper-" : "left-") + (perLevel ? "P" : "A"),
                perLevel, {rectangle(0,0.5,upper ? 0.5 : 0,1)}));
        auto fineBoxes = corner(perLevel);
        reference = run_sample(fineBoxes);
        fineBoxes.spec.maxGridSize = {amrex::IntVect(32,32,1)};
        compare_selected(reference, run_sample(fineBoxes), fineBoxes);
    }
    if (mpi) return;
    run_sample(sample("G3-small", false, {rectangle(0.25,17.0/64.0,0.25,17.0/64.0)},32,2));
    auto standard = g3(true);
    auto reference = run_sample(standard);
    standard.spec.maxGridSize = {amrex::IntVect(8,8,1)};
    compare_selected(reference, run_sample(standard), standard);
    standard.spec.maxGridSize = {amrex::IntVect(32,32,1),amrex::IntVect(16,16,1),amrex::IntVect(8,8,1)};
    standard.name = "G3-P-per-level-grid-sizes";
    compare_selected(reference, run_sample(standard), standard);
    auto shifted = g3(true);
    shifted.name = "G3-P-shifted-origin";
    shifted.spec.probLo = {2.0,-1.0,3.0};
    shifted.spec.probHi = {3.0,0.0,3.0+1.0/16.0};
    translate_regions(shifted.spec, 2.0, -1.0, 3.0);
    compare_selected(reference, run_sample(shifted), shifted);
    negative_checks();
}

void run_static_input_checks()
{
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts/layout");
    amrex::ParallelDescriptor::Barrier();
    amrex::ParmParse pp("static_test");
    std::string name;
    pp.get("fixture",name);
    require(name == "G3-A" || name == "G3-P","unknown direct test fixture");
    auto direct = g3(name == "G3-P");
    Sample loaded = direct;
    loaded.name += "-inputs";
    loaded.spec = topology::read_spec_inputs();
    compare_selected(run_sample(direct), run_sample(loaded), loaded);
    amrex::Print() << "static_layout_inputs_pass fixture=" << name << '\n';
}

} // namespace lbm::tests::static_hierarchy
