#include "integration/runtime/StaticHierarchy3DTests.H"
#include "integration/runtime/StaticHierarchy3DFixtures.H"
#include "integration/runtime/StaticHierarchySnapshot.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"

#include <AMReX_ParmParse.H>
#include <limits>

namespace lbm::tests::static_hierarchy::d3 {
namespace {

void check_geometry(View const& view, Fixture const& fixture, std::vector<Frame> const& frames)
{
    require(view.levels.size() == std::size_t(fixture.finestLevel+1),"D3 level count");
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        auto const& level = view.levels[lev];
        const int scale = 1 << lev;
        for (int d = 0; d < 3; ++d) {
            require(level.geometry.Domain().length(d) == fixture.spec.nCell[d]*scale &&
                level.geometry.Domain().smallEnd(d) == 0 && level.geometry.isPeriodic(d),
                "D3 directional domain/periodicity");
            close(level.geometry.ProbLo(d),fixture.spec.probLo[d],"D3 origin");
            close(level.geometry.ProbHi(d),fixture.spec.probHi[d],"D3 upper bound");
            close(level.geometry.CellSize(d),
                (fixture.spec.probHi[d]-fixture.spec.probLo[d])/(fixture.spec.nCell[d]*scale),"D3 spacing");
        }
        if (lev) require(view.refRatios[lev-1] == amrex::IntVect(2),"D3 refinement ratio");
        require(level.boxes.isDisjoint() && level.boxes.ixType().cellCentered(),"D3 valid ownership");
        for (int b = 0; b < level.boxes.size(); ++b) {
            auto const& box = level.boxes[b];
            require(level.geometry.Domain().contains(box) &&
                box.length().allLE(fixture.spec.maxGridSize.front()),"D3 Box bounds");
            if (lev) require(amrex::refine(amrex::coarsen(box,2),2) == box,"D3 coarsenable Box");
        }
    }
    topology::validate_coverage(fixture.spec,frames,3);
}

void check_ghosts(View const& view, Fixture const& fixture, std::vector<Frame> const& frames)
{
    int sameLevel = 0, coarseFine = 0, crossRank = 0, crossZ = 0, crossDiagonal = 0;
    std::array<int,3> codim{};
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        auto const& level = view.levels[lev];
        auto const& f = frames[lev].populations;
        require(level.populations.nGrowVect() == amrex::IntVect(lev ? 2 : 1),
                "D3 level population ghost width");
        auto host = host_field_snapshot(level.populations.multifab());
        for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const data = host.const_array(mfi);
            for (amrex::BoxIterator it(amrex::grow(mfi.validbox(),host.nGrowVect())); it.ok(); ++it) {
                auto const cell = it();
                if (mfi.validbox().contains(cell)) continue;
                auto donor = cell;
                int outsideAxes = 0;
                for (int d = 0; d < 3; ++d) {
                    donor[d] = (cell[d]+f.domain.length(d))%f.domain.length(d);
                    outsideAxes += cell[d] < mfi.validbox().smallEnd(d) || cell[d] > mfi.validbox().bigEnd(d);
                }
                const auto index = f.offset(donor);
                if (index == f.owners.size()) {
                    require(lev > 0,"D3 L0 ghost without valid donor");
                    auto const& coarse = frames[lev-1].populations;
                    require(coarse.offset(donor/2) < coarse.owners.size(),"D3 coarse/fine ghost source");
                    ++coarseFine;
                    continue; // no coarse/fine population value contract before Stage 6.03
                }
                ++sameLevel; ++codim[outsideAxes-1];
                auto const donors = level.boxes.intersections(amrex::Box(donor,donor));
                const bool remote = level.mapping[donors.front().first] != amrex::ParallelDescriptor::MyProc();
                if (lev && remote) {
                    ++crossRank;
                    if (cell[2] != donor[2] || cell[2] < mfi.validbox().smallEnd(2) ||
                        cell[2] > mfi.validbox().bigEnd(2)) ++crossZ;
                }
                for (int q = 0; q < 27; ++q) {
                    require(data(cell,q) == f.values[index*27+q],"D3 exact same-level/periodic ghost copy");
                    const auto& c = directions[q];
                    const int norm = c[0]*c[0]+c[1]*c[1]+c[2]*c[2];
                    // Diagonal populations can enter through a face ghost.
                    // Count only components whose pull receiver is actually valid.
                    if (lev && remote && norm >= 2 &&
                        mfi.validbox().contains(cell+amrex::IntVect(c[0],c[1],c[2]))) ++crossDiagonal;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntSum(sameLevel);
    amrex::ParallelDescriptor::ReduceIntSum(coarseFine);
    amrex::ParallelDescriptor::ReduceIntSum(crossRank);
    amrex::ParallelDescriptor::ReduceIntSum(crossZ);
    amrex::ParallelDescriptor::ReduceIntSum(crossDiagonal);
    amrex::ParallelDescriptor::ReduceIntSum(codim.data(),3);
    require(sameLevel > 0 && codim[0] && codim[1] && codim[2],"D3 face/edge/vertex ghost coverage");
    if (amrex::ParallelDescriptor::NProcs() > 1 && fixture.requireCrossRank)
        require(crossRank > 0 && crossZ > 0 && crossDiagonal > 0,"D3 actual cross-rank z/diagonal donors");
    if (amrex::ParallelDescriptor::NProcs() > 1 && fixture.requireEmptyFineRank) {
        const auto& ranks = view.levels[1].mapping.ProcessorMap();
        int empty = std::find(ranks.begin(),ranks.end(),amrex::ParallelDescriptor::MyProc()) == ranks.end();
        amrex::ParallelDescriptor::ReduceIntSum(empty);
        require(empty > 0,"D3 actual empty fine rank");
    }
    amrex::Print() << "d3_donors sample=" << fixture.name << " same=" << sameLevel
        << " coarse_fine=" << coarseFine << " remote=" << crossRank
        << " remote_z=" << crossZ << " remote_diagonal_populations=" << crossDiagonal << '\n';
}

std::vector<Frame> run(Fixture const& fixture)
{
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(fixture.spec,make_profile(fixture.spec));
    for (auto const& error : prepared.errors) amrex::Print() << fixture.name << ": " << error << '\n';
    require(prepared.ok(),"D3 layout preparation: "+fixture.name);
    auto simulation = prepared.take();
    auto frames = observe(*simulation);
    simulation->observe(false,[&](View const& view) {
        require(view.step == 0 && view.physicalTime == 0,"D3 multilevel initialization time");
        check_geometry(view,fixture,frames);
        check_ghosts(view,fixture,frames);
    });
    auto evolutionIntent = intent();
    evolutionIntent.lengthRefLb.reset(); evolutionIntent.lengthRefPhys.reset();
    auto parameters = runtime::prepare_amr_evolution(*simulation,evolutionIntent,BGKDynamicsSelection{});
    require(parameters.ok(),"D3 multilevel evolution preparation");
    auto evolution = std::move(parameters).take();
    int calls = 0;
    auto callback = [&](View const&) { ++calls; };
    require(simulation->evolve(evolution,0,false,callback).ok() && calls == 0,"D3 multilevel zero no-op");
    auto rejected = simulation->evolve(evolution,-1,false,callback);
    require(rejected.error.find("negative_steps") != std::string::npos && calls == 0,
        "D3 multilevel negative-step refusal");
    auto after = observe(*simulation);
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        require(frames[lev].step == after[lev].step && frames[lev].physicalTime == after[lev].physicalTime,
            "D3 rejected advance time unchanged");
        compare(frames[lev].populations,after[lev].populations,"D3 rejected populations unchanged",true);
        compare(frames[lev].macro,after[lev].macro,"D3 rejected macros unchanged",true);
    }
    write_static_snapshot("artifacts/layout/"+fixture.name+"/box"+std::to_string(fixture.spec.maxGridSize.front()[0]),
        fixture.name,fixture.spec,frames,"d3-periodic-xyz-v1");
    amrex::Print() << "d3_layout_pass sample=" << fixture.name
        << " levels=" << frames.size() << " box=" << fixture.spec.maxGridSize.front()[0] << '\n';
    return frames;
}

void reject(amr::StaticHierarchySpec const& spec, std::string const& code, int level = 0)
{
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(spec,Profile{});
    bool matched = false;
    for (auto const& error : prepared.errors)
        matched |= error.find(code+":") != std::string::npos && error.find("level="+std::to_string(level)) != std::string::npos;
    require(!prepared.ok() && !prepared.simulation && matched,"D3 missing refusal: "+code);
}

void negative_checks()
{
    auto spec = base_spec();
    spec.refRatio = amrex::IntVect(2,2,1); reject(spec,"invalid_ref_ratio");
    for (int axis = 0; axis < 3; ++axis) {
        spec = base_spec(); spec.nCell[axis] = 0; reject(spec,"invalid_mesh");
        spec = base_spec(); spec.nCell[axis] = 15; reject(spec,"unequal_active_spacing");
        spec = base_spec(); spec.maxGridSize[0][axis] = 7; reject(spec,"invalid_max_grid_size");
        spec = base_spec(); spec.probHi[axis] = 2; reject(spec,"unequal_active_spacing");
        spec = base_spec(); spec.probHi[axis] = std::numeric_limits<Real>::infinity(); reject(spec,"invalid_mesh");
        auto expanded = central_fixture(true);
        expanded.name = "D3-expanded-"+std::to_string(axis);
        expanded.spec.regions[0].box.lo[axis] = 0;
        expanded.spec.regions[0].box.hi[axis] = 0.5;
        expanded.spec.regions[1].box.lo[axis] = 0;
        expanded.spec.regions[1].box.hi[axis] = 0.25;
        run(expanded);
        for (Real hi : {Real(0.25),Real(0.125),Real(1.5),std::numeric_limits<Real>::quiet_NaN()}) {
            spec = central_fixture(false,2).spec;
            spec.regions.front().box.hi[axis] = hi;
            reject(spec,"invalid_region",1);
        }
        spec = central_fixture(false,2).spec;
        spec.regions.front().box.lo[axis] += 1.0/32;
        reject(spec,"unaligned_region",1);
    }
    spec = base_spec(); spec.coarseNestingMarginCells = 0; reject(spec,"invalid_nesting_controls");
    spec = base_spec(); spec.maxGridSize.clear(); reject(spec,"invalid_max_grid_size");
    spec = base_spec(); spec.gapFillRadiusCells = -1; reject(spec,"invalid_nesting_controls");
    spec = base_spec(); spec.gridEfficiency = 0; reject(spec,"invalid_grid_efficiency");
    spec = base_spec(); spec.probLo = {1e16,1e16,1e16}; spec.probHi = {1e16+2,1e16+2,1e16+2};
    reject(spec,"unrepresentable_geometry");
    spec = base_spec(); spec.nCell = {1<<21,1<<21,1<<21};
    reject(spec,"index_overflow");
    spec = central_fixture(false).spec; spec.regions.front().level = 30;
    reject(spec,"index_overflow",17);
    amrex::Print() << "d3_negative_checks: PASS\n";
}

} // namespace

void layout_checks()
{
    const bool mpi = amrex::ParallelDescriptor::NProcs() > 1;
    for (auto const& fixture : layout_fixtures(mpi)) run(fixture);
    negative_checks();
}

void input_checks()
{
    amrex::ParmParse pp("static_test");
    std::string name; pp.get("fixture",name);
    require(name == "D3-G3-A" || name == "D3-G3-P","D3 direct test fixture");
    auto direct = central_fixture(name == "D3-G3-P");
    auto loaded = direct; loaded.name += "-inputs";
    loaded.spec = topology::read_spec_inputs();
    auto expected = run(direct), actual = run(loaded);
    for (std::size_t lev = 0; lev < actual.size(); ++lev) compare(actual[lev],expected[lev],"D3 input/direct");
}

} // namespace lbm::tests::static_hierarchy::d3
