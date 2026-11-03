#include "integration/runtime/StaticHierarchyTopologyTests.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "integration/runtime/StaticHierarchyTopologyOracle.H"
#include "integration/runtime/StaticHierarchySnapshot.H"
#include "integration/runtime/StaticHierarchy3DTestSupport.H"

#include <AMReX_ParmParse.H>
#include <chrono>
#include <sys/resource.h>

namespace lbm::tests::static_hierarchy::topology {
namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = D3Q27Descriptor;
using Profile = d3::Profile;
#else
using Descriptor = D2Q9Descriptor;
struct Profile {
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const noexcept
    {
        constexpr Real twoPi = 6.283185307179586476925286766559;
        return {1+0.001*std::sin(twoPi*p[0])*std::cos(twoPi*p[1]),
            0.01*std::cos(twoPi*p[0]),0.005*std::sin(twoPi*p[1]),0};
    }
};
#endif
using Simulation = amr::LbmAmrSimulation<Descriptor,Profile>;
using View = amr::LbmAmrView<Descriptor>;
constexpr int dimensions = Descriptor::d;
constexpr int qCount = Descriptor::q;

void check_fields(Frame const& frame)
{
#ifdef LBM_DESCRIPTOR_D3Q27
    d3::check_fields(frame);
#else
    check_moments(frame);
    auto const& f = frame.populations;
    const Real twoPi = 2*std::acos(Real(-1));
    for (std::size_t row = 0; row < f.owners.size(); ++row) {
        auto const cell = f.cell_at(row);
        const Real x = twoPi*(cell[0]+0.5)/f.domain.length(0);
        const Real y = twoPi*(cell[1]+0.5)/f.domain.length(1);
        const Real rho = 1+0.001*std::sin(x)*std::cos(y);
        const Real ux = 0.01*std::cos(x), uy = 0.005*std::sin(y);
        close(frame.macro.values[row*4],rho,"topology D2 analytic rho");
        close(frame.macro.values[row*4+1],ux,"topology D2 analytic ux");
        close(frame.macro.values[row*4+2],uy,"topology D2 analytic uy");
        for (int q = 0; q < 9; ++q) {
            const Real cu = cx[q]*ux+cy[q]*uy;
            close(f.values[row*9+q],weights[q]*rho*(1+3*cu+4.5*cu*cu-1.5*(ux*ux+uy*uy)),
                "topology D2 independent equilibrium");
        }
    }
#endif
}

std::vector<Frame> observe(Simulation& simulation, Fixture const& fixture)
{
    std::vector<Frame> frames;
    simulation.observe(true,[&](View const& view) {
        require(view.step == 0 && view.physicalTime == 0,"topology initialization time");
        for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
            auto const& level = view.levels[lev];
            auto const& f = level.populations.multifab();
            require(level.macro && f.boxArray() == level.boxes &&
                f.DistributionMap() == level.mapping,"topology final population BA/DM");
            require(level.macro->multifab().boxArray() == level.boxes &&
                level.macro->multifab().DistributionMap() == level.mapping,"topology final macro BA/DM");
            for (int d = 0; d < 3; ++d) {
                const int scale = fixture.spec.refRatio[d] == 2 ? 1 << lev : 1;
                require(level.geometry.Domain().length(d) == fixture.spec.nCell[d]*scale &&
                    level.geometry.Domain().smallEnd(d) == 0 && level.geometry.isPeriodic(d),"topology actual Domain");
                close(level.geometry.CellSize(d),
                    (fixture.spec.probHi[d]-fixture.spec.probLo[d])/(fixture.spec.nCell[d]*scale),
                    "topology directional spacing");
            }
            if (lev) require(view.refRatios[lev-1] == fixture.spec.refRatio,"topology actual refinement ratio");
            auto frame = capture(level.step,level.physicalTime,level.geometry,f,&level.macro->multifab());
            frame.level = int(lev);
            check_fields(frame);
            frames.push_back(std::move(frame));
        }
    });
    return frames;
}

int check_donors(Simulation& simulation, Fixture const& fixture, std::vector<Frame> const& frames)
{
    int remote = 0, remoteZ = 0, diagonal = 0, sameLevel = 0, coarseFine = 0, crossCoarseBoxes = 0;
    simulation.observe(false,[&](View const& view) {
        for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
            auto const& level = view.levels[lev];
            auto const& reference = frames[lev].populations;
            if (lev) for (int b = 0; b < level.boxes.size(); ++b)
                if (view.levels[lev-1].boxes.intersections(amrex::coarsen(level.boxes[b],fixture.spec.refRatio)).size() > 1)
                    ++crossCoarseBoxes;
            auto host = host_field_snapshot(level.populations.multifab());
            for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
                auto data = host.const_array(mfi);
                for (amrex::BoxIterator it(amrex::grow(mfi.validbox(),host.nGrowVect())); it.ok(); ++it) {
                    auto const cell = it();
                    if (mfi.validbox().contains(cell)) continue;
                    auto donor = cell;
                    for (int d = 0; d < 3; ++d) {
                        const int n = reference.domain.length(d);
                        donor[d] = (donor[d]%n+n)%n;
                    }
                    const auto row = reference.offset(donor);
                    if (row == reference.owners.size()) {
                        require(lev > 0,"topology L0 ghost misses periodic donor");
                        auto const& coarse = frames[lev-1].populations;
                        require(coarse.offset(donor/fixture.spec.refRatio) < coarse.owners.size(),"topology missing coarse donor");
                        ++coarseFine;
                        continue;
                    }
                    ++sameLevel;
                    const auto owner = level.boxes.intersections(amrex::Box(donor,donor));
                    const bool isRemote = lev && level.mapping[owner.front().first] != amrex::ParallelDescriptor::MyProc();
                    if (isRemote) {
                        ++remote;
                        if (cell[2] < mfi.validbox().smallEnd(2) || cell[2] > mfi.validbox().bigEnd(2)) ++remoteZ;
                    }
                    for (int q = 0; q < qCount; ++q) {
                        require(data(cell,q) == reference.values[row*qCount+q],"topology complete ghost population copy");
                        amrex::IntVect direction;
#ifdef LBM_DESCRIPTOR_D3Q27
                        auto const c = d3::directions[q];
                        direction = amrex::IntVect(c[0],c[1],c[2]);
#else
                        direction = amrex::IntVect(cx[q],cy[q],0);
#endif
                        const int norm = std::abs(direction[0])+std::abs(direction[1])+std::abs(direction[2]);
                        if (isRemote && norm > 1 && mfi.validbox().contains(cell+direction)) ++diagonal;
                    }
                }
            }
        }
    });
    for (int* count : {&remote,&remoteZ,&diagonal,&sameLevel,&coarseFine})
        amrex::ParallelDescriptor::ReduceIntSum(*count);
    require(sameLevel > 0,"topology did not exercise same-level donors");
    if (fixture.requireRemoteDonor && amrex::ParallelDescriptor::NProcs() > 1)
        require(remote && diagonal && (dimensions == 2 || remoteZ),"topology did not exercise remote directional donors");
    amrex::Print() << "topology_donors sample=" << fixture.name << " same=" << sameLevel
        << " coarse=" << coarseFine << " remote=" << remote << " remote_z=" << remoteZ
        << " diagonal=" << diagonal << " fine_boxes_across_coarse_boxes=" << crossCoarseBoxes << '\n';
    return crossCoarseBoxes;
}

struct RunResult { std::vector<Frame> frames; int crossCoarseBoxes; };

RunResult run(Fixture const& fixture)
{
    amrex::Print() << "topology_begin sample=" << fixture.name << '\n';
    std::ostringstream messages;
    auto& output = amrex::OutStream();
    auto* previous = output.rdbuf(messages.rdbuf());
    const auto start = std::chrono::steady_clock::now();
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(fixture.spec,Profile{});
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    output.rdbuf(previous);
    amrex::Print() << messages.str();
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(),"topology real preparation failed: "+fixture.name);
    int passes = 0;
    const std::string marker = "static_hierarchy_initialized passes=";
    const auto at = messages.str().find(marker);
    if (at != std::string::npos) passes = std::stoi(messages.str().substr(at+marker.size()));
    amrex::ParallelDescriptor::ReduceIntMax(passes);
    if (fixture.requireFeedback) require(passes >= 2,"candidate did not exercise actual AMReX feedback: "+fixture.name);
    auto simulation = prepared.take();
    auto frames = observe(*simulation,fixture);
    validate_coverage(fixture,frames,dimensions);
    if (fixture.expectedFinestTagCells >= 0)
        require(frames.back().populations.owners.size() ==
            std::size_t(fixture.expectedFinestTagCells)*(1 << dimensions),"hand-counted topology example changed");
    const int crossCoarseBoxes = check_donors(*simulation,fixture,frames);
    config::LbmConfigIntent intent;
    intent.parameterizationKind = config::LbmParameterizationKind::FromTau;
    intent.tau = 0.65; intent.uRefLb = 0.02; intent.uRefPhys = 1.0;
    auto parameters = runtime::prepare_amr_evolution(*simulation,intent,BGKDynamicsSelection{});
    require(parameters.ok(),"topology evolution preparation");
    auto evolution = std::move(parameters).take();
    int observations = 0;
    auto callback = [&](View const&) { ++observations; };
    require(simulation->evolve(evolution,0,false,callback).ok(),"topology zero-step no-op");
    const auto rejected = simulation->evolve(evolution,-1,false,callback);
    require(rejected.error.find("negative_steps") != std::string::npos && observations == 0,
        "topology negative-step advance must fail before state changes");
    auto after = observe(*simulation,fixture);
    for (std::size_t lev = 0; lev < frames.size(); ++lev) {
        require(frames[lev].step == after[lev].step && frames[lev].physicalTime == after[lev].physicalTime,
            "topology rejected time unchanged");
        compare(frames[lev].populations,after[lev].populations,"topology rejected populations unchanged",true);
        compare(frames[lev].macro,after[lev].macro,"topology rejected macros unchanged",true);
    }
    const auto path = "artifacts/topology/"+fixture.name+"/box"+std::to_string(fixture.spec.maxGridSize.front()[0]);
    write_static_snapshot(path,fixture.name,fixture.spec,frames,
        dimensions == 3 ? "d3-periodic-xyz-v1" : "layout-periodic-v1");
    amrex::Print() << "topology_pass sample=" << fixture.name << " levels=" << frames.size()
        << " passes=" << passes << " preparation_seconds=" << seconds << '\n';
    return {std::move(frames),crossCoarseBoxes};
}

void compare_frames(std::vector<Frame> const& expected, std::vector<Frame> const& actual)
{
    require(expected.size() == actual.size(),"topology equivalent request level count");
    for (std::size_t lev = 0; lev < actual.size(); ++lev) compare(expected[lev],actual[lev],"equivalent topology request");
}

class CostProbe final : public Simulation {
public:
    CostProbe(amr::StaticHierarchyLayout layout) : Simulation(std::move(layout),Profile{}) {}
    std::vector<std::string> prepare() { return initialize({}); }
    int scratches = 0;
protected:
    void MakeNewLevelFromScratch(int lev, Real time, amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm) override
    { ++scratches; Simulation::MakeNewLevelFromScratch(lev,time,ba,dm); }
};

} // namespace

void run_checks()
{
    const bool mpi = amrex::ParallelDescriptor::NProcs() > 1;
    bool supplementOnly = false;
    amrex::ParmParse("static_test").query("supplement_only",supplementOnly);
    if (supplementOnly) {
        for (auto const& fixture : supplement_fixtures(dimensions,mpi)) {
            auto reference = run(fixture);
            auto smaller = fixture;
            for (int d = 0; d < dimensions; ++d) smaller.spec.maxGridSize[0][d] = 16;
            smaller.requireRemoteDonor = mpi;
            compare_frames(reference.frames,run(smaller).frames);
            if (!mpi && fixture.name.find("-half-x") == std::string::npos) {
                auto permuted = fixture;
                permuted.name += "-permuted";
                std::reverse(permuted.spec.regions.begin(),permuted.spec.regions.end());
                permuted.spec.regions.push_back(permuted.spec.regions.front());
                compare_frames(reference.frames,run(permuted).frames);
            }
        }
        return;
    }
    int crossCoarseBoxes = 0;
    for (auto fixture : fixtures(dimensions,mpi)) {
        // These requests have one fine Box at size 32. Use size 16 in both
        // serial/MPI donor comparisons; output fixtures keep their original size.
        if (fixture.name.find("-Sparse-L1-L3") != std::string::npos ||
            fixture.name.find("-Through-") != std::string::npos)
            for (int d = 0; d < dimensions; ++d) fixture.spec.maxGridSize[0][d] = 16;
        auto reference = run(fixture);
        crossCoarseBoxes += reference.crossCoarseBoxes;
        const bool splitBoxes = fixture.name.find("Gap-8") != std::string::npos ||
            fixture.name.find("Vertex") != std::string::npos || fixture.requireFeedback;
        if (splitBoxes) {
            auto smaller = fixture;
            for (int d = 0; d < dimensions; ++d)
                smaller.spec.maxGridSize[0][d] = fixture.name.find("Vertex") != std::string::npos
                    ? 16 : smaller.spec.maxGridSize[0][d]/2;
            auto actual = run(smaller);
            if (fixture.spec.gridEfficiency == 1) compare_frames(reference.frames,actual.frames);
        }
        if (!mpi && fixture.name.find("-Union") != std::string::npos) {
            auto permuted = fixture; permuted.name += "-permuted";
            std::reverse(permuted.spec.regions.begin(),permuted.spec.regions.end());
            compare_frames(reference.frames,run(permuted).frames);
            auto split = fixture; split.name += "-split";
            split.spec.regions.resize(2);
            auto right = split.spec.regions.front();
            const Real middle = (right.box.lo[0]+right.box.hi[0])/2;
            right.box.lo[0] = middle; split.spec.regions.front().box.hi[0] = middle;
            split.spec.regions.push_back(right);
            compare_frames(reference.frames,run(split).frames);
        }
        if (!mpi && (fixture.name.find("-Vertex") != std::string::npos ||
            fixture.name.find("-PeriodicFace-Seam") != std::string::npos)) {
            for (int sign : {-1,1}) {
                auto shifted = fixture; shifted.name += sign < 0 ? "-shift-minus" : "-shift-plus";
                for (auto& request : shifted.spec.regions) for (int d = 0; d < dimensions; ++d) {
                    request.box.lo[d] += sign; request.box.hi[d] += sign;
                }
                compare_frames(reference.frames,run(shifted).frames);
            }
        }
    }
    require(crossCoarseBoxes > 0,"topology fixture set lacks fine Box spanning multiple coarse Boxes");
}

void input_checks()
{
    std::string name;
    amrex::ParmParse("static_test").get("fixture",name);
    const auto candidates = fixtures(dimensions);
    auto found = std::find_if(candidates.begin(),candidates.end(),[&](Fixture const& f) { return f.name == name; });
    require(found != candidates.end(),"unknown topology input fixture");
    auto loaded = *found;
    loaded.name += "-inputs";
    loaded.spec = read_spec_inputs();
    compare_frames(run(*found).frames,run(loaded).frames);
}

void cost_checks()
{
    std::string selected;
    amrex::ParmParse("static_test").query("fixture",selected);
    for (auto const& fixture : fixtures(dimensions)) {
        if (!selected.empty() && fixture.name != selected) continue;
        if (fixture.name.find("-Union") == std::string::npos &&
            fixture.name.find("-Sparse-L1-L3") == std::string::npos && !fixture.requireFeedback) continue;
        amrex::Print() << "topology_cost_begin sample=" << fixture.name << '\n';
        const auto start = std::chrono::steady_clock::now();
        auto layout = amr::prepare_static_layout(fixture.spec,dimensions);
        require(layout.ok(),"cost probe layout preparation");
        CostProbe simulation(std::move(*layout.layout));
        require(simulation.prepare().empty(),"cost probe initialization");
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        struct rusage usage{};
        require(getrusage(RUSAGE_SELF,&usage) == 0,"cost probe resource query");
        amrex::Print() << "topology_cost sample=" << fixture.name << " scratch_calls=" << simulation.scratches
            << " preparation_seconds=" << seconds << " process_peak_rss_kib=" << usage.ru_maxrss
            << " note=cumulative_process_peak\n";
        validate_coverage(fixture,observe(simulation,fixture),dimensions);
    }
}

} // namespace lbm::tests::static_hierarchy::topology
