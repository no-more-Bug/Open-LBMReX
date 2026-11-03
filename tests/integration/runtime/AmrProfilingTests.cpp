#include "integration/runtime/AmrProfilingTests.H"
#include "integration/runtime/AmrEvolutionTestCase.H"
#include "integration/runtime/AmrBoundaryEvolutionTests.H"
#include "lbm/profiling/AmrTimestepProfiler.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include <AMReX_Print.H>

namespace lbm::tests::amr_evolution {
namespace {
namespace fixture = lbm::tests::integration::amr_boundary_test;
using Mode = profiling::TimestepProfilingMode;
using Stage = profiling::AmrLevelStage;

struct Result {
    std::vector<Frame> fields;
    std::vector<Real> forces;
    std::optional<profiling::AmrPerformanceRecord> record;
};

Real exclusive_seconds(profiling::AmrProfileDurations const& sample)
{
    Real result = sample.hierarchy.seconds;
    for (auto const& level : sample.levels) for (auto const& item : level) result += item.seconds;
    for (auto const& pair : sample.pairs) for (auto const& item : pair) result += item.seconds;
    return result;
}

auto simulation(int levels)
{
    auto boundaries = fixture::evolution_boundaries(levels,true);
    boundaries.collectForces = true;
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(fixture::evolution_spec<Descriptor>(levels,true),
        fixture::BoundaryEvolutionProfile<Descriptor>{}, boundaries);
    for (auto const& error : prepared.errors) amrex::Print()<<error<<'\n';
    require(prepared.ok(),"profiling boundary hierarchy preparation");
    return prepared.take();
}

Result run(int levels, std::optional<Mode> mode)
{
    auto owner = simulation(levels);
    auto prepared = runtime::prepare_amr_evolution(*owner,evolution_intent(.8),BGKDynamicsSelection{});
    require(prepared.ok(),"profiling evolution preparation");
    auto evolution = std::move(prepared).take();
    profiling::AmrTimestepProfiler profiler(mode.value_or(Mode::Disabled),2,4,levels);
    Result result;
    int observations = 0;
    std::vector<int> fineCalls(levels,0);
    auto observer = [&](auto const& view) {
        ++observations;
        require(!view.forceBatch.empty(),"profiling force publication is missing");
        for (auto const& sample : view.forceBatch.boundarySteps)
            for (auto const& force : sample.boundaryForces)
                for (auto value : force.forceLb) result.forces.push_back(value);
        for (auto const& sample : view.forceBatch.objectSteps)
            for (auto const& force : sample.objectForces)
                for (auto value : force.forceLb) result.forces.push_back(value);
        for (auto const& sample : view.forceBatch.objectIntervals)
            for (auto const& force : sample.objectForces)
                for (auto value : force.forceLb) result.forces.push_back(value);
    };
    auto fineObserver = [&](auto const& view) { ++fineCalls[view.level]; };
    for (int segment=0; segment<2; ++segment) {
        auto status = mode ? owner->evolve(evolution,3,true,observer,fineObserver,&profiler)
                           : owner->evolve(evolution,3,true,observer,fineObserver);
        require(status.ok(),"profiling segmented evolution failed");
    }
    require(observations == 6,"profiling changed L0 observation count");
    for (int lev=1; lev<levels; ++lev) require(fineCalls[lev] == 6*(1<<lev),"profiling changed fine observer count");
    owner->observe(true,[&](auto const& view) { result.fields = capture(view); });
    result.record = profiler.performanceRecord();
    if (!mode || *mode == Mode::Disabled) {
        require(!result.record && exclusive_seconds(profiler.localDurations()) == 0,
                "disabled profiling collected a sample");
        for (auto steps : profiler.levelSteps()) require(steps == 0,"disabled profiling counted work");
        return result;
    }
    require(result.record.has_value(),"active profiling has no sample");
    auto const& record = *result.record;
    require(record.window.sampleStart == 3 && record.window.sampleEnd == 6 && record.window.nSteps == 4,
            "segmented L0 window was reset or finalized early");
    require(record.window.synchronizedWindow > 0,"missing synchronized window");
    for (int lev=0; lev<levels; ++lev)
        require(record.levelSteps[lev] == 4*(1<<lev),"N/2N/4N measured layer steps");
    if (*mode == Mode::Throughput) {
        require(record.window.formalMlups && record.criticalRank.levels.empty() && record.rankMax.levels.empty() &&
                exclusive_seconds(profiler.localDurations()) == 0,"Throughput collected stage measurements");
        return result;
    }
    require(!record.window.formalMlups && record.window.criticalRank >= 0 &&
            record.window.criticalRank < amrex::ParallelDescriptor::NProcs(),"invalid critical rank/mode");
    auto const& durations = record.criticalRank;
    require(int(durations.pairs.size()) == levels-1,"single-level pair degeneration");
    for (int lev=0; lev<levels; ++lev) {
        int const steps = 4*(1<<lev);
        auto const& values = durations.levels[lev];
        for (auto stage : {Stage::Collision,Stage::Communication,Stage::Streaming,Stage::BoundaryPost})
            require(values[static_cast<int>(stage)].calls == steps,"layer operation call count");
        require(values[static_cast<int>(Stage::BoundaryPre)].calls == (lev == 0 ? steps : 0),
                "physical ghosts must only belong to L0");
        require(values[static_cast<int>(Stage::Management)].calls == 2*steps+(lev ? steps/2 : 0),
                "commit/force/fine observation ownership");
        for (auto const& value : values)
            require(std::isfinite(value.seconds) && value.seconds >= 0,"invalid stage duration");
    }
    for (int lev=0; lev<levels-1; ++lev)
        for (auto const& value : durations.pairs[lev])
            require(value.calls == 4*(1<<lev) && std::isfinite(value.seconds) && value.seconds >= 0,
                    "pair operation ownership or duration");
    require(durations.hierarchy.calls == 8,"hierarchy synchronization/publication ownership");
    require(exclusive_seconds(durations) <= record.window.criticalRankWindow.total+1e-12,
            "exclusive stages overlap or use different-rank denominators");
    return result;
}

void reduction_check()
{
    int const rank = amrex::ParallelDescriptor::MyProc();
    profiling::TimestepProfileSample window;
    window.mode = Mode::StageBreakdown;
    window.sampleStart = window.sampleEnd = window.nSteps = 1;
    window.window.total = rank == 0 ? 10 : 9;
    window.synchronizedWindow = 11;
    profiling::AmrProfileDurations local;
    local.levels.resize(1);
    local.levels[0][0] = {rank == 0 ? Real(1) : Real(7),1};
    local.levels[0][1] = {rank == 0 ? Real(8) : Real(.5),1};
    local.hierarchy = {Real(.2),1};
    auto record = profiling::make_amr_performance_record(
        profiling::make_timestep_performance_record(window),{1},local);
    require(record.window.criticalRank == 0,"critical rank must use complete L0 total");
    close(record.criticalRank.levels[0][0].seconds,1,"critical collision");
    close(record.criticalRank.levels[0][1].seconds,8,"critical communication");
    close(record.window.criticalRankWindow.total,10,"common percentage denominator");
    close(1-exclusive_seconds(record.criticalRank)/record.window.criticalRankWindow.total,.08,"other share");
    close(record.rankMax.levels[0][0].seconds,amrex::ParallelDescriptor::NProcs() > 1 ? 7 : 1,"independent stage maximum");
    if (amrex::ParallelDescriptor::NProcs() > 1)
        require(exclusive_seconds(record.rankMax) > record.window.criticalRankWindow.total,
                "fixture must expose why independent maxima cannot form percentages");
}

void open_window_check()
{
    auto owner = simulation(1);
    auto prepared = runtime::prepare_amr_evolution(*owner,evolution_intent(.8),BGKDynamicsSelection{});
    require(prepared.ok(),"open profiling preparation");
    auto evolution = std::move(prepared).take();
    profiling::AmrTimestepProfiler profiler(Mode::Throughput,0,0,1);
    for (int segment=0; segment<2; ++segment)
        require(owner->evolve(evolution,1,false,[](auto const&) {},nullptr,&profiler).ok(),"open profiling evolution");
    profiler.finalize();
    auto record = profiler.performanceRecord();
    require(record && record->window.nSteps == 2 && record->window.sampleStart == 1 &&
            record->window.sampleEnd == 2,"case-owned open window continuation");
}
} // namespace

void profiling_checks()
{
    int calls = 0;
    close(profiling::TimestepProfiler::measureOperation(false,[&] { ++calls; }),0,"disabled operation time");
    require(calls == 1,"disabled timer must execute the operation exactly once");
    for (int levels : {1,3}) {
        auto baseline = run(levels,std::nullopt);
        for (auto mode : {Mode::Disabled,Mode::Throughput,Mode::StageBreakdown}) {
            auto actual = run(levels,mode);
            compare_frames(actual.fields,baseline.fields,"profiling mode terminal state");
            require(actual.forces.size() == baseline.forces.size(),"profiling force sample count");
            for (std::size_t i=0; i<actual.forces.size(); ++i)
                close(actual.forces[i],baseline.forces[i],"profiling raw and Object forces");
        }
    }
    reduction_check();
    open_window_check();
    amrex::Print()<<"amr_profiling_pass descriptor="<<Descriptor::name()
                  <<" levels=1,3 measured_steps=4,8,16 modes=disabled,throughput,stage_breakdown\n";
}
} // namespace lbm::tests::amr_evolution
