#include "integration/runtime/StaticHierarchy3DTests.H"
#include "integration/runtime/StaticHierarchy3DTestSupport.H"
#include "integration/runtime/StaticHierarchySnapshot.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

namespace lbm::tests::static_hierarchy::d3 {
namespace {

struct Trace {
    int completed = 0;
    std::vector<Frame> frames;
};

class ReferenceSession : public runtime::NoCaseRunSession<Descriptor> {
public:
    explicit ReferenceSession(std::shared_ptr<Trace> trace) : trace_(std::move(trace)) {}
    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = plan.macroAtInitialized = true;
        plan.completed.everyStep = plan.completedMacro.everyStep = true;
        return plan;
    }
    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    { save(state); return {}; }
    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        require(state.step() == ++trace_->completed,"D3 reference observation cadence");
        if (state.step() <= 2 || state.step() == 20) save(state);
        return {};
    }
private:
    void save(runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        close(state.physicalTime(),state.step()*timestep,"D3 reference time");
        auto const& context = state.context();
        trace_->frames.push_back(capture(state.step(),state.physicalTime(),context.geometry,
            state.populations().multifab(),&state.macro().multifab()));
    }
    std::shared_ptr<Trace> trace_;
};

struct ReferencePreparation {
    std::shared_ptr<Trace> trace;
    template<class> using RunSession = ReferenceSession;
    template<class View>
    runtime::SingleLevelCasePreparationResult<ReferenceSession> resolve(View const&) const
    { return {ReferenceSession(trace),{}}; }
};

std::vector<Frame> reference_run(bool uniform, int boxSize)
{
    runtime::SingleLevelProgram<Descriptor,BGKDynamicsSelection,
        initialization::ProfileInitializationSelection<Profile>,ReferencePreparation> program;
    auto spec = base_spec(boxSize);
    program.mesh.nCell = spec.nCell;
    program.mesh.envelopeLo = spec.probLo;
    program.mesh.envelopeHi = spec.probHi;
    program.mesh.isPeriodic = spec.isPeriodic;
    program.mesh.maxGridSize = boxSize;
    program.lbm = intent();
    program.run.maxStep = 20; program.run.finalTimePhys = 1.0;
    program.run.name = "static_hierarchy_d3_reference";
    program.initialization = initialization::from_profile(make_profile(spec,uniform));
    auto trace = std::make_shared<Trace>();
    program.casePreparation.trace = trace;
    auto prepared = runtime::prepare_single_level(std::move(program));
    for (auto const& error : prepared.diagnostics())
        amrex::Print() << error.code << ": " << error.message << '\n';
    require(prepared.ok(),"D3 reference preparation");
    auto result = runtime::run_single_level(std::move(prepared).take());
    require(result.ok() && result.finalStep == 20 && trace->completed == 20,"D3 reference completion");
    return trace->frames;
}

std::vector<Frame> amr_run(bool uniform, int boxSize, bool split)
{
    auto spec = base_spec(boxSize);
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(spec,make_profile(spec,uniform));
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(),"D3 single-level preparation");
    auto simulation = prepared.take();
    auto parameters = runtime::prepare_amr_evolution(*simulation,intent(),BGKDynamicsSelection{});
    require(parameters.ok(),"D3 single-level parameters");
    auto evolution = std::move(parameters).take();
    close(evolution.level(0).lbm.omega,1.0/0.65,"D3 BGK omega");
    close(evolution.level(0).dtPhys,timestep,"D3 physical timestep");
    auto frames = observe(*simulation,uniform);
    int calls = 0;
    auto callback = [&](View const& view) {
        ++calls;
        require(view.step == calls && view.levels.size() == 1,"D3 completion cadence");
        close(view.physicalTime,calls*timestep,"D3 accumulated time");
        if (calls <= 2 || calls == 20) {
            auto const& level = view.levels.front();
            auto frame = capture(view.step,view.physicalTime,level.geometry,
                level.populations.multifab(),&level.macro->multifab());
            check_fields(frame,uniform);
            frames.push_back(std::move(frame));
        }
    };
    require(simulation->evolve(evolution,0,true,callback).ok() && calls == 0,"D3 zero-step no-op");
    const auto negative = simulation->evolve(evolution,-1,true,callback);
    require(negative.error.find("negative_steps") != std::string::npos && calls == 0,"D3 negative steps");
    compare(frames.front(),observe(*simulation,uniform).front(),"D3 rejected run unchanged");
    require(simulation->evolve(evolution,split ? 2 : 20,true,callback).ok(),"D3 evolve");
    if (split) require(simulation->evolve(evolution,18,true,callback).ok(),"D3 split evolve");
    require(calls == 20,"D3 complete observations");
    return frames;
}

} // namespace

void single_level_checks()
{
    for (bool uniform : {true,false}) {
        std::vector<Frame> boxReference;
        for (int boxSize : {16,8}) {
            auto actual = amr_run(uniform,boxSize,false);
            auto old = reference_run(uniform,boxSize);
            require(actual.size() == 4 && old.size() == 4,"D3 requested snapshot steps");
            for (std::size_t i = 0; i < actual.size(); ++i) {
                compare(actual[i],old[i],"D3 AMR/old reference");
                if (!boxReference.empty()) compare(actual[i],boxReference[i],"D3 Box decomposition");
            }
            auto split = amr_run(uniform,boxSize,true);
            compare(actual.back(),split.back(),"D3 20 versus 2+18");
            if (boxReference.empty()) boxReference = actual;
            if (!uniform) {
                Real change = 0;
                for (std::size_t i = 0; i < actual.front().populations.values.size(); ++i)
                    change = std::max(change,std::abs(actual.front().populations.values[i]-actual.back().populations.values[i]));
                require(change > 1e-8,"D3 nonuniform state did not evolve");
            }
            const std::string name = uniform ? "D3-S0" : "D3-S1";
            const auto path = "artifacts/"+name+"/box"+std::to_string(boxSize);
            auto spec = base_spec(boxSize);
            const auto profile = name+"-bgk-nu0.05-uref0.02-v1";
            write_static_snapshot(path+"/amrcore",name,spec,actual,profile);
            write_static_snapshot(path+"/single_level",name,spec,old,profile);
            amrex::Print() << "d3_single_pass sample=" << name << " box=" << boxSize
                << " ranks=" << amrex::ParallelDescriptor::NProcs() << " steps=0,1,2,20 split=2+18\n";
        }
    }
}

} // namespace lbm::tests::static_hierarchy::d3
