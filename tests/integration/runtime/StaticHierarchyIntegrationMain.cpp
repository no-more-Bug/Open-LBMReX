#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "integration/runtime/StaticHierarchyLayoutTests.H"
#include "integration/runtime/StaticHierarchyLifecycleTests.H"
#include "integration/runtime/StaticHierarchyOutputTests.H"
#include "integration/runtime/StaticHierarchySnapshot.H"
#include "integration/runtime/StaticHierarchy3DTests.H"
#include "integration/runtime/StaticHierarchyTopologyTests.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <cstdlib>
#include <memory>

#ifndef LBM_DESCRIPTOR_D3Q27
namespace lbm::tests::static_hierarchy {
namespace {

struct Profile {
    bool tgv;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<Real, AMREX_SPACEDIM> const& position, Real) const noexcept
    {
        if (!tgv) return {1.0, speed, 0.0, 0.0};
        constexpr Real twoPi = 6.283185307179586476925286766559;
        const Real x = twoPi * position[0], y = twoPi * position[1];
        return {1.0 - 0.75 * speed * speed * (std::cos(2*x) + std::cos(2*y)),
            -speed * std::cos(x) * std::sin(y), speed * std::sin(x) * std::cos(y), 0.0};
    }
};

config::LbmConfigIntent intent()
{
    config::LbmConfigIntent result;
    result.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    result.rhoRefLb = 1.0;
    result.uRefLb = speed;
    result.uRefPhys = 1.0;
    result.nuLb = 0.05;
    result.lengthRefLb = 32.0;
    result.lengthRefPhys = 1.0;
    return result;
}

void check_geometry(amrex::Geometry const& geometry, amrex::BoxArray const& boxes,
                    amrex::DistributionMapping const& mapping,
                    amrex::MultiFab const& populations, int maxGridSize)
{
    require(geometry.Domain() == amrex::Box(amrex::IntVect(0,0,0), amrex::IntVect(31,31,0)),
        "single-level domain mismatch");
    require(populations.boxArray() == boxes && populations.DistributionMap() == mapping,
        "published fields do not use the final hierarchy layout");
    for (int d = 0; d < 3; ++d) {
        close(geometry.ProbLo(d), 0.0, "physical lower bound");
        close(geometry.ProbHi(d), d == 2 ? 1.0/16.0 : 1.0, "physical upper bound");
        close(geometry.CellSize(d), d == 2 ? 1.0/16.0 : 1.0/32.0, "directional cell size");
        require(geometry.isPeriodic(d), "periodic geometry flag lost");
    }
    for (int i = 0; i < boxes.size(); ++i) {
        require(boxes[i].ixType().cellCentered() && boxes[i].length(2) == 1 &&
            boxes[i].length(0) <= maxGridSize && boxes[i].length(1) <= maxGridSize,
            "unexpected cell centering or Box dimensions");
    }
}

void check_frame(Frame const& frame, bool tgv)
{
    close(frame.physicalTime, frame.step * dtPhys, "accumulated physical time");
    for (int count : frame.populations.owners) require(count == 1, "L0 coverage has missing cells");
    check_moments(frame);
    if (tgv && frame.step != 0) return;
    const Real pi = std::acos(Real(-1.0));
    for (int j = 0; j < 32; ++j) for (int i = 0; i < 32; ++i) {
        const int cell = j * 32 + i;
        const Real x = 2*pi*(i+0.5)/32, y = 2*pi*(j+0.5)/32;
        const Real rho = tgv ? 1.0 - 0.75*speed*speed*(std::cos(2*x)+std::cos(2*y)) : 1.0;
        const Real ux = tgv ? -speed*std::cos(x)*std::sin(y) : speed;
        const Real uy = tgv ? speed*std::sin(x)*std::cos(y) : 0.0;
        close(frame.macro.values[cell * 4], rho, "analytic density");
        close(frame.macro.values[cell * 4 + 1], ux, "analytic ux");
        close(frame.macro.values[cell * 4 + 2], uy, "analytic uy");
        for (int q = 0; q < 9; ++q) {
            const Real cu = cx[q]*ux + cy[q]*uy;
            close(frame.populations.values[cell * 9 + q],
                weights[q]*rho*(1.0 + 3*cu + 4.5*cu*cu - 1.5*(ux*ux+uy*uy)),
                "independent initial/uniform equilibrium");
        }
    }
}

struct OldTrace {
    int starts = 0;
    int completed = 0;
    int maxGridSize = 32;
    std::vector<Frame> frames;
};

class OldSession : public runtime::NoCaseRunSession<D2Q9Descriptor> {
public:
    explicit OldSession(std::shared_ptr<OldTrace> trace) : trace_(std::move(trace)) {}
    runtime::SingleLevelRunSessionResult start(runtime::SingleLevelRunContext<D2Q9Descriptor> const& context)
    {
        ++trace_->starts;
        close(context.lbm.tau, 0.65, "old tau");
        return {};
    }
    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = plan.macroAtInitialized = true;
        plan.completed.everyStep = plan.completedMacro.everyStep = true;
        return plan;
    }
    runtime::SingleLevelRunSessionResult observeInitialized(runtime::SingleLevelPublishedState<D2Q9Descriptor> const& state)
    { save(state); return {}; }
    runtime::SingleLevelCompletedObservationResult observeCompleted(runtime::SingleLevelPublishedState<D2Q9Descriptor> const& state)
    {
        require(state.step() == ++trace_->completed, "old completed observation cadence");
        close(state.latticeTime(), state.step(), "old lattice time");
        close(state.physicalTime(), state.step() * dtPhys, "old physical time");
        if (state.step() <= 2 || state.step() == 20) save(state);
        return {};
    }
private:
    void save(runtime::SingleLevelPublishedState<D2Q9Descriptor> const& state)
    {
        auto const& context = state.context();
        check_geometry(context.geometry, context.boxArray, context.distributionMapping,
            state.populations().multifab(), trace_->maxGridSize);
        require(state.hasMacro(), "old observer missing requested macro");
        trace_->frames.push_back(capture(state.step(), state.physicalTime(), context.geometry,
            state.populations().multifab(), &state.macro().multifab()));
    }
    std::shared_ptr<OldTrace> trace_;
};

struct OldPreparation {
    std::shared_ptr<OldTrace> trace;
    template<class> using RunSession = OldSession;
    template<class View>
    runtime::SingleLevelCasePreparationResult<OldSession> resolve(View const&) const
    { return {OldSession(trace), {}}; }
};

std::vector<Frame> run_old(bool tgv, int maxGridSize)
{
    runtime::SingleLevelProgram<D2Q9Descriptor, BGKDynamicsSelection,
        initialization::ConfiguredInitializationSelection, OldPreparation> program;
    program.mesh.nCell = {32,32,1};
    program.mesh.envelopeLo = {0.0,0.0,0.0};
    program.mesh.envelopeHi = {1.0,1.0,1.0/16.0};
    program.mesh.isPeriodic = {1,1,1};
    program.mesh.maxGridSize = maxGridSize;
    program.lbm = intent();
    program.run.maxStep = 20;
    program.run.finalTimePhys = 1.0;
    program.run.name = "static_hierarchy_single_level_reference";
    program.initialization.intent.kind = tgv ? initialization::InitializationKind::TaylorGreen
                                            : initialization::InitializationKind::Uniform;
    program.initialization.intent.rhoLb = 1.0;
    if (tgv) program.initialization.intent.taylorGreenVelocityAmplitudeLb = speed;
    else program.initialization.intent.uniformULb = amrex::Array<Real,3>{speed,0.0,0.0};
    auto trace = std::make_shared<OldTrace>();
    trace->maxGridSize = maxGridSize;
    program.casePreparation.trace = trace;
    auto prepared = runtime::prepare_single_level(std::move(program));
    for (auto const& diagnostic : prepared.diagnostics())
        amrex::Print() << diagnostic.code << ": " << diagnostic.message << '\n';
    require(prepared.ok(), "old full single-level preparation failed");
    auto result = runtime::run_single_level(std::move(prepared).take());
    require(result.ok() && result.finalStep == 20 && trace->starts == 1 && trace->completed == 20,
        "old full single-level run did not complete");
    return trace->frames;
}

Frame observe(amr::LbmAmrSimulation<D2Q9Descriptor, Profile>& simulation, bool macro, int maxGridSize)
{
    Frame frame;
    simulation.observe(macro, [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        require(view.levels.size() == 1, "single-level view has unexpected levels");
        auto const& level = view.levels.front();
        require((level.macro != nullptr) == macro && level.step == view.step &&
            level.physicalTime == view.physicalTime, "observation metadata or macro request mismatch");
        check_geometry(level.geometry, level.boxes, level.mapping, level.populations.multifab(), maxGridSize);
        frame = capture(view.step, view.physicalTime, level.geometry, level.populations.multifab(),
            level.macro ? &level.macro->multifab() : nullptr);
    });
    return frame;
}

std::vector<Frame> run_new(bool tgv, int maxGridSize, bool split)
{
    amr::StaticHierarchySpec spec;
    spec.maxGridSize = {amrex::IntVect(maxGridSize, maxGridSize, 1)};
    auto prepared = amr::prepare_static_hierarchy<D2Q9Descriptor>(spec, Profile{tgv});
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(), "new AmrCore preparation failed");
    auto simulation = prepared.take();
    require(simulation->numLevels() == 1, "new max_level=0 did not create L0");
    auto parameters = runtime::prepare_amr_evolution(*simulation, intent(), BGKDynamicsSelection{});
    require(parameters.ok(), "new single-level evolution preparation failed");
    auto evolution = std::move(parameters).take();
    close(evolution.level(0).lbm.omega, 1.0/0.65, "new omega");
    close(evolution.level(0).dtPhys, dtPhys, "new physical timestep");
    const auto noMacro = observe(*simulation, false, maxGridSize);
    auto initial = observe(*simulation, true, maxGridSize);
    compare(noMacro.populations, initial.populations, "first macro observation", true);
    const auto afterMacro = observe(*simulation, false, maxGridSize);
    compare(initial.populations, afterMacro.populations, "populations-only observation", true);
    require(initial.step == 0 && noMacro.step == 0 && afterMacro.step == 0 &&
        initial.physicalTime == 0.0 && noMacro.physicalTime == 0.0 && afterMacro.physicalTime == 0.0,
        "initial observations modified time or step");
    std::vector<Frame> frames{initial};
    int calls = 0;
    auto callback = [&](amr::LbmAmrView<D2Q9Descriptor> const& view) {
        require(view.step == ++calls, "new completed observation cadence");
        close(view.physicalTime, calls * dtPhys, "new observed physical time");
        auto const& level = view.levels.front();
        require((level.macro != nullptr) == !split, "evolve ignored macro request");
        if (!split && (view.step <= 2 || view.step == 20)) {
            frames.push_back(capture(view.step, view.physicalTime, level.geometry,
                level.populations.multifab(), &level.macro->multifab()));
        }
    };
    if (split) {
        require(simulation->evolve(evolution, 2, false, callback).ok(), "first split evolve failed");
        const auto refreshed = observe(*simulation, true, maxGridSize);
        require(refreshed.step == 2, "split evolve lost accumulated step");
        check_frame(refreshed, tgv);
        require(simulation->evolve(evolution, 18, false, callback).ok(), "second split evolve failed");
        frames.push_back(observe(*simulation, true, maxGridSize));
    } else {
        require(simulation->evolve(evolution, 20, true, callback).ok(), "continuous evolve failed");
    }
    require(calls == 20, "new evolve missing observations");
    const auto beforeNoop = observe(*simulation, true, maxGridSize);
    int noopCalls = 0;
    auto noopObserver = [&](amr::LbmAmrView<D2Q9Descriptor> const&) { ++noopCalls; };
    require(simulation->evolve(evolution, 0, true, noopObserver).ok(), "zero-step evolve failed");
    const auto zero = observe(*simulation, true, maxGridSize);
    auto negative = simulation->evolve(evolution, -1, true, noopObserver);
    require(!negative.ok() && negative.error.find("negative_steps") != std::string::npos,
        "negative steps were not explicitly rejected");
    const auto rejected = observe(*simulation, true, maxGridSize);
    require(noopCalls == 0 && zero.step == beforeNoop.step && rejected.step == beforeNoop.step &&
        zero.physicalTime == beforeNoop.physicalTime && rejected.physicalTime == beforeNoop.physicalTime,
        "zero/negative steps changed state or called observer");
    compare(beforeNoop.populations, zero.populations, "zero-step state", true);
    compare(beforeNoop.populations, rejected.populations, "negative-step state", true);
    for (auto const& frame : frames) check_frame(frame, tgv);
    return frames;
}

void single_level_checks()
{
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts");
    amrex::ParallelDescriptor::Barrier();
    for (bool tgv : {false, true}) {
        std::vector<Frame> boxReference;
        const std::string name = tgv ? "tgv" : "uniform";
        for (int maxGridSize : {32, 8}) {
            auto old = run_old(tgv, maxGridSize);
            auto actual = run_new(tgv, maxGridSize, false);
            require(old.size() == 4 && actual.size() == 4, "missing step 0/1/2/20 snapshots");
            constexpr std::array<int,4> expectedSteps{0,1,2,20};
            for (std::size_t i = 0; i < actual.size(); ++i) {
                require(actual[i].step == expectedSteps[i], "unexpected sample step");
                check_frame(old[i], tgv);
                compare(old[i], actual[i], "old/new " + name);
                if (!boxReference.empty()) compare(boxReference[i], actual[i], "Box decomposition " + name);
            }
            if (boxReference.empty()) boxReference = actual;
            if (tgv) {
                auto split = run_new(true, maxGridSize, true);
                compare(actual.back(), split.back(), "20 versus 2+18 evolve");
                Real initialEnergy = 0.0, finalEnergy = 0.0, maxChange = 0.0;
                for (int cell = 0; cell < 32*32; ++cell) {
                    for (int d = 1; d <= 2; ++d) {
                        const Real a = actual.front().macro.values[cell*4+d];
                        const Real b = actual.back().macro.values[cell*4+d];
                        initialEnergy += a*a; finalEnergy += b*b;
                        maxChange = std::max(maxChange, std::abs(a-b));
                    }
                }
                require(finalEnergy < initialEnergy && maxChange > 1.0e-8,
                    "TGV did not evolve and decay");
            }
            const auto directory = "artifacts/" + name + "/box" + std::to_string(maxGridSize);
            amr::StaticHierarchySpec spec;
            spec.maxGridSize = {amrex::IntVect(maxGridSize,maxGridSize,1)};
            const std::string profile = name + "-bgk-nu0.05-u0.02-v1";
            write_static_snapshot(directory + "/amrcore", name, spec, actual, profile);
            write_static_snapshot(directory + "/single_level", name, spec, old, profile);
            amrex::Print() << "static_single_level_pass sample=" << name
                << " max_grid_size=" << maxGridSize << " ranks=" << amrex::ParallelDescriptor::NProcs()
                << " steps=0,1,2,20\n";
        }
    }
}

} // namespace
} // namespace lbm::tests::static_hierarchy
#endif

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        using namespace lbm::tests::static_hierarchy;
        std::string mode = "single_level";
        amrex::ParmParse("static_test").query("mode", mode);
#ifdef LBM_DESCRIPTOR_D3Q27
        if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts");
        amrex::ParallelDescriptor::Barrier();
        if (mode == "topology") topology::run_checks();
        else if (mode == "topology_inputs") topology::input_checks();
        else if (mode == "topology_cost") topology::cost_checks();
        else if (mode == "single_level") d3::single_level_checks();
        else if (mode == "layout") d3::layout_checks();
        else if (mode == "layout_inputs") d3::input_checks();
        else if (mode == "lifecycle") d3::lifecycle_checks();
        else if (mode == "output") d3::output_checks();
#else
        if (mode == "topology") topology::run_checks();
        else if (mode == "topology_inputs") topology::input_checks();
        else if (mode == "topology_cost") topology::cost_checks();
        else if (mode == "single_level") single_level_checks();
        else if (mode == "layout") run_static_layout_checks();
        else if (mode == "layout_inputs") run_static_input_checks();
        else if (mode == "lifecycle") run_static_lifecycle_checks();
        else if (mode == "output") run_static_output_checks();
#endif
        else require(false, "unknown static hierarchy test mode");
        amrex::Print() << "Static hierarchy checks: PASS mode=" << mode << '\n';
    }
    amrex::Finalize();
    return 0;
}
