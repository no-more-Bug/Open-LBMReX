#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/dynamics/DynamicsPreparation.H"
#include "lbm/dynamics/DynamicsModel.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/fields/MacroField.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include <cmath>
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

void print_preparation_diagnostics(
    std::vector<lbm::runtime::SingleLevelSetupDiagnostic> const& diagnostics)
{
    for (auto const& diagnostic : diagnostics) {
        amrex::Print() << "[" << lbm::runtime::to_string(diagnostic.stage)
                       << ":" << diagnostic.code << ":"
                       << diagnostic.subject << "] "
                       << diagnostic.message << "\n";
    }
}

struct RecordingState {
    lbm::runtime::SingleLevelObservationPlan plan;
    int starts = 0;
    int initializations = 0;
    int completedSteps = 0;
    int finalizations = 0;
    int performanceWrites = 0;
    bool failFinal = false;
    bool recordPopulationSums = false;
    bool recordRestState = false;
    bool expectDomainWalls = false;
    bool resolvedDomainWallsValid = true;
    int stopAtStep = -1;
    amrex::Real initializedPopulationSum = 0.0;
    amrex::Real initializedRestPopulationError = 0.0;
    amrex::Real initializedRestMacroError = 0.0;
    std::vector<amrex::Real> completedRestPopulationErrors;
    std::vector<amrex::Real> completedRestMacroErrors;
    std::vector<amrex::Real> completedPopulationSums;
    std::vector<int> completedPopulationSteps;
    std::vector<bool> completedHasMacro;
    std::vector<std::vector<int>> completedForceSteps;
    std::vector<int> finalForceSteps;
    int finalStep = -1;
    bool initializedHasMacro = false;
    bool finalHasMacro = false;
    bool publishedFactsValid = true;
    std::optional<lbm::profiling::TimestepPerformanceRecord> performance;
};

template <class Descriptor>
amrex::Real population_sum(lbm::LatticeField<Descriptor> const& field)
{
    amrex::Real sum = 0.0;
    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
        sum += field.multifab().sum(iPop);
    }
    return sum;
}
inline amrex::Real component_max_error(
    amrex::MultiFab const& field,
    int component,
    amrex::Real expected)
{
    const amrex::Real minimum = field.min(component);
    const amrex::Real maximum = field.max(component);
    return std::max(
        std::abs(minimum - expected),
        std::abs(maximum - expected));
}

template <class Descriptor>
amrex::Real rest_population_max_error(
    lbm::LatticeField<Descriptor> const& field,
    amrex::Real rho)
{
    amrex::Real error = 0.0;
    for (int population = 0; population < Descriptor::q; ++population) {
        error = std::max(
            error,
            component_max_error(
                field.multifab(),
                population,
                Descriptor::w(population) * rho));
    }
    return error;
}

inline amrex::Real rest_macro_max_error(
    lbm::MacroField const& macro,
    amrex::Real rho)
{
    amrex::Real error =
        component_max_error(macro.multifab(), lbm::MacroField::Rho, rho);
    error = std::max(
        error,
        component_max_error(macro.multifab(), lbm::MacroField::Ux, 0.0));
    error = std::max(
        error,
        component_max_error(macro.multifab(), lbm::MacroField::Uy, 0.0));
    error = std::max(
        error,
        component_max_error(macro.multifab(), lbm::MacroField::Uz, 0.0));
    return error;
}

template <class Descriptor>
class RecordingRunSession {
public:
    RecordingRunSession() = default;
    explicit RecordingRunSession(std::shared_ptr<RecordingState> state)
        : state_(std::move(state))
    {}

    lbm::runtime::SingleLevelRunSessionResult start(
        lbm::runtime::SingleLevelRunContext<Descriptor> const&)
    {
        ++state_->starts;
        return {};
    }

    lbm::runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        return state_->plan;
    }

    lbm::runtime::SingleLevelRunSessionResult observeInitialized(
        lbm::runtime::SingleLevelPublishedState<Descriptor> const& published)
    {
        ++state_->initializations;
        state_->initializedHasMacro = published.hasMacro();
        state_->publishedFactsValid = state_->publishedFactsValid &&
            published.step() == 0 && published.latticeTime() == 0.0 &&
            published.physicalTime() == 0.0 &&
            published.forceBatch().empty() &&
            published.populations().multifab().boxArray() ==
                published.context().boxArray &&
            published.context().lbm.lengthRefLb > 0.0;
        if (state_->recordPopulationSums) {
            state_->initializedPopulationSum =
                population_sum(published.populations());
        }
        if (state_->recordRestState) {
            state_->initializedRestPopulationError =
                rest_population_max_error(published.populations(), 1.0);
            state_->publishedFactsValid =
                state_->publishedFactsValid && published.hasMacro();
            if (published.hasMacro()) {
                state_->initializedRestMacroError =
                    rest_macro_max_error(published.macro(), 1.0);
            }
        }
        return {};
    }

    lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
        lbm::runtime::SingleLevelPublishedState<Descriptor> const& published)
    {
        ++state_->completedSteps;
        state_->completedPopulationSteps.push_back(published.step());
        state_->completedHasMacro.push_back(published.hasMacro());
        std::vector<int> forceSteps;
        appendForceSteps(published, forceSteps);
        state_->completedForceSteps.push_back(std::move(forceSteps));
        state_->publishedFactsValid = state_->publishedFactsValid &&
            published.latticeTime() ==
                static_cast<amrex::Real>(published.step()) &&
            published.physicalTime() > 0.0;
        if (state_->recordPopulationSums) {
            state_->completedPopulationSums.push_back(
                population_sum(published.populations()));
        }
        if (state_->recordRestState) {
            state_->completedRestPopulationErrors.push_back(
                rest_population_max_error(published.populations(), 1.0));
            state_->publishedFactsValid =
                state_->publishedFactsValid && published.hasMacro();
            if (published.hasMacro()) {
                state_->completedRestMacroErrors.push_back(
                    rest_macro_max_error(published.macro(), 1.0));
            }
        }

        return {{}, published.step() == state_->stopAtStep};
    }

    lbm::runtime::SingleLevelRunSessionResult observeFinal(
        lbm::runtime::SingleLevelPublishedState<Descriptor> const& published)
    {
        ++state_->finalizations;
        state_->finalStep = published.step();
        state_->finalHasMacro = published.hasMacro();
        appendForceSteps(published, state_->finalForceSteps);
        state_->publishedFactsValid = state_->publishedFactsValid &&
            published.latticeTime() ==
                static_cast<amrex::Real>(published.step());
        if (state_->failFinal) {
            return {
                lbm::runtime::SingleLevelRunStatus::NumericalFailure,
                "injected numerical acceptance failure"};
        }
        return {};
    }

    lbm::runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<lbm::profiling::TimestepPerformanceRecord> const& record)
    {
        ++state_->performanceWrites;
        state_->performance = record;
        return {};
    }

private:
    void appendForceSteps(lbm::runtime::SingleLevelPublishedState<Descriptor> const& published,
                          std::vector<int>& steps)
    {
        auto const& batch = published.forceBatch();
        auto const& parameters = published.context().lbm;
        const auto dt = parameters.uRefLb * parameters.lengthRefPhys / parameters.lengthRefLb / parameters.uRefPhys;
        const auto validLabel = [dt](lbm::boundary::InternalForceStep const& step) {
            return step.level == 0 && step.coarsestStep == step.populationStep &&
                std::abs(step.beginPhysicalTime-(step.populationStep-1)*dt) < 1.0e-12 &&
                std::abs(step.endPhysicalTime-step.populationStep*dt) < 1.0e-12 &&
                std::abs(step.samplePhysicalTime()-(step.populationStep-0.5)*dt) < 1.0e-12;
        };
        state_->publishedFactsValid = state_->publishedFactsValid && batch.objectIntervals.empty() &&
            batch.boundarySteps.size() == batch.objectSteps.size();
        for (auto const& sample : batch.objectSteps) {
            steps.push_back(sample.step.populationStep);
            state_->publishedFactsValid = state_->publishedFactsValid && validLabel(sample.step) &&
                sample.objectForces.size() == 1 && sample.objectForces.front().objectId.value == 0;
        }
        for (auto const& sample : batch.boundarySteps)
            state_->publishedFactsValid = state_->publishedFactsValid && validLabel(sample.step) &&
                sample.boundaryForces.size() == 1 && sample.boundaryForces.front().boundaryId.value == 4;
    }

    std::shared_ptr<RecordingState> state_;
};

template <class Descriptor>
class PopulationIncrementDynamics
    : public lbm::LocalDynamicsModel<Descriptor> {
public:
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    explicit PopulationIncrementDynamics(amrex::Real increment) noexcept
        : increment_(increment)
    {}

    template <class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void collide(Cell&& cell) const noexcept
    {
        cell[0] += increment_;
    }

private:
    amrex::Real increment_;
};

struct PopulationIncrementSelection {
    amrex::Real increment = 0.0;

    template <class Descriptor>
    using Dynamics = PopulationIncrementDynamics<Descriptor>;

    template <class Descriptor>
    lbm::DynamicsPreparationResult<Dynamics<Descriptor>> prepare(
        lbm::config::ResolvedLbmConfig const&) const
    {
        lbm::DynamicsPreparationResult<Dynamics<Descriptor>> result;
        result.dynamics.emplace(increment);
        return result;
    }
};

struct LinearDensityProfile {
    amrex::Real slope = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    lbm::initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real) const noexcept
    {
        return {1.0 + slope * position[0], 0.0, 0.0, 0.0};
    }
};

struct RecordingPreparation {
    std::shared_ptr<RecordingState> state;

    template <class Descriptor>
    using RunSession = RecordingRunSession<Descriptor>;

    template <class View>
    lbm::runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view) const
    {
        if (state->expectDomainWalls) {
            bool valid = view.boundaries.domain.size() == 4;
            for (auto const& wall : view.boundaries.domain) {
                valid = valid &&
                    wall.physical ==
                        lbm::boundary::BoundaryPhysical::NoSlipWall &&
                    wall.scheme ==
                        lbm::boundary::BoundaryScheme::HalfwayBounceBack &&
                    wall.schemeOrigin ==
                        lbm::boundary::BoundarySchemeOrigin::Default &&
                    wall.stage ==
                        lbm::boundary::BoundaryExecutionStage::
                            PreStreamingPhysicalGhost &&
                    wall.profile == lbm::boundary::BoundaryProfileKind::None;
            }
            state->resolvedDomainWallsValid = valid;
        }
        return {RunSession<typename View::DescriptorType>(state), {}};
    }
};

using RecordingProgram = lbm::runtime::SingleLevelProgram<
    ActiveDescriptor,
    lbm::BGKDynamicsSelection,
    lbm::initialization::ConfiguredInitializationSelection,
    RecordingPreparation>;

RecordingProgram
make_program(std::shared_ptr<RecordingState> state, int maxStep)
{
    state->plan.observeInitialized = true;
    state->plan.observeFinal = true;
    state->plan.completed.everyStep = true;
    RecordingProgram program;
    program.mesh.nCell = {8, 8, ActiveDescriptor::d == 3 ? 2 : 1};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        ActiveDescriptor::d == 3 ? 0.25 : 0.125};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = 4;
    program.lbm.parameterizationKind =
        lbm::config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.01;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = 0.01;
    program.run.maxStep = maxStep;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "focused_check";
    program.run.name = "runtime_interface";
    program.initialization.intent.kind =
        lbm::initialization::InitializationKind::TaylorGreen;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb = 0.01;
    program.casePreparation.state = std::move(state);
    return program;
}

void check_success()
{
    auto state = std::make_shared<RecordingState>();
    RecordingProgram program = make_program(state, 2);
    program.profiling = {1, 0, 2};
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    AMREX_ALWAYS_ASSERT(prepared.ok());
    lbm::runtime::SingleLevelRunReport const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 2);
    AMREX_ALWAYS_ASSERT(
        state->starts == 1 &&
        state->initializations == 1 &&
        state->completedSteps == 2 &&
        state->finalizations == 1 &&
        state->completedPopulationSteps == std::vector<int>({1, 2}) &&
        state->finalStep == 2 && state->publishedFactsValid &&
        !state->initializedHasMacro && !state->finalHasMacro &&
        state->completedHasMacro == std::vector<bool>({false, false}) &&
        state->performance.has_value() &&
        std::isfinite(state->performance->rankMaxWindow.postStream) &&
        std::isfinite(state->performance->rankMaxWindow.domainPostStream) &&
        std::isfinite(
            state->performance->rankMaxWindow.internalSolidReconstruction) &&
        state->performance->rankMaxWindow.domainPostStream >= 0.0 &&
        state->performance->rankMaxWindow.internalSolidReconstruction >= 0.0 &&
        state->performance->rankMaxWindow.postStream >=
            state->performance->rankMaxWindow.domainPostStream &&
        state->performance->rankMaxWindow.postStream >=
            state->performance->rankMaxWindow.internalSolidReconstruction);
}

void check_throughput_profiling_path()
{
    auto state = std::make_shared<RecordingState>();
    RecordingProgram program = make_program(state, 2);
    program.profiling = {
        1,
        0,
        2,
        lbm::profiling::TimestepProfilingMode::Throughput};
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    AMREX_ALWAYS_ASSERT(prepared.ok());
    lbm::runtime::SingleLevelRunReport const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && state->performance.has_value());
    auto const& performance = *state->performance;
    AMREX_ALWAYS_ASSERT(
        performance.mode ==
            lbm::profiling::TimestepProfilingMode::Throughput &&
        performance.formalMlups && performance.criticalRank == -1 &&
        performance.rankMaxWindow.collision == 0.0 &&
        performance.rankMaxWindow.preStream == 0.0 &&
        performance.rankMaxWindow.preStreamFillBoundary == 0.0 &&
        performance.rankMaxWindow.streaming == 0.0 &&
        performance.rankMaxWindow.domainPostStream == 0.0 &&
        performance.rankMaxWindow.internalSolidReconstruction == 0.0 &&
        performance.rankMaxWindow.postStream == 0.0 &&
        performance.rankMaxWindow.diagnostics == 0.0 &&
        performance.rankMaxWindow.total == 0.0 &&
        performance.synchronizedWindow >= 0.0);
}

void check_invalid_profiling_mode()
{
    auto state = std::make_shared<RecordingState>();
    RecordingProgram program = make_program(state, 1);
    program.profiling.enabled = 1;
    program.profiling.mode =
        static_cast<lbm::profiling::TimestepProfilingMode>(99);
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    AMREX_ALWAYS_ASSERT(!prepared.ok());
    bool found = false;
    for (auto const& diagnostic : prepared.diagnostics()) {
        found = found ||
            (diagnostic.code == "invalid_profiling_mode" &&
             diagnostic.subject == "profiling.mode");
    }
    AMREX_ALWAYS_ASSERT(found);
}

void check_numerical_failure_report()
{
    auto state = std::make_shared<RecordingState>();
    state->failFinal = true;
    auto prepared = lbm::runtime::prepare_single_level(
        make_program(state, 1));
    AMREX_ALWAYS_ASSERT(prepared.ok());
    lbm::runtime::SingleLevelRunReport const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(
        report.status ==
            lbm::runtime::SingleLevelRunStatus::NumericalFailure &&
        report.finalStep == 1 &&
        report.message == "injected numerical acceptance failure");
}

void check_user_composed_dynamics_and_initialization()
{
    constexpr amrex::Real densitySlope = 0.01;
    constexpr amrex::Real populationIncrement = 0.0005;
    using InitializationSelection =
        lbm::initialization::ProfileInitializationSelection<
            LinearDensityProfile>;
    using Program = lbm::runtime::SingleLevelProgram<
        ActiveDescriptor,
        PopulationIncrementSelection,
        InitializationSelection,
        RecordingPreparation>;

    auto state = std::make_shared<RecordingState>();
    state->recordPopulationSums = true;
    state->plan.observeInitialized = true;
    state->plan.observeFinal = true;
    state->plan.completed.everyStep = true;
    Program program;
    program.mesh.nCell = {8, 8, ActiveDescriptor::d == 3 ? 2 : 1};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        ActiveDescriptor::d == 3 ? 0.25 : 0.125};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = 4;
    program.lbm.parameterizationKind =
        lbm::config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.01;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = 0.01;
    program.dynamics.increment = populationIncrement;
    program.run.maxStep = 2;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "focused_check";
    program.run.name = "user_composed_numerics";
    program.initialization =
        lbm::initialization::from_profile(
            LinearDensityProfile{densitySlope});
    program.casePreparation.state = state;

    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    if (!prepared.ok()) {
        print_preparation_diagnostics(prepared.diagnostics());
    }
    AMREX_ALWAYS_ASSERT(prepared.ok());
    lbm::runtime::SingleLevelRunReport const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 2);

    const amrex::Real cellCount = static_cast<amrex::Real>(
        8 * 8 * (ActiveDescriptor::d == 3 ? 2 : 1));
    const amrex::Real initializedExpected =
        cellCount * (1.0 + 0.5 * densitySlope);
    const amrex::Real tolerance = 1.0e-11 * cellCount;
    AMREX_ALWAYS_ASSERT(
        std::abs(
            state->initializedPopulationSum - initializedExpected) <
        tolerance);
    AMREX_ALWAYS_ASSERT(state->completedPopulationSums.size() == 2);
    for (std::size_t step = 0;
         step < state->completedPopulationSums.size();
         ++step) {
        const amrex::Real expected = initializedExpected +
            static_cast<amrex::Real>(step + 1) *
                cellCount * populationIncrement;
        AMREX_ALWAYS_ASSERT(
            std::abs(state->completedPopulationSums[step] - expected) <
            tolerance);
    }
}

void check_observation_plan_and_macro_views()
{
    lbm::runtime::SingleLevelObservationCadence cadence;
    cadence.addInterval(2);
    cadence.includeMaxStep = true;
    AMREX_ALWAYS_ASSERT(
        !cadence.due(1, 3) && cadence.due(2, 3) && cadence.due(3, 3));

    auto state = std::make_shared<RecordingState>();
    RecordingProgram program = make_program(state, 3);
    state->plan.completed.everyStep = false;
    state->plan.completed.includeMaxStep = true;
    state->plan.macroAtInitialized = true;
    state->plan.macroAtFinal = true;
    state->plan.completedMacro.addInterval(2);

    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    AMREX_ALWAYS_ASSERT(prepared.ok());
    auto const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 3);
    AMREX_ALWAYS_ASSERT(
        state->initializedHasMacro && state->finalHasMacro &&
        state->completedPopulationSteps == std::vector<int>({2, 3}) &&
        state->completedHasMacro == std::vector<bool>({true, false}));
}

void check_early_stop_final_observation()
{
    auto state = std::make_shared<RecordingState>();
    state->stopAtStep = 2;
    RecordingProgram program = make_program(state, 4);
    program.profiling = {1, 0, 0};
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    AMREX_ALWAYS_ASSERT(prepared.ok());
    auto const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(
        report.ok() && report.stoppedOnSteady && report.finalStep == 2 &&
        state->completedPopulationSteps == std::vector<int>({1, 2}) &&
        state->finalizations == 1 && state->finalStep == 2 &&
        state->performanceWrites == 1 && state->performance.has_value() &&
        state->performance->sampleStart == 1 &&
        state->performance->sampleEnd == 2 &&
        state->performance->nSteps == 2 &&
        state->performance->synchronizedWindow >= 0.0);
}

void check_domain_wall_runtime()
{
    auto state = std::make_shared<RecordingState>();
    state->recordPopulationSums = true;
    state->recordRestState = true;
    state->expectDomainWalls = true;
    RecordingProgram program = make_program(state, 10);
    program.mesh.nCell = {16, 16, ActiveDescriptor::d == 3 ? 4 : 1};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        ActiveDescriptor::d == 3 ? 0.25 : 0.0625};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = 8;
    program.initialization.intent.kind =
        lbm::initialization::InitializationKind::Rest;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb.reset();

    for (auto const side :
         {lbm::boundary::DomainSide::XLo,
          lbm::boundary::DomainSide::XHi,
          lbm::boundary::DomainSide::YLo,
          lbm::boundary::DomainSide::YHi}) {
        lbm::boundary::DomainBoundarySpec wall;
        wall.name = "runtime_wall_" +
            std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
        program.domainBoundaries.push_back(std::move(wall));
    }

    state->plan.completed = {};
    state->plan.completed.addInterval(10);
    state->plan.macroAtInitialized = true;
    state->plan.macroAtFinal = true;
    state->plan.completedMacro = {};
    state->plan.completedMacro.addInterval(10);
    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    if (!prepared.ok()) {
        print_preparation_diagnostics(prepared.diagnostics());
    }
    AMREX_ALWAYS_ASSERT(prepared.ok());
    auto const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 10);
    AMREX_ALWAYS_ASSERT(
        state->starts == 1 && state->initializations == 1 &&
        state->completedSteps == 1 && state->finalizations == 1 &&
        state->initializedHasMacro && state->finalHasMacro &&
        state->completedPopulationSteps == std::vector<int>({10}) &&
        state->completedForceSteps ==
            std::vector<std::vector<int>>({{}}) &&
        state->finalForceSteps.empty() && state->publishedFactsValid &&
        state->resolvedDomainWallsValid);
    AMREX_ALWAYS_ASSERT(state->completedPopulationSums.size() == 1);
    const amrex::Real cellCount = static_cast<amrex::Real>(
        16 * 16 * (ActiveDescriptor::d == 3 ? 4 : 1));
    AMREX_ALWAYS_ASSERT(
        std::abs(
            state->completedPopulationSums.front() -
            state->initializedPopulationSum) <
        1.0e-12 * cellCount);
    const amrex::Real restTolerance = 1.0e-13;
    AMREX_ALWAYS_ASSERT(
        state->initializedRestPopulationError < restTolerance &&
        state->initializedRestMacroError < restTolerance &&
        state->completedRestPopulationErrors.size() == 1 &&
        state->completedRestMacroErrors.size() == 1 &&
        state->completedRestPopulationErrors.front() < restTolerance &&
        state->completedRestMacroErrors.front() < restTolerance);
}

void check_internal_force_delivery_and_final_drain()
{
    auto state = std::make_shared<RecordingState>();
    RecordingProgram program = make_program(state, 3);
    program.mesh.nCell = {32, 32, ActiveDescriptor::d == 3 ? 4 : 1};
    program.mesh.envelopeHi = {
        1.0,
        1.0,
        ActiveDescriptor::d == 3 ? 0.125 : 0.03125};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = 16;
    program.lbm.lengthRefLb = 31.0;
    program.lbm.lengthRefPhys = 31.0 / 32.0;
    program.initialization.intent.kind =
        lbm::initialization::InitializationKind::Rest;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb.reset();

    for (auto const side :
         {lbm::boundary::DomainSide::XLo,
          lbm::boundary::DomainSide::XHi,
          lbm::boundary::DomainSide::YLo,
          lbm::boundary::DomainSide::YHi}) {
        lbm::boundary::DomainBoundarySpec wall;
        wall.name = "runtime_force_wall_" +
            std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
        wall.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
        program.domainBoundaries.push_back(std::move(wall));
    }

    lbm::boundary::InternalBoundarySpec cylinder;
    cylinder.name = "runtime_force_cylinder";
    cylinder.geometry = "cylinder";
    cylinder.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
    cylinder.hasPhysical = true;
    cylinder.scheme = lbm::boundary::BoundaryScheme::Yms;
    cylinder.profile = lbm::boundary::BoundaryProfileKind::None;
    cylinder.objectName = "runtime_force_object";
    cylinder.centerPhys = {
        0.5,
        0.5,
        ActiveDescriptor::d == 3 ? 0.0625 : 0.015625};
    cylinder.hasCenterPhys = true;
    cylinder.radiusPhys = 0.15;
    cylinder.hasRadiusPhys = true;
    cylinder.axis = "z";
    cylinder.hasAxis = true;
    program.internalBoundaries.push_back(std::move(cylinder));

    state->plan.completed.everyStep = false;
    state->plan.completed.includeMaxStep = false;
    state->plan.forceCalculationInterval = 1;
    state->plan.forceDeliveryInterval = 2;
    program.profiling = {1, 0, 3};

    auto prepared = lbm::runtime::prepare_single_level(std::move(program));
    if (!prepared.ok()) {
        print_preparation_diagnostics(prepared.diagnostics());
    }
    AMREX_ALWAYS_ASSERT(prepared.ok());
    auto const report =
        lbm::runtime::run_single_level(std::move(prepared).take());
    AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 3);
    AMREX_ALWAYS_ASSERT(
        state->starts == 1 && state->initializations == 1 &&
        state->completedSteps == 1 && state->finalizations == 1 &&
        state->completedPopulationSteps == std::vector<int>({2}) &&
        state->completedForceSteps ==
            std::vector<std::vector<int>>({{1, 2}}) &&
        state->finalStep == 3 &&
        state->finalForceSteps == std::vector<int>({3}) &&
        state->publishedFactsValid && state->performance.has_value() &&
        std::isfinite(
            state->performance->rankMaxWindow.domainPostStream) &&
        std::isfinite(
            state->performance->rankMaxWindow.internalSolidReconstruction) &&
        state->performance->rankMaxWindow.domainPostStream >= 0.0 &&
        state->performance->rankMaxWindow.internalSolidReconstruction >= 0.0 &&
        state->performance->rankMaxWindow.postStream >=
            state->performance->rankMaxWindow.domainPostStream &&
        state->performance->rankMaxWindow.postStream >=
            state->performance->rankMaxWindow.internalSolidReconstruction);
}

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        check_success();
        check_throughput_profiling_path();
        check_invalid_profiling_mode();
        check_numerical_failure_report();
        check_user_composed_dynamics_and_initialization();
        check_observation_plan_and_macro_views();
        check_early_stop_final_observation();
        check_domain_wall_runtime();
        check_internal_force_delivery_and_final_drain();
        amrex::Print() << "Single-level runtime integration tests: PASS\n";
    }
    amrex::Finalize();
    return 0;
}
