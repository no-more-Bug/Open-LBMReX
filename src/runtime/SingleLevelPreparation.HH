#ifndef LBM_RUNTIME_SINGLELEVELPREPARATION_HH
#define LBM_RUNTIME_SINGLELEVELPREPARATION_HH

#include "lbm/boundary/core/BoundarySystem.H"
#include "lbm/boundary/domain/OnLatticeReferenceGeometryResolver.H"
#include "lbm/config/LbmConfigResolver.H"
#include "lbm/fields/LatticeField.H"
#include "runtime/SingleLevelRunControl.H"
#include "runtime/SingleLevelRunSession.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <string>
#include <utility>

namespace lbm::runtime {

namespace preparation_detail {

inline void append_errors(
    std::vector<SingleLevelSetupDiagnostic>& diagnostics,
    SingleLevelSetupStage stage,
    char const* code,
    char const* subject,
    std::vector<std::string> const& errors)
{
    for (std::string const& error : errors) {
        diagnostics.push_back({stage, code, subject, error});
    }
}

template <class Descriptor>
std::vector<SingleLevelSetupDiagnostic> validate_mesh(
    SingleLevelMeshIntent const& mesh)
{
    std::vector<SingleLevelSetupDiagnostic> diagnostics;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (mesh.nCell[d] <= 0) {
            diagnostics.push_back({
                SingleLevelSetupStage::Mesh,
                "non_positive_cell_count",
                "mesh.n_cell",
                "cell counts must be positive in every direction"});
        }
        if (!std::isfinite(mesh.envelopeLo[d]) ||
            !std::isfinite(mesh.envelopeHi[d]) ||
            !std::isfinite(mesh.envelopeHi[d] - mesh.envelopeLo[d]) ||
            !(mesh.envelopeHi[d] > mesh.envelopeLo[d])) {
            diagnostics.push_back({
                SingleLevelSetupStage::Mesh,
                "invalid_envelope",
                "mesh.envelope",
                "AMReX envelope bounds must be finite and increasing"});
        }
        if (mesh.isPeriodic[d] != 0 && mesh.isPeriodic[d] != 1) {
            diagnostics.push_back({
                SingleLevelSetupStage::Mesh,
                "invalid_periodicity",
                "mesh.periodicity",
                "periodicity flags must be 0 or 1"});
        }
    }
    if (mesh.maxGridSize <= 0) {
        diagnostics.push_back({
            SingleLevelSetupStage::Mesh,
            "non_positive_max_grid_size",
            "mesh.max_grid_size",
            "max grid size must be positive"});
    }
    if constexpr (Descriptor::d == 2 && AMREX_SPACEDIM == 3) {
        if (diagnostics.empty()) {
            const amrex::Real dx =
                (mesh.envelopeHi[0] - mesh.envelopeLo[0]) / mesh.nCell[0];
            const amrex::Real dz =
                (mesh.envelopeHi[2] - mesh.envelopeLo[2]) / mesh.nCell[2];
            if (!config::detail::nearly_equal(dx, dz) &&
                (mesh.nCell[2] != 1 || mesh.isPeriodic[2] != 1)) {
                diagnostics.push_back({
                    SingleLevelSetupStage::Mesh,
                    "invalid_d2q9_thickness_layout",
                    "mesh.z",
                    "non-cubic D2Q9 meshes require one periodic z cell"});
            }
        }
    }
    return diagnostics;
}

inline amrex::Geometry make_geometry(SingleLevelMeshIntent const& mesh)
{
    const amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    const amrex::IntVect domHi(AMREX_D_DECL(
        mesh.nCell[0] - 1,
        mesh.nCell[1] - 1,
        mesh.nCell[2] - 1));
    const amrex::Box domain(domLo, domHi);
    const amrex::RealBox realBox(
        mesh.envelopeLo.data(), mesh.envelopeHi.data());
    return amrex::Geometry(
        domain,
        &realBox,
        amrex::CoordSys::cartesian,
        mesh.isPeriodic.data());
}

inline std::vector<SingleLevelSetupDiagnostic> validate_profiling(
    SingleLevelProfilingIntent const& profiling)
{
    std::vector<SingleLevelSetupDiagnostic> diagnostics;
    if (profiling.enabled != 0 && profiling.enabled != 1) {
        diagnostics.push_back({
            SingleLevelSetupStage::RunControl,
            "invalid_profiling_flag",
            "profiling.enabled",
            "profiling enabled flag must be 0 or 1"});
    }
    if (profiling.warmupSteps < 0 || profiling.measurementSteps < 0) {
        diagnostics.push_back({
            SingleLevelSetupStage::RunControl,
            "invalid_profiling_window",
            "profiling.window",
            "profiling warmup and measurement steps must be non-negative"});
    }
    if (!profiling::is_valid(profiling.mode)) {
        diagnostics.push_back({
            SingleLevelSetupStage::RunControl,
            "invalid_profiling_mode",
            "profiling.mode",
            "profiling mode is not a recognized value"});
    } else if (profiling.enabled == 1 &&
        profiling.mode == profiling::TimestepProfilingMode::Disabled) {
        diagnostics.push_back({
            SingleLevelSetupStage::RunControl,
            "invalid_profiling_mode",
            "profiling.mode",
            "enabled profiling requires throughput or stage breakdown mode"});
    }
    return diagnostics;
}

} // namespace preparation_detail

template <class Descriptor, class Initialization>
struct SingleLevelPreparationView {
    using DescriptorType = Descriptor;

    SingleLevelMeshIntent const& mesh;
    SingleLevelProfilingIntent const& profiling;
    config::ResolvedLbmConfig const& lbm;
    ResolvedSingleLevelRunControl const& run;
    Initialization const& initialization;
    amrex::Geometry const& geometry;
    boundary::ResolvedBoundarySet const& boundaries;
    boundary::PostStreamingBoundaryPlan const& postStreamingPlan;
};

template <class PreparationView>
SingleLevelCasePreparationResult<
    NoCasePreparation::RunSession<
        typename PreparationView::DescriptorType>>
NoCasePreparation::resolve(PreparationView const&) const
{
    return {};
}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
struct PreparedSingleLevelRun<
    Descriptor,
    Dynamics,
    Initialization,
    RunSession>::State {
    State(
        ResolvedSingleLevelRunControl runIn,
        config::ResolvedLbmConfig lbmIn,
        Dynamics dynamicsIn,
        Initialization initializationIn,
        amrex::Geometry geometryIn,
        amrex::BoxArray boxArrayIn,
        amrex::DistributionMapping distributionMappingIn,
        boundary::BoundarySystem<Descriptor> boundarySystemIn,
        RunSession runSessionIn,
        SingleLevelProfilingIntent const& profilingIn,
        int onLatticeReferenceAxisIn)
        : run(std::move(runIn)),
          lbm(std::move(lbmIn)),
          dynamics(std::move(dynamicsIn)),
          initialization(std::move(initializationIn)),
          geometry(std::move(geometryIn)),
          boxArray(std::move(boxArrayIn)),
          distributionMapping(std::move(distributionMappingIn)),
          boundarySystem(std::move(boundarySystemIn)),
          runSession(std::move(runSessionIn)),
          profiler(
              profilingIn.enabled != 0
                  ? profilingIn.mode
                  : profiling::TimestepProfilingMode::Disabled,
              profilingIn.warmupSteps,
              profilingIn.measurementSteps),
          src(std::make_unique<LatticeField<Descriptor>>(
              "populations_src", boxArray, distributionMapping)),
          dst(std::make_unique<LatticeField<Descriptor>>(
              "populations_dst", boxArray, distributionMapping)),
          onLatticeReferenceAxis(onLatticeReferenceAxisIn)
    {}

    ResolvedSingleLevelRunControl run;
    config::ResolvedLbmConfig lbm;
    Dynamics dynamics;
    Initialization initialization;
    amrex::Geometry geometry;
    amrex::BoxArray boxArray;
    amrex::DistributionMapping distributionMapping;
    boundary::BoundarySystem<Descriptor> boundarySystem;
    RunSession runSession;
    profiling::TimestepProfiler profiler;
    std::unique_ptr<LatticeField<Descriptor>> src;
    std::unique_ptr<LatticeField<Descriptor>> dst;
    int onLatticeReferenceAxis = -1;
};

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>::
PreparedSingleLevelRun(
    std::unique_ptr<State> state) noexcept
    : state_(std::move(state))
{}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>::
PreparedSingleLevelRun(
    PreparedSingleLevelRun&&) noexcept = default;

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>&
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>::operator=(
    PreparedSingleLevelRun&&) noexcept = default;

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>::
~PreparedSingleLevelRun() = default;

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
SingleLevelPreparationResult<Descriptor, Dynamics, Initialization, RunSession>::
SingleLevelPreparationResult(
    std::vector<SingleLevelSetupDiagnostic> diagnostics)
    : diagnostics_(std::move(diagnostics))
{}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
SingleLevelPreparationResult<Descriptor, Dynamics, Initialization, RunSession>::
SingleLevelPreparationResult(
    PreparedSingleLevelRun<
        Descriptor,
        Dynamics,
        Initialization,
        RunSession> prepared)
    : prepared_(std::move(prepared))
{}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
bool SingleLevelPreparationResult<Descriptor, Dynamics, Initialization, RunSession>::
ok() const noexcept
{
    return prepared_.has_value();
}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
std::vector<SingleLevelSetupDiagnostic> const&
SingleLevelPreparationResult<Descriptor, Dynamics, Initialization, RunSession>::
diagnostics() const noexcept
{
    return diagnostics_;
}

template <class Descriptor, class Dynamics, class Initialization, class RunSession>
PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession>
SingleLevelPreparationResult<Descriptor, Dynamics, Initialization, RunSession>::
take() &&
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        prepared_.has_value(),
        "Successful single-level preparation is required before take()");
    PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession> result =
        std::move(*prepared_);
    prepared_.reset();
    return result;
}

template <
    class Descriptor,
    class DynamicsSelection,
    class InitializationSelection,
    class CasePreparation>
SingleLevelPreparationResult<
    Descriptor,
    typename DynamicsSelection::template Dynamics<Descriptor>,
    initialization::PreparedInitialization<
        Descriptor,
        InitializationSelection>,
    typename CasePreparation::template RunSession<Descriptor>>
prepare_single_level(
    SingleLevelProgram<
        Descriptor,
        DynamicsSelection,
        InitializationSelection,
        CasePreparation> program)
{
    using Dynamics =
        typename DynamicsSelection::template Dynamics<Descriptor>;
    using Initialization =
        initialization::PreparedInitialization<
            Descriptor,
            InitializationSelection>;
    using RunSession =
        typename CasePreparation::template RunSession<Descriptor>;
    using Result = SingleLevelPreparationResult<
        Descriptor,
        Dynamics,
        Initialization,
        RunSession>;
    std::vector<SingleLevelSetupDiagnostic> diagnostics =
        preparation_detail::validate_mesh<Descriptor>(program.mesh);
    if (!diagnostics.empty()) {
        return Result(std::move(diagnostics));
    }

    config::LbmConfigResolveResult lbmResult =
        config::resolve_lbm_config<Descriptor>(
            program.lbm,
            program.mesh.nCell,
            program.mesh.envelopeLo,
            program.mesh.envelopeHi);
    if (!lbmResult.ok()) {
        preparation_detail::append_errors(
            diagnostics,
            SingleLevelSetupStage::LbmClosure,
            "invalid_lbm_intent",
            "lbm",
            lbmResult.errors);
        return Result(std::move(diagnostics));
    }
    auto dynamicsResult =
        program.dynamics.template prepare<Descriptor>(lbmResult.resolved);
    if (!dynamicsResult.ok()) {
        preparation_detail::append_errors(
            diagnostics,
            SingleLevelSetupStage::LbmClosure,
            "invalid_dynamics_selection",
            "dynamics",
            dynamicsResult.errors);
        return Result(std::move(diagnostics));
    }
    if (lbmResult.resolved.streamingBackend != "baseline") {
        diagnostics.push_back({
            SingleLevelSetupStage::LbmClosure,
            "unsupported_streaming_backend",
            "lbm.streaming_backend",
            "single-level runtime currently supports only baseline streaming"});
        return Result(std::move(diagnostics));
    }

    SingleLevelRunControlResolveResult runResult =
        resolve_single_level_run_control(program.run, lbmResult.resolved);
    if (!runResult.ok()) {
        preparation_detail::append_errors(
            diagnostics,
            SingleLevelSetupStage::RunControl,
            "invalid_run_control",
            "run",
            runResult.errors);
        return Result(std::move(diagnostics));
    }
    std::vector<SingleLevelSetupDiagnostic> profilingDiagnostics =
        preparation_detail::validate_profiling(program.profiling);
    diagnostics.insert(
        diagnostics.end(),
        std::make_move_iterator(profilingDiagnostics.begin()),
        std::make_move_iterator(profilingDiagnostics.end()));
    if (!diagnostics.empty()) {
        return Result(std::move(diagnostics));
    }

    amrex::Geometry geometry = preparation_detail::make_geometry(program.mesh);
    boundary::BoundarySystem<Descriptor> boundarySystem;
    std::string boundaryError;
    if (!boundarySystem.tryInitialize(
            program.domainBoundaries,
            program.internalBoundaries,
            geometry,
            boundaryError,
            -1)) {
        diagnostics.push_back({
            SingleLevelSetupStage::PhysicalBoundary,
            "invalid_physical_boundary",
            "physical_boundary",
            std::move(boundaryError)});
        return Result(std::move(diagnostics));
    }

    auto initializationResult =
        initialization::prepare_initialization<Descriptor>(
            program.initialization,
            lbmResult.resolved,
            program.mesh.nCell);
    if (!initializationResult.ok()) {
        preparation_detail::append_errors(
            diagnostics,
            SingleLevelSetupStage::Initialization,
            "invalid_initialization",
            "initialization",
            initializationResult.errors);
        return Result(std::move(diagnostics));
    }

    boundary::OnLatticeReferenceGeometryResolveResult referenceGeometry =
        boundary::resolve_lbm_reference_geometry(
            boundarySystem.postStreamingPlan(), lbmResult.resolved);
    if (!referenceGeometry.ok()) {
        preparation_detail::append_errors(
            diagnostics,
            SingleLevelSetupStage::PhysicalBoundary,
            "inconsistent_lbm_physical_reference",
            "lbm.physical_reference",
            referenceGeometry.errors);
        return Result(std::move(diagnostics));
    }

    SingleLevelPreparationView<Descriptor, Initialization> view{
        program.mesh,
        program.profiling,
        lbmResult.resolved,
        runResult.resolved,
        *initializationResult.initialization,
        geometry,
        boundarySystem.resolved(),
        boundarySystem.postStreamingPlan()};
    SingleLevelCasePreparationResult<RunSession> caseResult =
        program.casePreparation.resolve(view);
    for (SingleLevelCaseDiagnostic const& diagnostic : caseResult.diagnostics) {
        diagnostics.push_back({
            SingleLevelSetupStage::CaseValidation,
            diagnostic.code,
            diagnostic.subject,
            diagnostic.message});
    }
    if (!diagnostics.empty()) {
        return Result(std::move(diagnostics));
    }

    amrex::BoxArray boxArray(geometry.Domain());
    boxArray.maxSize(program.mesh.maxGridSize);
    amrex::DistributionMapping distributionMapping(boxArray);
    std::string layoutError;
    if (!boundarySystem.tryBindLayout(
            boxArray, distributionMapping, layoutError, -1)) {
        diagnostics.push_back({
            SingleLevelSetupStage::RuntimePreparation,
            "boundary_layout_binding_failed",
            "runtime.boundary_layout",
            std::move(layoutError)});
        return Result(std::move(diagnostics));
    }

    auto state = std::make_unique<
        typename PreparedSingleLevelRun<
            Descriptor,
            Dynamics,
            Initialization,
            RunSession>::State>(
        std::move(runResult.resolved),
        std::move(lbmResult.resolved),
        std::move(*dynamicsResult.dynamics),
        std::move(*initializationResult.initialization),
        std::move(geometry),
        std::move(boxArray),
        std::move(distributionMapping),
        std::move(boundarySystem),
        std::move(caseResult.session),
        program.profiling,
        referenceGeometry.axis);
    return Result(
        PreparedSingleLevelRun<
            Descriptor,
            Dynamics,
            Initialization,
            RunSession>(std::move(state)));
}

} // namespace lbm::runtime

#endif
