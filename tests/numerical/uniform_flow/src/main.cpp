#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace numerical_uniform {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

enum class Variant { HbbVelocityIo, GuoVelocityIo, GuoFreeOutflow };

struct Controls {
  Variant variant = Variant::HbbVelocityIo;
  amrex::Array<int, 2> intervals{10, 10};
  int depthCells = 1;
  int maxGridSize = 10;
  amrex::Real rhoLb = 1.0;
  amrex::Real velocityLb = 0.1;
  amrex::Real velocityReferencePhys = 1.0;
  amrex::Real nuLb = 1.0 / 12.0;
  int maxStep = 10000;
  std::string runVariant;
  std::string outputDirectory;
  std::string logDirectory;
};

Controls read_controls() {
  Controls controls;
  amrex::ParmParse uniform("uniform_flow");
  std::string variant;
  uniform.get("variant", variant);
  if (variant == "hbb_velocity_io") {
    controls.variant = Variant::HbbVelocityIo;
  } else if (variant == "guo_velocity_io") {
    controls.variant = Variant::GuoVelocityIo;
  } else if (variant == "guo_velocity_free_outflow") {
    controls.variant = Variant::GuoFreeOutflow;
  } else {
    amrex::Abort("unknown numerical uniform-flow variant");
  }
  amrex::Vector<int> intervals;
  uniform.getarr("intervals", intervals, 0, 2);
  controls.intervals = {intervals[0], intervals[1]};
  uniform.get("depth_cells", controls.depthCells);
  uniform.get("rho_lb", controls.rhoLb);
  uniform.get("velocity_lb", controls.velocityLb);
  uniform.get("velocity_reference_phys", controls.velocityReferencePhys);
  uniform.get("nu_lb", controls.nuLb);
  amrex::ParmParse mesh("mesh");
  mesh.get("max_grid_size", controls.maxGridSize);
  amrex::ParmParse run("run");
  run.get("max_step", controls.maxStep);
  run.get("variant", controls.runVariant);
  amrex::ParmParse output("output");
  output.get("directory", controls.outputDirectory);
  output.get("log_directory", controls.logDirectory);
  return controls;
}

bool uses_guo(Variant variant) noexcept {
  return variant != Variant::HbbVelocityIo;
}

struct ExactReference {
  lbm::initialization::MacroscopicState state;
  amrex::Real massReferenceDensity = 1.0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE lbm::initialization::MacroscopicState
  operator()(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &, int, int,
             int) const noexcept {
    return state;
  }
};

template <class Descriptor> class Output {
public:
  Output(Controls controls, lbm::config::ResolvedLbmConfig const &lbm)
      : controls_(std::move(controls)), lbm_(lbm) {}

  std::string prepare() {
    std::string error;
    if (amrex::ParallelDescriptor::IOProcessor()) {
      try {
        std::filesystem::create_directories(controls_.outputDirectory);
        std::filesystem::create_directories(controls_.logDirectory);
        std::filesystem::remove(metrics_path());
      } catch (std::filesystem::filesystem_error const &exception) {
        error = exception.what();
      }
    }
    amrex::BroadcastString(error, amrex::ParallelDescriptor::MyProc(),
                           amrex::ParallelDescriptor::IOProcessorNumber(),
                           amrex::ParallelDescriptor::Communicator());
    return error;
  }

  void print_identity() const {
    char const *condition = controls_.variant == Variant::GuoFreeOutflow
                                ? "velocity_inlet -> free_outflow"
                                : "velocity_inlet -> velocity_outlet";
    char const *scheme = uses_guo(controls_.variant) ? "guo_extrapolation"
                                                     : "halfway_bounce_back";
    amrex::Print() << "descriptor = " << Descriptor::name() << '\n'
                   << "case_name = uniform_flow\n"
                   << "case_variant = " << controls_.runVariant << '\n'
                   << "condition = " << condition << '\n'
                   << "scheme = " << scheme << '\n';
  }

  void printAndLog(std::string const &text) { amrex::Print() << text; }
  void writeDiagnosticObservation(lbm::runtime::DiagnosticRow const &) {}
  void writeDiagnosticsRow(lbm::runtime::DiagnosticRow const &, bool) const {}
  void writeConvergenceRow(lbm::runtime::DiagnosticRow const &) const {}
  void writePlotfile(lbm::MacroField const &, amrex::Geometry const &,
                     amrex::BoxArray const &,
                     amrex::DistributionMapping const &, int, amrex::Real) {}

  void writeFinalSummary(lbm::runtime::DiagnosticRow const &row, bool,
                         lbm::runtime::SingleLevelFinalSummaryView const &) {
    if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
    }
    std::ofstream stream(metrics_path(), std::ios::out);
    const amrex::Real flowThroughs = static_cast<amrex::Real>(row.step) *
                                     std::abs(lbm_.uRefLb) / lbm_.lengthRefLb;
    stream << std::setprecision(17)
           << "step,flow_throughs,E_2_u,relative_mass_error\n"
           << row.step << ',' << flowThroughs << ',' << row.relativeL2U << ','
           << row.relativeMassError << '\n';
  }

private:
  std::string metrics_path() const {
    return controls_.outputDirectory + "/uniform_flow_metrics.csv";
  }

  Controls controls_;
  lbm::config::ResolvedLbmConfig const &lbm_;
};

template <class Descriptor> class RunSession {
public:
  RunSession() = default;
  explicit RunSession(Controls controls) : controls_(std::move(controls)) {}
  RunSession(RunSession const &) = delete;
  RunSession &operator=(RunSession const &) = delete;
  RunSession(RunSession &&) noexcept = default;
  RunSession &operator=(RunSession &&) noexcept = default;

  lbm::runtime::SingleLevelRunSessionResult
  start(lbm::runtime::SingleLevelRunContext<Descriptor> const &context) {
    lbm_ = context.lbm;
    output_ = std::make_unique<Output<Descriptor>>(controls_, lbm_);
    std::string error = output_->prepare();
    if (!error.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, error};
    }
    diagnostics_ = std::make_unique<Diagnostics>(
        lbm::runtime::SingleLevelDiagnosticsConfig{controls_.maxStep, 0, 0, 0,
                                                   0},
        lbm_, context.geometry, context.boxArray, context.distributionMapping,
        lbm::runtime::SingleLevelSteadyObservationPolicy{}, *output_);
    output_->print_identity();
    return {};
  }

  lbm::runtime::SingleLevelObservationPlan observationPlan() const noexcept {
    lbm::runtime::SingleLevelObservationPlan plan;
    plan.observeInitialized = true;
    plan.observeFinal = true;
    plan.macroAtInitialized = true;
    plan.macroAtFinal = true;
    return plan;
  }
  lbm::runtime::SingleLevelRunSessionResult observeInitialized(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->initialize(state.macro(), exact_reference());
    return {};
  }
  lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &) {
    return {};
  }
  lbm::runtime::SingleLevelRunSessionResult observeFinal(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->ensureFinalObservation(state.macro(), exact_reference(),
                                         state.step());
    diagnostics_->writeFinalSummary();
    return {};
  }
  lbm::runtime::SingleLevelRunSessionResult writePerformance(
      std::optional<lbm::profiling::TimestepPerformanceRecord> const &) {
    return {};
  }

private:
  using Diagnostics =
      lbm::runtime::SingleLevelDiagnostics<Descriptor, Output<Descriptor>>;
  ExactReference exact_reference() const noexcept {
    return {{controls_.rhoLb, controls_.velocityLb, 0.0, 0.0}, controls_.rhoLb};
  }

  Controls controls_;
  lbm::config::ResolvedLbmConfig lbm_;
  std::unique_ptr<Output<Descriptor>> output_;
  std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
  Controls controls;
  template <class Descriptor>
  using RunSession = numerical_uniform::RunSession<Descriptor>;
  template <class View>
  lbm::runtime::SingleLevelCasePreparationResult<
      RunSession<typename View::DescriptorType>>
  resolve(View const &view) {
    using Descriptor = typename View::DescriptorType;
    lbm::runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>>
        result;
    if constexpr (Descriptor::d == 3) {
      if (view.mesh.nCell[2] < 2) {
        result.diagnostics.push_back(
            {"invalid_uniform_depth", "mesh.n_cell.z",
             "D3Q27 numerical uniform flow requires at least two z cells"});
        return result;
      }
    }
    controls.maxStep = view.run.maxStep;
    result.session = RunSession<Descriptor>(std::move(controls));
    return result;
  }
};

lbm::boundary::DomainBoundarySpec inlet(Controls const &controls) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = uses_guo(controls.variant) ? "numerical_guo_uniform_inlet"
                                         : "numerical_hbb_uniform_inlet";
  spec.location = lbm::boundary::DomainSide::XLo;
  spec.physical = lbm::boundary::BoundaryPhysical::VelocityInlet;
  spec.scheme = uses_guo(controls.variant)
                    ? lbm::boundary::BoundaryScheme::GuoExtrapolation
                    : lbm::boundary::BoundaryScheme::HalfwayBounceBack;
  spec.profile = lbm::boundary::BoundaryProfileKind::Uniform;
  spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
  spec.hasPrescribedULb = true;
  return spec;
}

lbm::boundary::DomainBoundarySpec outlet(Controls const &controls) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = controls.variant == Variant::GuoFreeOutflow
                  ? "numerical_guo_free_outflow"
                  : "numerical_uniform_outlet";
  spec.location = lbm::boundary::DomainSide::XHi;
  spec.physical = controls.variant == Variant::GuoFreeOutflow
                      ? lbm::boundary::BoundaryPhysical::FreeOutflow
                      : lbm::boundary::BoundaryPhysical::VelocityOutlet;
  spec.scheme = uses_guo(controls.variant)
                    ? lbm::boundary::BoundaryScheme::GuoExtrapolation
                    : lbm::boundary::BoundaryScheme::HalfwayBounceBack;
  if (spec.physical == lbm::boundary::BoundaryPhysical::VelocityOutlet) {
    spec.profile = lbm::boundary::BoundaryProfileKind::Uniform;
    spec.prescribedULb = {controls.velocityLb, 0.0, 0.0};
    spec.hasPrescribedULb = true;
  }
  return spec;
}

template <class Descriptor> auto make_program(Controls const &controls) {
  using Profile = lbm::initialization::ConstantInitializationProfile;
  using Initialization =
      lbm::initialization::ProfileInitializationSelection<Profile>;
  using Program =
      lbm::runtime::SingleLevelProgram<Descriptor, lbm::BGKDynamicsSelection,
                                       Initialization, Preparation>;
  Program program;
  const bool guo = uses_guo(controls.variant);
  const amrex::Real dx = 1.0 / controls.intervals[0];
  program.mesh.nCell = {controls.intervals[0] + (guo ? 1 : 0),
                        controls.intervals[1], controls.depthCells};
  program.mesh.envelopeLo = {guo ? -0.5 * dx : 0.0, 0.0, 0.0};
  program.mesh.envelopeHi = {guo ? 1.0 + 0.5 * dx : 1.0, 1.0,
                             dx * controls.depthCells};
  program.mesh.isPeriodic = {0, 1, 1};
  program.mesh.maxGridSize = controls.maxGridSize;
  program.lbm.parameterizationKind =
      lbm::config::LbmParameterizationKind::FromNuLb;
  program.lbm.rhoRefLb = controls.rhoLb;
  program.lbm.uRefLb = controls.velocityLb;
  program.lbm.uRefPhys = controls.velocityReferencePhys;
  program.lbm.lengthRefLb = static_cast<amrex::Real>(controls.intervals[0]);
  program.lbm.lengthRefPhys = 1.0;
  program.lbm.nuLb = controls.nuLb;
  program.lbm.streamingBackend = std::string("baseline");
  program.dynamics = lbm::BGKDynamicsSelection{};
  program.run.maxStep = controls.maxStep;
  program.run.finalTimePhys = 1.0;
  program.run.kind = "long-run";
  program.run.name = "numerical_uniform_flow";
  program.run.variant = controls.runVariant;
  program.initialization = lbm::initialization::from_profile(
      Profile{{controls.rhoLb, controls.velocityLb, 0.0, 0.0}});
  program.domainBoundaries = {inlet(controls), outlet(controls)};
  program.casePreparation = Preparation{controls};
  return program;
}

std::string format_diagnostics(
    std::vector<lbm::runtime::SingleLevelSetupDiagnostic> const &diagnostics) {
  std::ostringstream stream;
  for (auto const &diagnostic : diagnostics) {
    stream << '[' << lbm::runtime::to_string(diagnostic.stage) << ':'
           << diagnostic.code << "] " << diagnostic.message << '\n';
  }
  return stream.str();
}

} // namespace numerical_uniform

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    auto controls = numerical_uniform::read_controls();
    auto preparation = lbm::runtime::prepare_single_level(
        numerical_uniform::make_program<numerical_uniform::ActiveDescriptor>(
            controls));
    if (!preparation.ok()) {
      amrex::Print() << numerical_uniform::format_diagnostics(
          preparation.diagnostics());
      status = 2;
    } else {
      auto report =
          lbm::runtime::run_single_level(std::move(preparation).take());
      if (!report.ok()) {
        amrex::Print() << report.message << '\n';
        status = 2;
      }
    }
  }
  amrex::Finalize();
  return status;
}
