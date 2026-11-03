#include "lbm/config/LbmConfigIntent.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/TaylorGreenVortexParams.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_Math.H>
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

namespace numerical_tgv {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

struct Controls {
  amrex::Array<int, 3> nCell{50, 50, 1};
  int maxGridSize = 25;
  amrex::Real rhoLb = 1.0;
  amrex::Real uRefPhys = 1.0;
  amrex::Real tau = 0.75;
  amrex::Real reynolds = 10.0;
  int maxStep = 100;
  amrex::Real finalTimePhys = 0.2;
  int convergenceEnabled = 1;
  int convergenceReset = 1;
  std::string runVariant;
  std::string outputDirectory;
  std::string logDirectory;
};

Controls read_controls() {
  Controls controls;
  amrex::ParmParse tgv("taylor_green");
  amrex::Vector<int> cells;
  tgv.getarr("n_cell", cells, 0, 3);
  controls.nCell = {cells[0], cells[1], cells[2]};
  tgv.get("rho_lb", controls.rhoLb);
  tgv.get("u_ref_phys", controls.uRefPhys);
  tgv.get("tau", controls.tau);
  tgv.get("re", controls.reynolds);

  amrex::ParmParse mesh("mesh");
  mesh.get("max_grid_size", controls.maxGridSize);
  amrex::ParmParse run("run");
  run.get("max_step", controls.maxStep);
  run.get("final_time_phys", controls.finalTimePhys);
  run.get("variant", controls.runVariant);
  run.get("convergence_enabled", controls.convergenceEnabled);
  run.get("convergence_reset", controls.convergenceReset);
  amrex::ParmParse output("output");
  output.get("directory", controls.outputDirectory);
  output.get("log_directory", controls.logDirectory);
  return controls;
}

struct ExactReference {
  lbm::initialization::TaylorGreenVortexParams params;
  amrex::Real massReferenceDensity = 1.0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE lbm::initialization::MacroscopicState
  operator()(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &position, int,
             int, int) const noexcept {
    const amrex::Real x = 2.0 * amrex::Math::pi<amrex::Real>() * position[0];
    const amrex::Real y = 2.0 * amrex::Math::pi<amrex::Real>() * position[1];
    const auto sx_cx = amrex::Math::sincos(x);
    const auto sy_cy = amrex::Math::sincos(y);
    const auto sx2_cx2 = amrex::Math::sincos(2.0 * x);
    const auto sy2_cy2 = amrex::Math::sincos(2.0 * y);
    return {params.rho0 - 0.25 * params.rho0 * params.u0 * params.u0 /
                              params.cs2 * (sx2_cx2.second + sy2_cy2.second) *
                              params.pressureDecay,
            -params.u0 * sx_cx.second * sy_cy.first * params.decay,
            params.u0 * sx_cx.first * sy_cy.second * params.decay, 0.0};
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
        std::filesystem::remove(csv_path());
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
    amrex::Print() << "descriptor = " << Descriptor::name() << '\n'
                   << "case_name = taylor_green_vortex\n"
                   << "case_variant = " << controls_.runVariant << '\n';
  }

  void printAndLog(std::string const &text) { amrex::Print() << text; }
  void writeDiagnosticObservation(lbm::runtime::DiagnosticRow const &) {}
  void writeDiagnosticsRow(lbm::runtime::DiagnosticRow const &, bool) const {}
  void writePlotfile(lbm::MacroField const &, amrex::Geometry const &,
                     amrex::BoxArray const &,
                     amrex::DistributionMapping const &, int, amrex::Real) {}

  void writeConvergenceRow(lbm::runtime::DiagnosticRow const &row) const {
    if (!amrex::ParallelDescriptor::IOProcessor()) {
      return;
    }
    std::ofstream stream(csv_path(), std::ios::out);
    stream << std::setprecision(17)
           << "N,h,Re,target_time_phys,time_phys,relative_l2_u,linf_uz\n"
           << controls_.nCell[0] << ','
           << 1.0 / static_cast<amrex::Real>(controls_.nCell[0]) << ','
           << lbm_.reynolds << ',' << controls_.finalTimePhys << ','
           << row.timePhys << ',' << row.relativeL2U << ',' << row.linfUz
           << '\n';
  }

  void writeFinalSummary(lbm::runtime::DiagnosticRow const &, bool,
                         lbm::runtime::SingleLevelFinalSummaryView const &) {}

private:
  std::string csv_path() const {
    return controls_.outputDirectory + "/convergence.csv";
  }

  Controls controls_;
  lbm::config::ResolvedLbmConfig const &lbm_;
};

template <class Descriptor> class RunSession {
public:
  RunSession() = default;
  RunSession(Controls controls,
             lbm::initialization::TaylorGreenVortexParams initial)
      : controls_(std::move(controls)), initial_(initial) {}
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
                                                   1},
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
    diagnostics_->initialize(state.macro(), exact_reference(0));
    return {};
  }

  lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &) {
    return {};
  }

  lbm::runtime::SingleLevelRunSessionResult observeFinal(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->ensureFinalObservation(
        state.macro(), exact_reference(state.step()), state.step());
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

  ExactReference exact_reference(int step) const {
    auto params = initial_;
    params.decay =
        std::exp(-params.nu * (params.kx * params.kx + params.ky * params.ky) *
                 static_cast<amrex::Real>(step));
    params.pressureDecay = params.decay * params.decay;
    return {params, initial_.rho0};
  }

  Controls controls_;
  lbm::initialization::TaylorGreenVortexParams initial_;
  lbm::config::ResolvedLbmConfig lbm_;
  std::unique_ptr<Output<Descriptor>> output_;
  std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
  Controls controls;

  template <class Descriptor>
  using RunSession = numerical_tgv::RunSession<Descriptor>;

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
            {"invalid_tgv_depth", "mesh.n_cell.z",
             "D3Q27 numerical TGV requires at least two z cells"});
        return result;
      }
    }
    controls.nCell = view.mesh.nCell;
    controls.maxStep = view.run.maxStep;
    controls.finalTimePhys = view.run.finalTimePhys;
    controls.runVariant = view.run.variant;
    result.session = RunSession<Descriptor>(
        std::move(controls), view.initialization.resolved().taylorGreenParams);
    return result;
  }
};

template <class Descriptor> auto make_program(Controls const &controls) {
  using Program = lbm::runtime::SingleLevelProgram<
      Descriptor, lbm::BGKDynamicsSelection,
      lbm::initialization::ConfiguredInitializationSelection, Preparation>;
  Program program;
  program.mesh.nCell = controls.nCell;
  program.mesh.envelopeLo = {0.0, 0.0, 0.0};
  program.mesh.envelopeHi = {1.0, 1.0,
                             static_cast<amrex::Real>(controls.nCell[2]) /
                                 static_cast<amrex::Real>(controls.nCell[0])};
  program.mesh.isPeriodic = {1, 1, 1};
  program.mesh.maxGridSize = controls.maxGridSize;
  program.lbm.parameterizationKind =
      lbm::config::LbmParameterizationKind::FromTau;
  program.lbm.rhoRefLb = controls.rhoLb;
  program.lbm.uRefPhys = controls.uRefPhys;
  program.lbm.tau = controls.tau;
  program.lbm.reynolds = controls.reynolds;
  program.lbm.streamingBackend = std::string("baseline");
  program.dynamics = lbm::BGKDynamicsSelection{};
  program.run.maxStep = controls.maxStep;
  program.run.finalTimePhys = controls.finalTimePhys;
  program.run.kind = "convergence";
  program.run.name = "numerical_taylor_green_vortex";
  program.run.variant = controls.runVariant;
  program.run.convergenceEnabled = controls.convergenceEnabled;
  program.run.convergenceReset = controls.convergenceReset;
  program.initialization.intent.kind =
      lbm::initialization::InitializationKind::TaylorGreen;
  program.initialization.intent.rhoLb = controls.rhoLb;
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

} // namespace numerical_tgv

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    auto controls = numerical_tgv::read_controls();
    auto preparation = lbm::runtime::prepare_single_level(
        numerical_tgv::make_program<numerical_tgv::ActiveDescriptor>(controls));
    if (!preparation.ok()) {
      amrex::Print() << numerical_tgv::format_diagnostics(
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
