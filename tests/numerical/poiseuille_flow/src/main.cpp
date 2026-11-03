#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
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

namespace numerical_poiseuille {

void fill_profile_error(
    amrex::Box const &box, amrex::Array4<amrex::Real const> values,
    amrex::Array4<amrex::Real> sums, int centerI, int loY, int hiY,
    amrex::Real intervals, amrex::Real uMax) {
  amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
    if (i == centerI && j != loY && j != hiY) {
      const amrex::Real eta = static_cast<amrex::Real>(j - loY) / intervals;
      const amrex::Real exact = 4.0 * uMax * eta * (1.0 - eta);
      const amrex::Real difference =
          values(i, j, k, lbm::MacroField::Ux) - exact;
      sums(i, j, k, 0) = difference * difference;
      sums(i, j, k, 1) = exact * exact;
    }
  });
}


#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

struct Controls {
  amrex::Array<int, 2> intervals{40, 10};
  int depthCells = 1;
  int maxGridSize = 16;
  amrex::Real rhoLb = 1.0;
  amrex::Real uMaxLb = 0.1;
  amrex::Real uRefPhys = 1.0;
  amrex::Real tau = 0.75;
  amrex::Real rhoInLb = 1.04;
  amrex::Real rhoOutLb = 0.96;
  int maxStep = 20000;
  int diagnosticsInterval = 100;
  int steadyCheckInterval = 100;
  amrex::Real steadyTolerance = 5.0e-8;
  int stopOnSteady = 1;
  std::string runVariant;
  std::string outputDirectory;
  std::string logDirectory;
};

Controls read_controls() {
  Controls controls;
  amrex::ParmParse flow("poiseuille_flow");
  std::string variant;
  std::string scaling;
  flow.get("variant", variant);
  flow.get("pressure_scaling", scaling);
  if (variant != "guo_pressure" || scaling != "fixed_re_fixed_tau") {
    amrex::Abort(
        "numerical Poiseuille requires guo_pressure/fixed_re_fixed_tau");
  }
  amrex::Vector<int> intervals;
  flow.getarr("intervals", intervals, 0, 2);
  controls.intervals = {intervals[0], intervals[1]};
  flow.get("depth_cells", controls.depthCells);
  flow.get("rho_lb", controls.rhoLb);
  flow.get("u_max_lb", controls.uMaxLb);
  flow.get("u_ref_phys", controls.uRefPhys);
  flow.get("tau", controls.tau);
  flow.get("rho_in_lb", controls.rhoInLb);
  flow.get("rho_out_lb", controls.rhoOutLb);
  amrex::ParmParse mesh("mesh");
  mesh.get("max_grid_size", controls.maxGridSize);
  amrex::ParmParse run("run");
  run.get("max_step", controls.maxStep);
  run.get("variant", controls.runVariant);
  amrex::ParmParse diagnostics("diagnostics");
  diagnostics.get("interval", controls.diagnosticsInterval);
  diagnostics.get("steady_check_interval", controls.steadyCheckInterval);
  diagnostics.get("steady_l2_tolerance", controls.steadyTolerance);
  diagnostics.get("stop_on_steady", controls.stopOnSteady);
  amrex::ParmParse output("output");
  output.get("directory", controls.outputDirectory);
  output.get("log_directory", controls.logDirectory);
  return controls;
}

struct ExactReference {
  amrex::Real rhoIn = 1.0;
  amrex::Real rhoOut = 1.0;
  amrex::Real uMax = 0.1;
  amrex::Real massReferenceDensity = 1.0;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE lbm::initialization::MacroscopicState
  operator()(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &position, int,
             int, int) const noexcept {
    const amrex::Real xi = position[0] / 4.0;
    const amrex::Real eta = position[1];
    return {rhoIn + xi * (rhoOut - rhoIn), 4.0 * uMax * eta * (1.0 - eta), 0.0,
            0.0};
  }
};

template <class Descriptor> class Output {
public:
  explicit Output(Controls controls) : controls_(std::move(controls)) {}

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
    amrex::Print() << "descriptor = " << Descriptor::name() << '\n'
                   << "case_name = poiseuille_flow\n"
                   << "case_variant = " << controls_.runVariant << '\n'
                   << "condition = pressure_inlet -> pressure_outlet\n"
                   << "scheme = guo_extrapolation\n"
                   << "profile = constant_density\n";
  }

  std::string write_profile_metric(int step, amrex::Real error) {
    std::string failure;
    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ofstream stream(metrics_path(), std::ios::out);
      if (!stream.is_open()) {
        failure = "failed to open Poiseuille metrics CSV";
      } else {
        stream << std::setprecision(17)
               << "n_intervals_y,step,E_2_profile,pressure_driven,"
                  "pressure_scaling_code\n"
               << controls_.intervals[1] << ',' << step << ',' << error
               << ",1,2\n";
        if (!stream) {
          failure = "failed to write Poiseuille metrics CSV";
        }
      }
    }
    amrex::BroadcastString(failure, amrex::ParallelDescriptor::MyProc(),
                           amrex::ParallelDescriptor::IOProcessorNumber(),
                           amrex::ParallelDescriptor::Communicator());
    return failure;
  }

  void printAndLog(std::string const &text) { amrex::Print() << text; }
  void writeDiagnosticObservation(lbm::runtime::DiagnosticRow const &) {}
  void writeDiagnosticsRow(lbm::runtime::DiagnosticRow const &, bool) const {}
  void writeConvergenceRow(lbm::runtime::DiagnosticRow const &) const {}
  void writePlotfile(lbm::MacroField const &, amrex::Geometry const &,
                     amrex::BoxArray const &,
                     amrex::DistributionMapping const &, int, amrex::Real) {}
  void writeFinalSummary(lbm::runtime::DiagnosticRow const &, bool,
                         lbm::runtime::SingleLevelFinalSummaryView const &) {}

private:
  std::string metrics_path() const {
    return controls_.outputDirectory + "/poiseuille_profile_metrics.csv";
  }
  Controls controls_;
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
    geometry_ = &context.geometry;
    boxArray_ = &context.boxArray;
    distributionMapping_ = &context.distributionMapping;
    output_ = std::make_unique<Output<Descriptor>>(controls_);
    std::string error = output_->prepare();
    if (!error.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, error};
    }
    diagnostics_ = std::make_unique<Diagnostics>(
        lbm::runtime::SingleLevelDiagnosticsConfig{
            controls_.maxStep, controls_.diagnosticsInterval, 0, 0, 0},
        lbm_, context.geometry, context.boxArray, context.distributionMapping,
        lbm::runtime::SingleLevelSteadyObservationPolicy{
            true, "poiseuille_steady", controls_.steadyCheckInterval,
            controls_.stopOnSteady, controls_.steadyTolerance},
        *output_);
    output_->print_identity();
    return {};
  }

  lbm::runtime::SingleLevelObservationPlan observationPlan() const noexcept {
    lbm::runtime::SingleLevelObservationPlan plan;
    plan.observeInitialized = true;
    plan.observeFinal = true;
    plan.macroAtInitialized = true;
    plan.macroAtFinal = true;
    plan.completed.everyStep = true;
    plan.completedMacro.everyStep = true;
    return plan;
  }
  lbm::runtime::SingleLevelRunSessionResult observeInitialized(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->initialize(state.macro(), exact_reference());
    return {};
  }
  lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    const bool stop =
        diagnostics_->onStep(state.macro(), exact_reference(), state.step());
    return {{}, stop};
  }
  lbm::runtime::SingleLevelRunSessionResult observeFinal(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->ensureFinalObservation(state.macro(), exact_reference(),
                                         state.step());
    const amrex::Real error = profile_error(state.macro());
    std::string failure = output_->write_profile_metric(state.step(), error);
    amrex::Print() << "poiseuille_steady_reached = "
                   << (diagnostics_->steadyReached() ? "true" : "false")
                   << '\n';
    if (!failure.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, failure};
    }
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
    return {controls_.rhoInLb, controls_.rhoOutLb, controls_.uMaxLb,
            controls_.rhoLb};
  }

  amrex::Real profile_error(lbm::MacroField const &macro) const {
    amrex::Box const domain = geometry_->Domain();
    const int centerI = domain.smallEnd(0) + domain.length(0) / 2;
    const int loY = domain.smallEnd(1);
    const int hiY = domain.bigEnd(1);
    const amrex::Real intervals =
        static_cast<amrex::Real>(controls_.intervals[1]);
    amrex::MultiFab scratch(*boxArray_, *distributionMapping_, 2, 0);
    scratch.setVal(0.0);
    auto const &field = macro.multifab();
    for (amrex::MFIter mfi(field, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      amrex::Box const box = mfi.tilebox();
      auto const values = field.const_array(mfi);
      auto const sums = scratch.array(mfi);
      const amrex::Real uMax = controls_.uMaxLb;
      fill_profile_error(box, values, sums, centerI, loY, hiY, intervals,
                         uMax);
    }
    amrex::Gpu::streamSynchronize();
    return std::sqrt(scratch.sum(0) / scratch.sum(1));
  }

  Controls controls_;
  lbm::config::ResolvedLbmConfig lbm_;
  amrex::Geometry const *geometry_ = nullptr;
  amrex::BoxArray const *boxArray_ = nullptr;
  amrex::DistributionMapping const *distributionMapping_ = nullptr;
  std::unique_ptr<Output<Descriptor>> output_;
  std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
  Controls controls;
  template <class Descriptor>
  using RunSession = numerical_poiseuille::RunSession<Descriptor>;
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
            {"invalid_poiseuille_depth", "mesh.n_cell.z",
             "D3Q27 numerical Poiseuille requires at least two z cells"});
        return result;
      }
    }
    controls.maxStep = view.run.maxStep;
    result.session = RunSession<Descriptor>(std::move(controls));
    return result;
  }
};

lbm::boundary::DomainBoundarySpec
pressure_face(Controls const &controls, lbm::boundary::DomainSide side) {
  const bool inlet = side == lbm::boundary::DomainSide::XLo;
  lbm::boundary::DomainBoundarySpec spec;
  spec.name =
      inlet ? "numerical_guo_pressure_inlet" : "numerical_guo_pressure_outlet";
  spec.location = side;
  spec.physical = inlet ? lbm::boundary::BoundaryPhysical::PressureInlet
                        : lbm::boundary::BoundaryPhysical::PressureOutlet;
  spec.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
  spec.profile = lbm::boundary::BoundaryProfileKind::Constant;
  spec.rhoLb = inlet ? controls.rhoInLb : controls.rhoOutLb;
  spec.hasRhoLb = true;
  return spec;
}

lbm::boundary::DomainBoundarySpec wall(lbm::boundary::DomainSide side) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = side == lbm::boundary::DomainSide::YLo
                  ? "numerical_guo_bottom_wall"
                  : "numerical_guo_top_wall";
  spec.location = side;
  spec.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
  spec.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
  return spec;
}

template <class Descriptor> auto make_program(Controls const &controls) {
  using Profile = lbm::initialization::PoiseuilleInitializationProfile;
  using Initialization =
      lbm::initialization::ProfileInitializationSelection<Profile>;
  using Program =
      lbm::runtime::SingleLevelProgram<Descriptor, lbm::BGKDynamicsSelection,
                                       Initialization, Preparation>;
  Program program;
  const amrex::Real dx = 1.0 / controls.intervals[1];
  program.mesh.nCell = {controls.intervals[0] + 1, controls.intervals[1] + 1,
                        controls.depthCells};
  program.mesh.envelopeLo = {-0.5 * dx, -0.5 * dx, 0.0};
  program.mesh.envelopeHi = {4.0 + 0.5 * dx, 1.0 + 0.5 * dx,
                             dx * controls.depthCells};
  program.mesh.isPeriodic = {0, 0, 1};
  program.mesh.maxGridSize = controls.maxGridSize;
  program.lbm.parameterizationKind =
      lbm::config::LbmParameterizationKind::FromTau;
  program.lbm.rhoRefLb = controls.rhoLb;
  program.lbm.uRefLb = controls.uMaxLb;
  program.lbm.uRefPhys = controls.uRefPhys;
  program.lbm.lengthRefLb = static_cast<amrex::Real>(controls.intervals[1]);
  program.lbm.lengthRefPhys = 1.0;
  program.lbm.tau = controls.tau;
  program.lbm.streamingBackend = std::string("baseline");
  program.dynamics = lbm::BGKDynamicsSelection{};
  program.run.maxStep = controls.maxStep;
  program.run.finalTimePhys = 1.0;
  program.run.kind = "steady";
  program.run.name = "numerical_poiseuille_flow";
  program.run.variant = controls.runVariant;
  lbm::initialization::PoiseuilleInitializationParams params;
  params.rhoLb = controls.rhoLb;
  params.uMaxLb = controls.uMaxLb;
  params.linearDensity = 1;
  params.rhoXLoLb = controls.rhoInLb;
  params.rhoXHiLb = controls.rhoOutLb;
  program.initialization = lbm::initialization::from_profile(Profile{params});
  program.domainBoundaries = {
      pressure_face(controls, lbm::boundary::DomainSide::XLo),
      pressure_face(controls, lbm::boundary::DomainSide::XHi),
      wall(lbm::boundary::DomainSide::YLo),
      wall(lbm::boundary::DomainSide::YHi)};
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

} // namespace numerical_poiseuille

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    auto controls = numerical_poiseuille::read_controls();
    auto preparation = lbm::runtime::prepare_single_level(
        numerical_poiseuille::make_program<
            numerical_poiseuille::ActiveDescriptor>(controls));
    if (!preparation.ok()) {
      amrex::Print() << numerical_poiseuille::format_diagnostics(
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
