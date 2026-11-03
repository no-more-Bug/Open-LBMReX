#include "common/HostFieldSnapshot.H"
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
#include <AMReX_BLassert.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace numerical_cavity {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

struct Controls {
  amrex::Array<amrex::Real, 2> physicalLo{0.0, 0.0};
  amrex::Array<amrex::Real, 2> physicalHi{1.0, 1.0};
  amrex::Array<int, 2> intervals{129, 129};
  int depthCells = 1;
  amrex::Real rhoLb = 1.0;
  amrex::Real reynolds = 1000.0;
  amrex::Real lidVelocityLb = 0.1;
  amrex::Real velocityReferencePhys = 1.0;
  int maxGridSize = 32;
  int maxStep = 200000;
  int diagnosticsInterval = 5000;
  int steadyCheckInterval = 5000;
  amrex::Real steadyTolerance = 5.0e-8;
  int stopOnSteady = 1;
  std::string runVariant;
  std::string outputDirectory;
  std::string logDirectory;
};

Controls read_controls() {
  Controls controls;
  amrex::ParmParse cavity("cavity");
  amrex::Vector<amrex::Real> lo;
  amrex::Vector<amrex::Real> hi;
  amrex::Vector<int> intervals;
  cavity.getarr("physical_lo", lo, 0, 2);
  cavity.getarr("physical_hi", hi, 0, 2);
  cavity.getarr("intervals", intervals, 0, 2);
  controls.physicalLo = {lo[0], lo[1]};
  controls.physicalHi = {hi[0], hi[1]};
  controls.intervals = {intervals[0], intervals[1]};
  cavity.get("depth_cells", controls.depthCells);
  std::string scheme;
  cavity.get("wall_scheme", scheme);
  if (scheme != "halfway_bounce_back") {
    amrex::Abort("numerical cavity requires halfway_bounce_back");
  }
  cavity.get("rho_lb", controls.rhoLb);
  cavity.get("re", controls.reynolds);
  cavity.get("lid_velocity_lb", controls.lidVelocityLb);
  cavity.get("u_ref_phys", controls.velocityReferencePhys);
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

struct QuiescentReference {
  amrex::Real massReferenceDensity = 1.0;
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE lbm::initialization::MacroscopicState
  operator()(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &, int, int,
             int) const noexcept {
    return {massReferenceDensity, 0.0, 0.0, 0.0};
  }
};

struct Centerlines {
  int centerI = 0;
  int centerJ = 0;
  std::vector<amrex::Real> y;
  std::vector<amrex::Real> ux;
  std::vector<amrex::Real> x;
  std::vector<amrex::Real> uy;
};

amrex::Real sample_z_average(amrex::MultiFab const &field, int i, int j,
                             int component, amrex::Box const &domain) {
  amrex::Real sum = 0.0;
  amrex::Real count = 0.0;
  for (amrex::MFIter mfi(field); mfi.isValid(); ++mfi) {
    amrex::Box const &box = mfi.validbox();
    auto const values = field.const_array(mfi);
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
      amrex::IntVect const index(AMREX_D_DECL(i, j, k));
      if (box.contains(index)) {
        sum += values(i, j, k, component);
        count += 1.0;
      }
    }
  }
  amrex::ParallelDescriptor::ReduceRealSum(sum);
  amrex::ParallelDescriptor::ReduceRealSum(count);
  AMREX_ALWAYS_ASSERT(count > 0.0);
  return sum / count;
}

Centerlines sample_centerlines(lbm::MacroField const &macro,
                               amrex::Geometry const &geometry) {
  auto const field = lbm::tests::host_field_snapshot(macro.multifab());
  amrex::Box const domain = geometry.Domain();
  const int nx = domain.length(0);
  const int ny = domain.length(1);
  AMREX_ALWAYS_ASSERT(nx % 2 == 1 && ny % 2 == 1);
  Centerlines rows;
  rows.centerI = domain.smallEnd(0) + nx / 2;
  rows.centerJ = domain.smallEnd(1) + ny / 2;
  rows.y.reserve(ny);
  rows.ux.reserve(ny);
  for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
    rows.y.push_back((static_cast<amrex::Real>(j - domain.smallEnd(1)) + 0.5) /
                     ny);
    rows.ux.push_back(
        sample_z_average(field, rows.centerI, j, lbm::MacroField::Ux, domain));
  }
  rows.x.reserve(nx);
  rows.uy.reserve(nx);
  for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
    rows.x.push_back((static_cast<amrex::Real>(i - domain.smallEnd(0)) + 0.5) /
                     nx);
    rows.uy.push_back(
        sample_z_average(field, i, rows.centerJ, lbm::MacroField::Uy, domain));
  }
  return rows;
}

template <class Descriptor> class Output {
public:
  explicit Output(Controls controls) : controls_(std::move(controls)) {}

  std::string prepare() {
    std::string error;
    if (amrex::ParallelDescriptor::IOProcessor()) {
      try {
        std::filesystem::create_directories(controls_.outputDirectory);
        std::filesystem::create_directories(controls_.logDirectory);
        std::filesystem::remove(ux_path());
        std::filesystem::remove(uy_path());
        std::filesystem::remove(metadata_path());
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
    amrex::Print() << std::setprecision(12)
                   << "descriptor = " << Descriptor::name() << '\n'
                   << "case_name = lid_driven_cavity\n"
                   << "case_variant = " << controls_.runVariant << '\n'
                   << "wall_scheme = halfway_bounce_back\n"
                   << "Re = " << controls_.reynolds << '\n';
  }

  std::string write_centerlines(Centerlines const &rows, int step,
                                bool steadyReached) {
    std::string error;
    if (amrex::ParallelDescriptor::IOProcessor()) {
      const amrex::Real inverseLidSpeed = 1.0 / controls_.lidVelocityLb;
      std::ofstream ux(ux_path(), std::ios::out);
      std::ofstream uy(uy_path(), std::ios::out);
      std::ofstream metadata(metadata_path(), std::ios::out);
      if (!ux.is_open() || !uy.is_open() || !metadata.is_open()) {
        error = "failed to open numerical cavity artifact";
      } else {
        ux << std::setprecision(17) << "y_over_L,ux_over_Ulid\n";
        for (std::size_t index = 0; index < rows.y.size(); ++index) {
          ux << rows.y[index] << ',' << rows.ux[index] * inverseLidSpeed
             << '\n';
        }
        uy << std::setprecision(17) << "x_over_L,uy_over_Ulid\n";
        for (std::size_t index = 0; index < rows.x.size(); ++index) {
          uy << rows.x[index] << ',' << rows.uy[index] * inverseLidSpeed
             << '\n';
        }
        metadata << "step = " << step << '\n'
                 << "steady_reached = " << (steadyReached ? "true" : "false")
                 << '\n'
                 << "boundary_geometry = domain_face_halfway\n"
                 << "sampling = interpolated_cell_center\n"
                 << "valid_nodes_x = " << controls_.intervals[0] << '\n'
                 << "valid_nodes_y = " << controls_.intervals[1] << '\n'
                 << "lattice_intervals_x = " << controls_.intervals[0] << '\n'
                 << "lattice_intervals_y = " << controls_.intervals[1] << '\n';
        if (!ux || !uy || !metadata) {
          error = "failed to write numerical cavity artifact";
        }
      }
    }
    amrex::BroadcastString(error, amrex::ParallelDescriptor::MyProc(),
                           amrex::ParallelDescriptor::IOProcessorNumber(),
                           amrex::ParallelDescriptor::Communicator());
    return error;
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
  std::string ux_path() const {
    return controls_.outputDirectory + "/cavity_centerline_x0p5_ux.csv";
  }
  std::string uy_path() const {
    return controls_.outputDirectory + "/cavity_centerline_y0p5_uy.csv";
  }
  std::string metadata_path() const {
    return controls_.outputDirectory + "/cavity_centerline_metadata.txt";
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
            true, "cavity_steady", controls_.steadyCheckInterval,
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
    diagnostics_->initialize(state.macro(),
                             QuiescentReference{controls_.rhoLb});
    return {};
  }
  lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    const bool stop = diagnostics_->onStep(
        state.macro(), QuiescentReference{controls_.rhoLb}, state.step());
    return {{}, stop};
  }
  lbm::runtime::SingleLevelRunSessionResult observeFinal(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->ensureFinalObservation(
        state.macro(), QuiescentReference{controls_.rhoLb}, state.step());
    Centerlines const rows = sample_centerlines(state.macro(), *geometry_);
    std::string error = output_->write_centerlines(
        rows, state.step(), diagnostics_->steadyReached());
    if (!error.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, error};
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
  Controls controls_;
  lbm::config::ResolvedLbmConfig lbm_;
  amrex::Geometry const *geometry_ = nullptr;
  std::unique_ptr<Output<Descriptor>> output_;
  std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
  Controls controls;
  template <class Descriptor>
  using RunSession = numerical_cavity::RunSession<Descriptor>;
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
            {"invalid_cavity_depth", "mesh.n_cell.z",
             "D3Q27 numerical cavity requires at least two z cells"});
        return result;
      }
    }
    controls.maxStep = view.run.maxStep;
    result.session = RunSession<Descriptor>(std::move(controls));
    return result;
  }
};

lbm::boundary::DomainBoundarySpec
stationary_wall(char const *name, lbm::boundary::DomainSide side) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = name;
  spec.location = side;
  spec.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
  spec.scheme = lbm::boundary::BoundaryScheme::HalfwayBounceBack;
  return spec;
}

lbm::boundary::DomainBoundarySpec moving_lid(Controls const &controls) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = "numerical_moving_lid";
  spec.location = lbm::boundary::DomainSide::YHi;
  spec.physical = lbm::boundary::BoundaryPhysical::MovingWall;
  spec.scheme = lbm::boundary::BoundaryScheme::HalfwayBounceBack;
  spec.profile = lbm::boundary::BoundaryProfileKind::Uniform;
  spec.prescribedULb = {controls.lidVelocityLb, 0.0, 0.0};
  spec.hasPrescribedULb = true;
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
  const amrex::Real length = controls.physicalHi[0] - controls.physicalLo[0];
  const amrex::Real dx = length / controls.intervals[0];
  program.mesh.nCell = {controls.intervals[0], controls.intervals[1],
                        controls.depthCells};
  program.mesh.envelopeLo = {controls.physicalLo[0], controls.physicalLo[1],
                             0.0};
  program.mesh.envelopeHi = {controls.physicalHi[0], controls.physicalHi[1],
                             dx * controls.depthCells};
  program.mesh.isPeriodic = {0, 0, 1};
  program.mesh.maxGridSize = controls.maxGridSize;
  program.lbm.parameterizationKind =
      lbm::config::LbmParameterizationKind::FromReynolds;
  program.lbm.rhoRefLb = controls.rhoLb;
  program.lbm.uRefLb = controls.lidVelocityLb;
  program.lbm.uRefPhys = controls.velocityReferencePhys;
  program.lbm.lengthRefLb = static_cast<amrex::Real>(controls.intervals[0]);
  program.lbm.lengthRefPhys = length;
  program.lbm.reynolds = controls.reynolds;
  program.lbm.streamingBackend = std::string("baseline");
  program.dynamics = lbm::BGKDynamicsSelection{};
  program.run.maxStep = controls.maxStep;
  program.run.finalTimePhys = 1.0;
  program.run.kind = "validation";
  program.run.name = "numerical_lid_driven_cavity";
  program.run.variant = controls.runVariant;
  program.initialization = lbm::initialization::from_profile(
      Profile{{controls.rhoLb, 0.0, 0.0, 0.0}});
  program.domainBoundaries = {
      stationary_wall("numerical_left_wall", lbm::boundary::DomainSide::XLo),
      stationary_wall("numerical_right_wall", lbm::boundary::DomainSide::XHi),
      stationary_wall("numerical_bottom_wall", lbm::boundary::DomainSide::YLo),
      moving_lid(controls)};
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

} // namespace numerical_cavity

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    auto controls = numerical_cavity::read_controls();
    auto preparation = lbm::runtime::prepare_single_level(
        numerical_cavity::make_program<numerical_cavity::ActiveDescriptor>(
            controls));
    if (!preparation.ok()) {
      amrex::Print() << numerical_cavity::format_diagnostics(
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
