#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/dynamics/Moments.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/SingleLevelDiagnosticOutput.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelProgram.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>
#include <AMReX_Utility.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace numerical_channel {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

struct Controls {
  int reynolds = 20;
  int diameterIntervals = 20;
  int depthCells = 1;
  int maxGridSize = 64;
  amrex::Real rhoLb = 1.0;
  amrex::Real uMeanLb = 0.035;
  amrex::Real uMeanPhys = 0.2;
  amrex::Real reynoldsHeight = 82.0;
  int maxStep = 60000;
  amrex::Real finalTimePhys = 52.5;
  int diagnosticsInterval = 1000;
  int steadyCheckInterval = 100;
  amrex::Real steadyTolerance = 5.0e-7;
  int forceInterval = 1;
  int outputInterval = 100;
  int writeHistory = 0;
  std::string runVariant;
  std::string outputDirectory;
  std::string logDirectory;

  std::string reference_label() const {
    return reynolds == 20 ? "dfg_2d_1_re20" : "dfg_2d_2_re100";
  }
};

Controls read_controls() {
  Controls controls;
  amrex::ParmParse channel("numerical_channel");
  channel.get("reynolds", controls.reynolds);
  channel.get("diameter_intervals", controls.diameterIntervals);
  channel.get("depth_cells", controls.depthCells);
  channel.get("rho_lb", controls.rhoLb);
  channel.get("u_mean_lb", controls.uMeanLb);
  channel.get("u_mean_phys", controls.uMeanPhys);
  channel.get("reynolds_height", controls.reynoldsHeight);

  amrex::ParmParse mesh("mesh");
  mesh.get("max_grid_size", controls.maxGridSize);
  amrex::ParmParse run("run");
  run.get("max_step", controls.maxStep);
  run.get("final_time_phys", controls.finalTimePhys);
  run.get("variant", controls.runVariant);
  amrex::ParmParse diagnostics("diagnostics");
  diagnostics.get("interval", controls.diagnosticsInterval);
  diagnostics.get("steady_check_interval", controls.steadyCheckInterval);
  diagnostics.get("steady_l2_tolerance", controls.steadyTolerance);
  amrex::ParmParse observation("channel_observation");
  observation.get("force_interval", controls.forceInterval);
  observation.get("output_interval", controls.outputInterval);
  observation.get("write_history", controls.writeHistory);
  amrex::ParmParse output("output");
  output.get("directory", controls.outputDirectory);
  output.get("log_directory", controls.logDirectory);

  const bool re20Grid = controls.diameterIntervals == 20 ||
                        controls.diameterIntervals == 40 ||
                        controls.diameterIntervals == 80;
  const bool re100Grid =
      controls.diameterIntervals == 40 || controls.diameterIntervals == 80;
  const amrex::Real expectedReH = controls.reynolds == 20 ? 82.0 : 410.0;
  const amrex::Real expectedUPhys = controls.reynolds == 20 ? 0.2 : 1.0;
  const bool valid =
      (controls.reynolds == 20 || controls.reynolds == 100) &&
      ((controls.reynolds == 20 && re20Grid) ||
       (controls.reynolds == 100 && re100Grid)) &&
      controls.depthCells > 0 && controls.maxGridSize > 0 &&
      controls.rhoLb == 1.0 && controls.uMeanLb == 0.035 &&
      std::abs(controls.reynoldsHeight - expectedReH) <= 1.0e-12 &&
      std::abs(controls.uMeanPhys - expectedUPhys) <= 1.0e-12 &&
      controls.maxStep > 0 && controls.finalTimePhys > 0.0 &&
      controls.diagnosticsInterval > 0 && controls.steadyCheckInterval >= 0 &&
      controls.steadyTolerance > 0.0 && controls.forceInterval == 1 &&
      controls.outputInterval >= controls.forceInterval &&
      (controls.writeHistory == 0 || controls.writeHistory == 1) &&
      !controls.runVariant.empty() && !controls.outputDirectory.empty() &&
      !controls.logDirectory.empty();
  if (!valid) {
    amrex::Abort("invalid tests-owned channel-cylinder benchmark identity");
  }
  return controls;
}

struct ConstantReference {
  amrex::Real massReferenceDensity = 1.0;
  amrex::Real ux = 0.035;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE lbm::initialization::MacroscopicState
  operator()(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &, int, int,
             int) const noexcept {
    return {massReferenceDensity, ux, 0.0, 0.0};
  }
};

struct PressureSample {
  amrex::Real rhoFront = 0.0;
  amrex::Real rhoRear = 0.0;
  amrex::Real coefficient = 0.0;
};

template <class Descriptor>
PressureSample sample_pressure(lbm::LatticeField<Descriptor> const &field,
                               amrex::IntVect const &front,
                               amrex::IntVect const &rear, int span,
                               amrex::Real rhoRef, amrex::Real uMean) {
  using ReduceOps = amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum,
                                     amrex::ReduceOpSum, amrex::ReduceOpSum>;
  using ReduceData =
      amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real>;
  using ReduceTuple = typename ReduceData::Type;
  amrex::MultiFab const &populations = field.multifab();
  amrex::Box const domain = populations.boxArray().minimalBox();
  amrex::Box const probes(
      amrex::IntVect(AMREX_D_DECL(std::min(front[0], rear[0]), front[1],
                                  domain.smallEnd(2))),
      amrex::IntVect(AMREX_D_DECL(std::max(front[0], rear[0]), front[1],
                                  domain.bigEnd(2))));
  ReduceOps operations;
  ReduceData data(operations);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
  for (amrex::MFIter mfi(populations, false); mfi.isValid(); ++mfi) {
    amrex::Box const box = probes & mfi.validbox();
    if (!box.ok()) {
      continue;
    }
    auto const values = populations.const_array(mfi);
    const int frontI = front[0];
    const int rearI = rear[0];
    operations.eval(box, data,
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                      if (i != frontI && i != rearI) {
                        return {0.0, 0.0, 0.0, 0.0};
                      }
                      const amrex::Real rho = lbm::Moments<Descriptor>::rho(
                          lbm::ConstPopulationCell{values, i, j, k});
                      return i == frontI ? ReduceTuple{rho, 0.0, 1.0, 0.0}
                                         : ReduceTuple{0.0, rho, 0.0, 1.0};
                    });
  }
  ReduceTuple const local = data.value();
  std::array<amrex::Real, 4> global{amrex::get<0>(local), amrex::get<1>(local),
                                    amrex::get<2>(local), amrex::get<3>(local)};
  amrex::ParallelDescriptor::ReduceRealSum(global.data(), 4);
  AMREX_ALWAYS_ASSERT(global[2] == static_cast<amrex::Real>(span) &&
                      global[3] == static_cast<amrex::Real>(span));
  PressureSample result;
  result.rhoFront = global[0] / global[2];
  result.rhoRear = global[1] / global[3];
  result.coefficient = Descriptor::cs2() * (result.rhoFront - result.rhoRear) /
                       (rhoRef * uMean * uMean);
  return result;
}

class ArtifactWriter {
public:
  explicit ArtifactWriter(Controls const &controls) : controls_(controls) {}

  std::string prepare() {
    std::string error;
    if (amrex::ParallelDescriptor::IOProcessor()) {
      try {
        std::filesystem::create_directories(controls_.outputDirectory);
        std::filesystem::create_directories(controls_.logDirectory);
        std::filesystem::remove(metrics_path());
        std::filesystem::remove(diagnostics_path());
        std::filesystem::remove(history_path());
        if (controls_.writeHistory != 0) {
          history_.open(history_path(), std::ios::out | std::ios::trunc);
          if (!history_.is_open()) {
            error = "failed to open channel-cylinder force history";
          } else {
            history_ << "population_step,force_time_lb,t_star,reference_label,"
                        "object_name,object_id,Fx_lb,Fy_lb,Fz_lb,Cd,Cl\n";
          }
        }
      } catch (std::filesystem::filesystem_error const &exception) {
        error = exception.what();
      }
    }
    amrex::BroadcastString(error, amrex::ParallelDescriptor::MyProc(),
                           amrex::ParallelDescriptor::IOProcessorNumber(),
                           amrex::ParallelDescriptor::Communicator());
    return error;
  }

  void append_history(int step, amrex::Real fx, amrex::Real fy, amrex::Real fz,
                      amrex::Real cd, amrex::Real cl) {
    if (!amrex::ParallelDescriptor::IOProcessor() ||
        controls_.writeHistory == 0 || !error_.empty()) {
      return;
    }
    const amrex::Real forceTime = static_cast<amrex::Real>(step) - 0.5;
    history_ << std::setprecision(
                    std::numeric_limits<amrex::Real>::max_digits10)
             << step << ',' << forceTime << ','
             << forceTime * controls_.uMeanLb /
                    static_cast<amrex::Real>(controls_.diameterIntervals)
             << ',' << controls_.reference_label() << ",cylinder,0," << fx
             << ',' << fy << ',' << fz << ',' << cd << ',' << cl << '\n';
    if (!history_) {
      error_ = "failed to append channel-cylinder force history";
    }
  }

  std::string finalize_metrics(char const *descriptor, int sampleCount,
                               int finalStep, amrex::Real cd, amrex::Real cl,
                               amrex::Real cDeltaP, amrex::Real maxAbsCz,
                               bool steadyReached, amrex::Real steadyTolerance,
                               amrex::Real finalResidual) {
    if (amrex::ParallelDescriptor::IOProcessor()) {
      if (history_.is_open()) {
        history_.flush();
        if (!history_ && error_.empty()) {
          error_ = "failed to flush channel-cylinder force history";
        }
        history_.close();
      }
      if (error_.empty()) {
        std::ofstream metrics(metrics_path(), std::ios::out | std::ios::trunc);
        if (!metrics.is_open()) {
          error_ = "failed to open channel-cylinder metrics";
        } else {
          metrics << "reference_label,descriptor,reynolds,diameter_intervals,"
                     "depth_cells,sample_count,final_population_step,Cd_final,"
                     "Cl_final,C_delta_p,max_abs_Cz,steady_reached,"
                     "steady_tolerance,final_steady_residual\n"
                  << std::setprecision(
                         std::numeric_limits<amrex::Real>::max_digits10)
                  << controls_.reference_label() << ',' << descriptor << ','
                  << controls_.reynolds << ',' << controls_.diameterIntervals
                  << ',' << controls_.depthCells << ',' << sampleCount << ','
                  << finalStep << ',' << cd << ',' << cl << ',' << cDeltaP
                  << ',' << maxAbsCz << ','
                  << (steadyReached ? "true" : "false") << ','
                  << steadyTolerance << ',' << finalResidual << '\n';
          if (!metrics) {
            error_ = "failed to write channel-cylinder metrics";
          }
        }
      }
    }
    amrex::BroadcastString(error_, amrex::ParallelDescriptor::MyProc(),
                           amrex::ParallelDescriptor::IOProcessorNumber(),
                           amrex::ParallelDescriptor::Communicator());
    return error_;
  }

  void write_diagnostics(lbm::runtime::DiagnosticRow const &row,
                         bool reset) const {
    lbm::runtime::write_diagnostics_row(diagnostics_path(), row, reset);
  }

private:
  std::string diagnostics_path() const {
    return controls_.outputDirectory + "/diagnostics.csv";
  }
  std::string history_path() const {
    return controls_.outputDirectory + "/channel_cylinder_force_history.csv";
  }
  std::string metrics_path() const {
    return controls_.outputDirectory + "/channel_cylinder_metrics.csv";
  }

  Controls const &controls_;
  std::ofstream history_;
  std::string error_;
};

template <class Descriptor> class Output {
public:
  Output(Controls const &controls, lbm::config::ResolvedLbmConfig const &lbm)
      : controls_(controls), lbm_(lbm), artifacts_(controls) {}

  std::string prepare() { return artifacts_.prepare(); }
  ArtifactWriter &artifacts() noexcept { return artifacts_; }

  void print_identity(amrex::Geometry const &geometry) const {
    amrex::Print() << std::setprecision(12)
                   << "descriptor = " << Descriptor::name() << '\n'
                   << "case_name = channel_cylinder\n"
                   << "case_variant = " << controls_.runVariant << '\n'
                   << "reference_label = " << controls_.reference_label()
                   << '\n'
                   << "reynolds = " << controls_.reynolds << '\n'
                   << "diameter_intervals = " << controls_.diameterIntervals
                   << '\n'
                   << "depth_cells = " << controls_.depthCells << '\n'
                   << "valid_nodes = " << geometry.Domain().length(0) << 'x'
                   << geometry.Domain().length(1) << 'x'
                   << geometry.Domain().length(2) << '\n'
                   << "domain_boundary_scheme = guo_extrapolation\n"
                   << "internal_scheme = yms\n"
                   << "Re_H = " << lbm_.reynolds << '\n'
                   << "tau = " << lbm_.tau << '\n'
                   << "max_step = " << controls_.maxStep << '\n';
  }

  void printAndLog(std::string const &text) { amrex::Print() << text; }
  void writeDiagnosticObservation(lbm::runtime::DiagnosticRow const &row) {
    amrex::Print() << lbm::runtime::diagnostic_console_text(row);
  }
  void writeDiagnosticsRow(lbm::runtime::DiagnosticRow const &row,
                           bool reset) const {
    artifacts_.write_diagnostics(row, reset);
  }
  void writeConvergenceRow(lbm::runtime::DiagnosticRow const &) const {}
  void writePlotfile(lbm::MacroField const &, amrex::Geometry const &,
                     amrex::BoxArray const &,
                     amrex::DistributionMapping const &, int, amrex::Real) {}
  void
  writeFinalSummary(lbm::runtime::DiagnosticRow const &row, bool,
                    lbm::runtime::SingleLevelFinalSummaryView const &view) {
    amrex::Print() << std::setprecision(12)
                   << "final_diagnostic_step = " << row.step << '\n'
                   << "final_relative_mass_error = " << row.relativeMassError
                   << '\n'
                   << "steady_reached = "
                   << (view.steadyReached ? "true" : "false") << '\n'
                   << "final_relative_l2_delta_u = " << view.lastSteadyResidual
                   << '\n';
  }

private:
  Controls const &controls_;
  lbm::config::ResolvedLbmConfig const &lbm_;
  ArtifactWriter artifacts_;
};

template <class Descriptor> class RunSession {
public:
  RunSession() = default;
  RunSession(Controls controls, amrex::IntVect frontProbe,
             amrex::IntVect rearProbe, int span)
      : controls_(std::move(controls)), frontProbe_(frontProbe),
        rearProbe_(rearProbe), span_(span) {}

  lbm::runtime::SingleLevelRunSessionResult
  start(lbm::runtime::SingleLevelRunContext<Descriptor> const &context) {
    lbm_ = context.lbm;
    output_ = std::make_unique<Output<Descriptor>>(controls_, lbm_);
    std::string error = output_->prepare();
    if (!error.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, error};
    }
    diagnostics_ = std::make_unique<Diagnostics>(
        lbm::runtime::SingleLevelDiagnosticsConfig{
            controls_.maxStep, controls_.diagnosticsInterval, 0, 0, 0},
        lbm_, context.geometry, context.boxArray, context.distributionMapping,
        lbm::runtime::SingleLevelSteadyObservationPolicy{
            controls_.steadyCheckInterval > 0, "channel_cylinder_steady",
            controls_.steadyCheckInterval, 0, controls_.steadyTolerance},
        *output_);
    output_->print_identity(context.geometry);
    return {};
  }

  lbm::runtime::SingleLevelObservationPlan observationPlan() const noexcept {
    lbm::runtime::SingleLevelObservationPlan plan;
    plan.observeInitialized = true;
    plan.observeFinal = true;
    plan.macroAtInitialized = true;
    plan.macroAtFinal = true;
    plan.completed.addInterval(controls_.diagnosticsInterval);
    plan.completed.includeMaxStep = true;
    plan.completedMacro = plan.completed;
    if (controls_.steadyCheckInterval > 0) {
      plan.completed.everyStep = true;
      plan.completedMacro.everyStep = true;
    }
    plan.forceCalculationInterval = controls_.forceInterval;
    plan.forceDeliveryInterval = controls_.outputInterval;
    return plan;
  }

  lbm::runtime::SingleLevelRunSessionResult observeInitialized(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    diagnostics_->initialize(
        state.macro(), ConstantReference{controls_.rhoLb, controls_.uMeanLb});
    return {};
  }

  lbm::runtime::SingleLevelCompletedObservationResult observeCompleted(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    consume(state.forceBatch());
    bool stop = false;
    if (state.hasMacro()) {
      stop = diagnostics_->onStep(
          state.macro(), ConstantReference{controls_.rhoLb, controls_.uMeanLb},
          state.step());
    }
    return {{}, stop};
  }

  lbm::runtime::SingleLevelRunSessionResult observeFinal(
      lbm::runtime::SingleLevelPublishedState<Descriptor> const &state) {
    consume(state.forceBatch());
    diagnostics_->ensureFinalObservation(
        state.macro(), ConstantReference{controls_.rhoLb, controls_.uMeanLb},
        state.step());
    PressureSample const pressure =
        sample_pressure(state.populations(), frontProbe_, rearProbe_, span_,
                        controls_.rhoLb, controls_.uMeanLb);
    std::string error = output_->artifacts().finalize_metrics(
        Descriptor::name(), sampleCount_, lastStep_, latestCd_, latestCl_,
        pressure.coefficient, maxAbsCz_, diagnostics_->steadyReached(),
        controls_.steadyTolerance, diagnostics_->lastSteadyResidual());
    if (!error.empty()) {
      return {lbm::runtime::SingleLevelRunStatus::OutputFailure, error};
    }
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

  void
  consume(lbm::boundary::InternalForceBatch const &batch) {
    const amrex::Real denominator =
        controls_.rhoLb * controls_.uMeanLb * controls_.uMeanLb *
        static_cast<amrex::Real>(controls_.diameterIntervals) *
        static_cast<amrex::Real>(span_);
    for (auto const &sample : batch.objectSteps) {
      AMREX_ALWAYS_ASSERT(sample.step.populationStep > lastStep_ &&
                          sample.objectForces.size() == 1 &&
                          sample.objectForces.front().objectId.value == 0);
      auto const &force = sample.objectForces.front().forceLb;
      latestCd_ = 2.0 * force[0] / denominator;
      latestCl_ = 2.0 * force[1] / denominator;
      const amrex::Real cz = 2.0 * force[2] / denominator;
      maxAbsCz_ = std::max(maxAbsCz_, std::abs(cz));
      output_->artifacts().append_history(sample.step.populationStep, force[0],
                                          force[1], force[2], latestCd_,
                                          latestCl_);
      lastStep_ = sample.step.populationStep;
      ++sampleCount_;
    }
  }

  Controls controls_;
  amrex::IntVect frontProbe_{AMREX_D_DECL(0, 0, 0)};
  amrex::IntVect rearProbe_{AMREX_D_DECL(0, 0, 0)};
  int span_ = 1;
  int sampleCount_ = 0;
  int lastStep_ = -1;
  amrex::Real latestCd_ = 0.0;
  amrex::Real latestCl_ = 0.0;
  amrex::Real maxAbsCz_ = 0.0;
  lbm::config::ResolvedLbmConfig lbm_;
  std::unique_ptr<Output<Descriptor>> output_;
  std::unique_ptr<Diagnostics> diagnostics_;
};

struct Preparation {
  Controls controls;

  template <class Descriptor>
  using RunSession = numerical_channel::RunSession<Descriptor>;

  template <class View>
  lbm::runtime::SingleLevelCasePreparationResult<
      RunSession<typename View::DescriptorType>>
  resolve(View const &view) {
    using Descriptor = typename View::DescriptorType;
    lbm::runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>>
        result;
    auto reject = [&](std::string message) {
      result.diagnostics.push_back({"invalid_numerical_channel_identity",
                                    "channel_cylinder", std::move(message)});
    };
    if constexpr (Descriptor::d == 2) {
      if (controls.depthCells != 1) {
        reject("D2Q9 requires one z cell");
      }
    } else {
      if (controls.reynolds != 20 || controls.diameterIntervals != 20 ||
          controls.depthCells != 4) {
        reject("quasi-D3 is restricted to Re20, N_D=20, N_z=4");
      }
    }
    if (view.boundaries.internal.size() != 1 ||
        view.boundaries.internalObjects.size() != 1 ||
        view.boundaries.internal.front().scheme !=
            lbm::boundary::BoundaryScheme::Yms ||
        view.boundaries.internalObjects.front().objectId.value != 0 ||
        view.boundaries.internalObjects.front().name != "cylinder") {
      reject("expected exactly one tests-owned YMS cylinder object");
    }
    if (!result.diagnostics.empty()) {
      return result;
    }
    const amrex::Real dx = view.geometry.CellSize(0);
    auto cell_index = [&](amrex::Real coordinate, int axis) {
      return static_cast<int>(
          std::llround((coordinate - view.geometry.ProbLo(axis)) /
                           view.geometry.CellSize(axis) -
                       0.5));
    };
    const int frontI = cell_index(0.15, 0);
    const int rearI = cell_index(0.25, 0);
    const int probeJ = cell_index(0.20, 1);
    const int span = view.geometry.Domain().length(2);
    const amrex::Real diameterLb = 0.1 / dx;
    if (std::abs(diameterLb - controls.diameterIntervals) > 1.0e-10 ||
        frontI < view.geometry.Domain().smallEnd(0) ||
        rearI > view.geometry.Domain().bigEnd(0) ||
        probeJ < view.geometry.Domain().smallEnd(1) ||
        probeJ > view.geometry.Domain().bigEnd(1)) {
      reject("grid does not align with the DFG cylinder and pressure probes");
      return result;
    }
    controls.maxStep = view.run.maxStep;
    result.session = RunSession<Descriptor>(
        std::move(controls),
        amrex::IntVect(
            AMREX_D_DECL(frontI, probeJ, view.geometry.Domain().smallEnd(2))),
        amrex::IntVect(
            AMREX_D_DECL(rearI, probeJ, view.geometry.Domain().smallEnd(2))),
        span);
    return result;
  }
};

lbm::boundary::DomainBoundarySpec inlet(Controls const &controls) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = "inlet";
  spec.location = lbm::boundary::DomainSide::XLo;
  spec.physical = lbm::boundary::BoundaryPhysical::VelocityInlet;
  spec.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
  spec.profile = lbm::boundary::BoundaryProfileKind::Poiseuille;
  spec.profileAxis = 1;
  spec.hasProfileAxis = true;
  spec.uMeanLb = controls.uMeanLb;
  spec.hasUMeanLb = true;
  return spec;
}

lbm::boundary::DomainBoundarySpec outlet(Controls const &controls) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = "outlet";
  spec.location = lbm::boundary::DomainSide::XHi;
  spec.physical = lbm::boundary::BoundaryPhysical::PressureOutlet;
  spec.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
  spec.profile = lbm::boundary::BoundaryProfileKind::Constant;
  spec.rhoLb = controls.rhoLb;
  spec.hasRhoLb = true;
  return spec;
}

lbm::boundary::DomainBoundarySpec wall(char const *name,
                                       lbm::boundary::DomainSide side) {
  lbm::boundary::DomainBoundarySpec spec;
  spec.name = name;
  spec.location = side;
  spec.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
  spec.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
  return spec;
}

lbm::boundary::InternalBoundarySpec cylinder(Controls const &controls,
                                             amrex::Real dx) {
  lbm::boundary::InternalBoundarySpec spec;
  spec.name = "cylinder_wall";
  spec.geometry = "cylinder";
  spec.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
  spec.hasPhysical = true;
  spec.scheme = lbm::boundary::BoundaryScheme::Yms;
  spec.objectName = "cylinder";
  spec.centerPhys = {0.2, 0.2, 0.5 * dx * controls.depthCells};
  spec.hasCenterPhys = true;
  spec.radiusPhys = 0.05;
  spec.hasRadiusPhys = true;
  spec.axis = "z";
  spec.hasAxis = true;
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
  const amrex::Real dx =
      0.1 / static_cast<amrex::Real>(controls.diameterIntervals);
  const int xIntervals = 22 * controls.diameterIntervals;
  const int yIntervals = 41 * controls.diameterIntervals / 10;
  program.mesh.nCell = {xIntervals + 1, yIntervals + 1, controls.depthCells};
  program.mesh.envelopeLo = {-0.5 * dx, -0.5 * dx, 0.0};
  program.mesh.envelopeHi = {2.2 + 0.5 * dx, 0.41 + 0.5 * dx,
                             dx * controls.depthCells};
  program.mesh.isPeriodic = {0, 0, 1};
  program.mesh.maxGridSize = controls.maxGridSize;
  program.lbm.parameterizationKind =
      lbm::config::LbmParameterizationKind::FromReynolds;
  program.lbm.rhoRefLb = controls.rhoLb;
  program.lbm.uRefLb = controls.uMeanLb;
  program.lbm.uRefPhys = controls.uMeanPhys;
  program.lbm.lengthRefLb = static_cast<amrex::Real>(yIntervals);
  program.lbm.lengthRefPhys = 0.41;
  program.lbm.reynolds = controls.reynoldsHeight;
  program.lbm.streamingBackend = std::string("baseline");
  program.dynamics = lbm::BGKDynamicsSelection{};
  program.run.maxStep = controls.maxStep;
  program.run.finalTimePhys = controls.finalTimePhys;
  program.run.kind = "validation";
  program.run.name = "numerical_channel_cylinder";
  program.run.variant = controls.runVariant;
  program.initialization = lbm::initialization::from_profile(
      Profile{{controls.rhoLb, controls.uMeanLb, 0.0, 0.0}});
  program.domainBoundaries = {
      inlet(controls), outlet(controls),
      wall("bottom_wall", lbm::boundary::DomainSide::YLo),
      wall("top_wall", lbm::boundary::DomainSide::YHi)};
  program.internalBoundaries = {cylinder(controls, dx)};
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

} // namespace numerical_channel

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  int status = 0;
  {
    auto controls = numerical_channel::read_controls();
    auto preparation = lbm::runtime::prepare_single_level(
        numerical_channel::make_program<numerical_channel::ActiveDescriptor>(
            controls));
    if (!preparation.ok()) {
      amrex::Print() << numerical_channel::format_diagnostics(
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
