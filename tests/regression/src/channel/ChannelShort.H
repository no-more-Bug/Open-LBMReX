#ifndef LBM_TESTS_REGRESSION_CHANNEL_CHANNELSHORT_H
#define LBM_TESTS_REGRESSION_CHANNEL_CHANNELSHORT_H

#include "common/ShortRunSupport.H"
#include "common/MacroSampling.H"

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/SingleLevelDiagnostics.H"
#include "runtime/SingleLevelProgram.H"

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::regression::channel {

enum class Variant {
    SingleYms,
    SingleBouzidi,
    TandemDistinct,
    TandemShared,
    Re20Yms,
    Re100Yms
};

struct Controls {
    Variant variant = Variant::SingleYms;
    int diameterIntervals = 20;
    int depthCells = 1;
    int maxGridSize = 64;
    amrex::Real rhoLb = 1.0;
    amrex::Real uMeanLb = 0.035;
    amrex::Real uMeanPhys = 0.2;
    amrex::Real reynoldsHeight = 82.0;
    int maxStep = 12;
    amrex::Real finalTimePhys = 1.0;
    int diagnosticsInterval = 12;
    int forceInterval = 1;
    int outputInterval = 10;
    int writeHistory = 1;
    std::string runKind = "regression";
    std::string runVariant;
    ShortOutputConfig output;
};

inline Variant parse_variant(std::string const& value)
{
    if (value == "single_yms") return Variant::SingleYms;
    if (value == "single_bouzidi") return Variant::SingleBouzidi;
    if (value == "tandem_distinct") return Variant::TandemDistinct;
    if (value == "tandem_shared") return Variant::TandemShared;
    if (value == "re20_yms") return Variant::Re20Yms;
    if (value == "re100_yms") return Variant::Re100Yms;
    amrex::Abort(
        "channel_cylinder.variant must be single_yms, single_bouzidi, "
        "tandem_distinct, tandem_shared, re20_yms or re100_yms");
    return Variant::SingleYms;
}

inline bool tandem(Variant variant) noexcept
{
    return variant == Variant::TandemDistinct ||
        variant == Variant::TandemShared;
}

inline bool uniform_initialization(Variant variant) noexcept
{
    return tandem(variant) || variant == Variant::Re20Yms ||
        variant == Variant::Re100Yms;
}

inline Controls read_direct_controls()
{
    Controls controls;
    amrex::ParmParse channel("channel_cylinder");
    std::string variant;
    channel.get("variant", variant);
    controls.variant = parse_variant(variant);
    channel.get("diameter_intervals", controls.diameterIntervals);
    channel.get("depth_cells", controls.depthCells);
    channel.get("rho_lb", controls.rhoLb);
    channel.get("u_mean_lb", controls.uMeanLb);
    channel.get("u_mean_phys", controls.uMeanPhys);
    channel.get("reynolds_height", controls.reynoldsHeight);
    amrex::ParmParse("mesh").get("max_grid_size", controls.maxGridSize);
    amrex::ParmParse run("run");
    run.get("max_step", controls.maxStep);
    run.get("final_time_phys", controls.finalTimePhys);
    run.get("kind", controls.runKind);
    run.get("variant", controls.runVariant);
    amrex::ParmParse diagnostics("diagnostics");
    diagnostics.get("interval", controls.diagnosticsInterval);
    amrex::ParmParse observation("channel_cylinder_observation");
    observation.get("force_interval", controls.forceInterval);
    observation.get("output_interval", controls.outputInterval);
    observation.get("write_history", controls.writeHistory);
    controls.output = {
        direct_output_directory(),
        direct_log_directory(),
        "Failed to prepare direct channel-cylinder output",
        "case_name = channel_cylinder\ncase_variant = " +
            controls.runVariant + "\n"};
    return controls;
}

struct ExactReference {
    amrex::Real massReferenceDensity = 1.0;
    amrex::Real ux = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        int,
        int,
        int) const noexcept
    {
        return {massReferenceDensity, ux, 0.0, 0.0};
    }
};

struct ObjectStatistics {
    std::string name;
    int objectId = -1;
    std::vector<amrex::Real> cd;
    std::vector<amrex::Real> cl;
};

inline std::string reference_label(Variant variant)
{
    if (tandem(variant)) return "tandem_two_cylinder";
    if (variant == Variant::Re100Yms) return "dfg_2d_2_re100";
    return "dfg_2d_1_re20";
}

inline std::vector<ObjectStatistics> observed_objects(Controls const& controls)
{
    if (controls.variant == Variant::TandemDistinct) {
        return {{"front_cylinder", 0, {}, {}},
                {"rear_cylinder", 1, {}, {}}};
    }
    if (controls.variant == Variant::TandemShared) {
        return {{"cylinder_pair", 0, {}, {}}};
    }
    return {{"cylinder", 0, {}, {}}};
}

class ForceObserver {
public:
    ForceObserver(Controls controls, amrex::Real spanLb)
        : controls_(std::move(controls)),
          spanLb_(spanLb),
          objects_(observed_objects(controls_))
    {}

    void consume(boundary::InternalForceBatch const& batch)
    {
        for (auto const& sample : batch.objectSteps) {
            AMREX_ALWAYS_ASSERT(
                sample.objectForces.size() == objects_.size());
            samples_.push_back(sample);
            for (std::size_t index = 0; index < objects_.size(); ++index) {
                auto const& force = sample.objectForces[index];
                AMREX_ALWAYS_ASSERT(force.objectId.value == objects_[index].objectId);
                const amrex::Real denominator = controls_.rhoLb *
                    controls_.uMeanLb * controls_.uMeanLb *
                    controls_.diameterIntervals * spanLb_;
                objects_[index].cd.push_back(2.0 * force.forceLb[0] / denominator);
                objects_[index].cl.push_back(2.0 * force.forceLb[1] / denominator);
            }
        }
    }

    std::string write(
        amrex::Real rhoFront,
        amrex::Real rhoRear,
        amrex::Real cs2) const
    {
        if (!amrex::ParallelDescriptor::IOProcessor()) return {};
        try {
            if (controls_.writeHistory != 0) write_history();
            write_metrics(rhoFront, rhoRear, cs2);
        } catch (std::ios_base::failure const& exception) {
            return exception.what();
        }
        return {};
    }

private:
    void write_history() const
    {
        std::ofstream stream(
            controls_.output.directory +
                "/channel_cylinder_force_history.csv",
            std::ios::out);
        stream << std::setprecision(17)
               << "population_step,force_time_lb,t_star,reference_label,"
                  "object_name,object_id,Fx_lb,Fy_lb,Fz_lb,Cd,Cl\n";
        for (std::size_t sampleIndex = 0; sampleIndex < samples_.size();
             ++sampleIndex) {
            auto const& sample = samples_[sampleIndex];
            const amrex::Real forceTime = sample.step.populationStep - 0.5;
            for (std::size_t objectIndex = 0; objectIndex < objects_.size();
                 ++objectIndex) {
                auto const& object = objects_[objectIndex];
                auto const& force = sample.objectForces[objectIndex].forceLb;
                stream << sample.step.populationStep << ',' << forceTime << ','
                       << forceTime * controls_.uMeanLb /
                              controls_.diameterIntervals
                       << ',' << reference_label(controls_.variant) << ','
                       << object.name << ',' << object.objectId << ','
                       << force[0] << ',' << force[1] << ',' << force[2] << ','
                       << object.cd[sampleIndex] << ','
                       << object.cl[sampleIndex] << '\n';
            }
        }
    }

    void write_metrics(
        amrex::Real rhoFront,
        amrex::Real rhoRear,
        amrex::Real cs2) const
    {
        std::ofstream stream(
            controls_.output.directory + "/channel_cylinder_metrics.csv",
            std::ios::out);
        stream << std::setprecision(17)
               << "reference_label,object_name,object_id,sample_count,"
                  "final_population_step,Cd_final,Cl_final,Cd_min,Cd_max,"
                  "Cd_midrange,Cd_peak_to_peak,Cl_min,Cl_max,Cl_midrange,"
                  "Cl_peak_to_peak,rho_front_lb,rho_rear_lb,delta_p_lb,"
                  "C_delta_p,gate_status\n";
        const amrex::Real deltaP = cs2 * (rhoFront - rhoRear);
        const amrex::Real pressureCoefficient =
            2.0 * deltaP /
            (controls_.rhoLb * controls_.uMeanLb * controls_.uMeanLb);
        for (auto const& object : objects_) {
            AMREX_ALWAYS_ASSERT(!object.cd.empty());
            const auto cdMinMax = std::minmax_element(
                object.cd.begin(), object.cd.end());
            const auto clMinMax = std::minmax_element(
                object.cl.begin(), object.cl.end());
            stream << reference_label(controls_.variant) << ',' << object.name
                   << ',' << object.objectId << ',' << object.cd.size() << ','
                   << samples_.back().step.populationStep << ',' << object.cd.back()
                   << ',' << object.cl.back() << ',' << *cdMinMax.first << ','
                   << *cdMinMax.second << ','
                   << 0.5 * (*cdMinMax.first + *cdMinMax.second) << ','
                   << *cdMinMax.second - *cdMinMax.first << ','
                   << *clMinMax.first << ',' << *clMinMax.second << ','
                   << 0.5 * (*clMinMax.first + *clMinMax.second) << ','
                   << *clMinMax.second - *clMinMax.first << ',' << rhoFront
                   << ',' << rhoRear << ',' << deltaP << ','
                   << pressureCoefficient << ",assert_disabled\n";
        }
    }

    Controls controls_;
    amrex::Real spanLb_ = 1.0;
    std::vector<ObjectStatistics> objects_;
    std::vector<boundary::InternalObjectForceSample> samples_;
};

template <class Descriptor>
class RunSession {
public:
    RunSession() = default;
    explicit RunSession(Controls controls) : controls_(std::move(controls)) {}
    RunSession(RunSession const&) = delete;
    RunSession& operator=(RunSession const&) = delete;
    RunSession(RunSession&&) noexcept = default;
    RunSession& operator=(RunSession&&) noexcept = default;

    runtime::SingleLevelRunSessionResult start(
        runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        lbm_ = context.lbm;
        domain_ = context.geometry.Domain();
        output_ = std::make_unique<ShortRunOutput>(controls_.output);
        std::string error = output_->prepare();
        if (!error.empty()) {
            return {runtime::SingleLevelRunStatus::OutputFailure, error};
        }
        observer_ = std::make_unique<ForceObserver>(
            controls_, Descriptor::d == 3 ? controls_.depthCells : 1.0);
        diagnostics_ = std::make_unique<Diagnostics>(
            runtime::SingleLevelDiagnosticsConfig{
                controls_.maxStep, controls_.diagnosticsInterval, 0, 0, 0},
            lbm_, context.geometry, context.boxArray,
            context.distributionMapping,
            runtime::SingleLevelSteadyObservationPolicy{}, *output_);
        output_->print_identity();
        return {};
    }

    runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        runtime::SingleLevelObservationPlan plan;
        plan.observeInitialized = true;
        plan.observeFinal = true;
        plan.macroAtInitialized = true;
        plan.macroAtFinal = true;
        plan.completed.addInterval(controls_.diagnosticsInterval);
        plan.completed.includeMaxStep = true;
        plan.completedMacro = plan.completed;
        plan.forceCalculationInterval = controls_.forceInterval;
        plan.forceDeliveryInterval = controls_.outputInterval;
        return plan;
    }

    runtime::SingleLevelRunSessionResult observeInitialized(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        diagnostics_->initialize(state.macro(), exact_reference());
        return {};
    }

    runtime::SingleLevelCompletedObservationResult observeCompleted(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        observer_->consume(state.forceBatch());
        if (state.hasMacro()) {
            diagnostics_->onStep(state.macro(), exact_reference(), state.step());
        }
        return {};
    }

    runtime::SingleLevelRunSessionResult observeFinal(
        runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        observer_->consume(state.forceBatch());
        diagnostics_->ensureFinalObservation(
            state.macro(), exact_reference(), state.step());
        const amrex::Real dx = 0.1 / controls_.diameterIntervals;
        const int frontI = domain_.smallEnd(0) +
            static_cast<int>(std::llround(0.15 / dx));
        const int rearI = domain_.smallEnd(0) +
            static_cast<int>(std::llround(0.25 / dx));
        const int probeJ = domain_.smallEnd(1) +
            static_cast<int>(std::llround(0.2 / dx));
        const amrex::Real rhoFront = sample_z_average(
            state.macro(), frontI, probeJ, MacroField::Rho, domain_);
        const amrex::Real rhoRear = sample_z_average(
            state.macro(), rearI, probeJ, MacroField::Rho, domain_);
        std::string error = observer_->write(
            rhoFront, rhoRear, Descriptor::cs2());
        if (!error.empty()) {
            return {runtime::SingleLevelRunStatus::OutputFailure, error};
        }
        diagnostics_->writeFinalSummary();
        return {};
    }

    runtime::SingleLevelRunSessionResult writePerformance(
        std::optional<profiling::TimestepPerformanceRecord> const& record)
    {
        AMREX_ALWAYS_ASSERT(!record.has_value());
        return {};
    }

private:
    using Diagnostics = runtime::SingleLevelDiagnostics<
        Descriptor, ShortRunOutput>;

    ExactReference exact_reference() const noexcept
    {
        return {
            controls_.rhoLb,
            uniform_initialization(controls_.variant)
                ? controls_.uMeanLb : 0.0};
    }

    Controls controls_;
    config::ResolvedLbmConfig lbm_;
    amrex::Box domain_;
    std::unique_ptr<ShortRunOutput> output_;
    std::unique_ptr<Diagnostics> diagnostics_;
    std::unique_ptr<ForceObserver> observer_;
};

struct Preparation {
    Controls controls;

    template <class Descriptor>
    using RunSession = channel::RunSession<Descriptor>;

    template <class View>
    runtime::SingleLevelCasePreparationResult<
        RunSession<typename View::DescriptorType>>
    resolve(View const& view)
    {
        using Descriptor = typename View::DescriptorType;
        runtime::SingleLevelCasePreparationResult<RunSession<Descriptor>> result;
        if constexpr (Descriptor::d == 2) {
            if (view.mesh.nCell[2] != 1) {
                result.diagnostics.push_back({
                    "invalid_channel_cylinder_depth",
                    "mesh.n_cell.z",
                    "D2Q9 channel regression requires one z cell"});
            }
        } else if (view.mesh.nCell[2] < 2) {
            result.diagnostics.push_back({
                "invalid_channel_cylinder_depth",
                "mesh.n_cell.z",
                "D3Q27 channel regression requires at least two z cells"});
        }
        if (!result.diagnostics.empty()) return result;
        controls.maxStep = view.run.maxStep;
        result.session = RunSession<Descriptor>(std::move(controls));
        return result;
    }
};

inline boundary::DomainBoundarySpec inlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "inlet";
    spec.location = boundary::DomainSide::XLo;
    spec.physical = boundary::BoundaryPhysical::VelocityInlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Poiseuille;
    spec.profileAxis = 1;
    spec.hasProfileAxis = true;
    spec.uMeanLb = controls.uMeanLb;
    spec.hasUMeanLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec outlet(Controls const& controls)
{
    boundary::DomainBoundarySpec spec;
    spec.name = "outlet";
    spec.location = boundary::DomainSide::XHi;
    spec.physical = boundary::BoundaryPhysical::PressureOutlet;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Constant;
    spec.rhoLb = controls.rhoLb;
    spec.hasRhoLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec wall(boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec;
    spec.name = side == boundary::DomainSide::YLo
        ? "bottom_wall" : "top_wall";
    spec.location = side;
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    return spec;
}

inline boundary::InternalBoundarySpec cylinder(
    Controls const& controls,
    amrex::Real dx,
    bool rear)
{
    boundary::InternalBoundarySpec spec;
    spec.name = tandem(controls.variant)
        ? (rear ? "rear_cylinder_wall" : "front_cylinder_wall")
        : "cylinder_wall";
    spec.geometry = "cylinder";
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.hasPhysical = true;
    spec.scheme = controls.variant == Variant::SingleBouzidi
        ? boundary::BoundaryScheme::Bouzidi
        : boundary::BoundaryScheme::Yms;
    spec.profile = boundary::BoundaryProfileKind::None;
    spec.objectName = controls.variant == Variant::TandemShared
        ? "cylinder_pair"
        : (controls.variant == Variant::TandemDistinct
               ? (rear ? "rear_cylinder" : "front_cylinder")
               : "cylinder");
    spec.centerPhys = {
        rear ? 0.4 : 0.2,
        0.2,
        0.5 * dx * controls.depthCells};
    spec.hasCenterPhys = true;
    spec.radiusPhys = 0.05;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

template <class Descriptor>
auto make_program(Controls controls)
{
    using Profile = initialization::ConstantInitializationProfile;
    using Initialization =
        initialization::ProfileInitializationSelection<Profile>;
    using Program = runtime::SingleLevelProgram<
        Descriptor, BGKDynamicsSelection, Initialization, Preparation>;
    Program program;
    const amrex::Real dx = 0.1 / controls.diameterIntervals;
    const int xIntervals = 22 * controls.diameterIntervals;
    const int yIntervals = 41 * controls.diameterIntervals / 10;
    program.mesh.nCell = {
        xIntervals + 1, yIntervals + 1, controls.depthCells};
    program.mesh.envelopeLo = {-0.5 * dx, -0.5 * dx, 0.0};
    program.mesh.envelopeHi = {
        2.2 + 0.5 * dx,
        0.41 + 0.5 * dx,
        dx * controls.depthCells};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = controls.maxGridSize;
    program.lbm.parameterizationKind =
        config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = controls.rhoLb;
    program.lbm.uRefLb = controls.uMeanLb;
    program.lbm.uRefPhys = controls.uMeanPhys;
    program.lbm.lengthRefLb = yIntervals;
    program.lbm.lengthRefPhys = 0.41;
    program.lbm.reynolds = controls.reynoldsHeight;
    program.lbm.streamingBackend = std::string("baseline");
    program.run.maxStep = controls.maxStep;
    program.run.finalTimePhys = controls.finalTimePhys;
    program.run.kind = controls.runKind;
    program.run.name = "channel_cylinder";
    program.run.variant = controls.runVariant;
    program.initialization = initialization::from_profile(Profile{{
        controls.rhoLb,
        uniform_initialization(controls.variant) ? controls.uMeanLb : 0.0,
        0.0,
        0.0}});
    program.domainBoundaries = {
        inlet(controls), outlet(controls),
        wall(boundary::DomainSide::YLo),
        wall(boundary::DomainSide::YHi)};
    program.internalBoundaries = {cylinder(controls, dx, false)};
    if (tandem(controls.variant)) {
        program.internalBoundaries.push_back(cylinder(controls, dx, true));
    }
    program.casePreparation = Preparation{controls};
    return program;
}

} // namespace lbm::tests::regression::channel

#endif
