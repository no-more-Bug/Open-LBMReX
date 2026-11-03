#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRun.H"

#include <AMReX.H>
#include <AMReX_BLassert.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using ActiveDescriptor = lbm::D3Q27Descriptor;
#else
using ActiveDescriptor = lbm::D2Q9Descriptor;
#endif

using Program = lbm::runtime::SingleLevelProgram<ActiveDescriptor, lbm::BGKDynamicsSelection>;

template <class Descriptor>
class ThicknessRunSession : public lbm::runtime::NoCaseRunSession<Descriptor> {
public:
    ThicknessRunSession(amrex::Real height, int depth) : height_(height), depth_(depth) {}

    lbm::runtime::SingleLevelRunSessionResult start(
        lbm::runtime::SingleLevelRunContext<Descriptor> const& context)
    {
        auto const& geom = context.geometry;
        AMREX_ALWAYS_ASSERT(geom.Domain().length(2) == depth_ && geom.isPeriodic(2));
        AMREX_ALWAYS_ASSERT(geom.ProbLo(2) == 0.0 && geom.ProbHi(2) == height_);
        AMREX_ALWAYS_ASSERT(std::abs(geom.CellSize(0) - 0.125) < 1.0e-12);
        AMREX_ALWAYS_ASSERT(std::abs(geom.CellSize(1) - 0.125) < 1.0e-12);
        AMREX_ALWAYS_ASSERT(std::abs(geom.CellSize(2) - height_ / depth_) < 1.0e-12);
        AMREX_ALWAYS_ASSERT(std::abs(context.lbm.tau - 0.53) < 1.0e-12);
        AMREX_ALWAYS_ASSERT(context.lbm.lengthRefLb == 8.0 && context.lbm.lengthRefPhys == 1.0);
        return {};
    }

    lbm::runtime::SingleLevelObservationPlan observationPlan() const noexcept
    {
        lbm::runtime::SingleLevelObservationPlan plan;
        plan.observeFinal = true;
        return plan;
    }

    lbm::runtime::SingleLevelRunSessionResult observeFinal(
        lbm::runtime::SingleLevelPublishedState<Descriptor> const& state)
    {
        AMREX_ALWAYS_ASSERT(state.step() == 1 && state.latticeTime() == 1.0);
        AMREX_ALWAYS_ASSERT(std::abs(state.physicalTime() - 0.00125) < 1.0e-12);
        return {};
    }

private:
    amrex::Real height_;
    int depth_;
};

struct ThicknessPreparation {
    amrex::Real height;
    int depth;
    template <class Descriptor> using RunSession = ThicknessRunSession<Descriptor>;

    template <class View>
    lbm::runtime::SingleLevelCasePreparationResult<RunSession<typename View::DescriptorType>>
    resolve(View const&) const
    {
        return {RunSession<typename View::DescriptorType>(height, depth), {}};
    }
};

void print_diagnostics(std::vector<lbm::runtime::SingleLevelSetupDiagnostic> const& diagnostics)
{
    for (auto const& diagnostic : diagnostics) {
        amrex::Print() << "[" << lbm::runtime::to_string(diagnostic.stage) << ":" << diagnostic.code
                       << ":" << diagnostic.subject << "] " << diagnostic.message << "\n";
    }
}

void expect_success(Program program)
{
    auto result = lbm::runtime::prepare_single_level(std::move(program));
    if (!result.ok()) {
        print_diagnostics(result.diagnostics());
    }
    AMREX_ALWAYS_ASSERT(result.ok());
    auto prepared = std::move(result).take();
    static_cast<void>(prepared);
}

void expect_failure(Program program, lbm::runtime::SingleLevelSetupStage stage,
                    std::string const& code)
{
    Program repeatedProgram = program;
    auto result = lbm::runtime::prepare_single_level(std::move(program));
    auto repeated = lbm::runtime::prepare_single_level(std::move(repeatedProgram));
    AMREX_ALWAYS_ASSERT(!result.ok());
    AMREX_ALWAYS_ASSERT(!repeated.ok());
    AMREX_ALWAYS_ASSERT(!result.diagnostics().empty());
    if (result.diagnostics().front().stage != stage ||
        result.diagnostics().front().code != code) {
        amrex::Print() << "Expected preparation failure ["
                       << lbm::runtime::to_string(stage) << ':' << code
                       << "] but received:\n";
        print_diagnostics(result.diagnostics());
    }
    AMREX_ALWAYS_ASSERT(result.diagnostics().front().stage == stage);
    AMREX_ALWAYS_ASSERT(result.diagnostics().front().code == code);
    AMREX_ALWAYS_ASSERT(result.diagnostics().size() == repeated.diagnostics().size());
    for (std::size_t i = 0; i < result.diagnostics().size(); ++i) {
        auto const& lhs = result.diagnostics()[i];
        auto const& rhs = repeated.diagnostics()[i];
        AMREX_ALWAYS_ASSERT(lhs.stage == rhs.stage);
        AMREX_ALWAYS_ASSERT(lhs.code == rhs.code);
        AMREX_ALWAYS_ASSERT(lhs.subject == rhs.subject);
        AMREX_ALWAYS_ASSERT(lhs.message == rhs.message);
    }
}

Program make_direct_periodic_program()
{
    Program program;
    program.mesh.nCell = {8, 8, ActiveDescriptor::d == 3 ? 4 : 1};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {1.0, 1.0, ActiveDescriptor::d == 3 ? 0.5 : 0.125};
    program.mesh.isPeriodic = {1, 1, 1};
    program.mesh.maxGridSize = 4;
    program.lbm.parameterizationKind = lbm::config::LbmParameterizationKind::FromNuLb;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.01;
    program.lbm.uRefPhys = 1.0;
    program.lbm.nuLb = 0.01;
    program.run.maxStep = 1;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "focused_check";
    program.run.name = "direct_periodic";
    program.initialization.intent.kind = lbm::initialization::InitializationKind::TaylorGreen;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.taylorGreenVelocityAmplitudeLb = 0.01;
    return program;
}

Program make_direct_guo_cavity_program()
{
    Program program;
    constexpr amrex::Real halfDx = amrex::Real(1.0 / 256.0);
    program.mesh.nCell = {129, 129, 1};
    program.mesh.envelopeLo = {-halfDx, -halfDx, 0.0};
    program.mesh.envelopeHi = {amrex::Real(1.0) + halfDx, amrex::Real(1.0) + halfDx,
                               amrex::Real(1.0 / 128.0)};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = 32;
    program.lbm.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.1;
    program.lbm.uRefPhys = 1.0;
    program.lbm.lengthRefLb = 128.0;
    program.lbm.lengthRefPhys = 1.0;
    program.lbm.reynolds = 1000.0;
    program.run.maxStep = 1;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "focused_check";
    program.run.name = "direct_guo_cavity";
    program.initialization.intent.kind = lbm::initialization::InitializationKind::Rest;
    program.initialization.intent.rhoLb = 1.0;

    auto add_wall = [&program](char const* name, lbm::boundary::DomainSide side, bool moving) {
        lbm::boundary::DomainBoundarySpec wall;
        wall.name = name;
        wall.location = side;
        wall.physical = moving ? lbm::boundary::BoundaryPhysical::MovingWall
                               : lbm::boundary::BoundaryPhysical::NoSlipWall;
        wall.scheme = lbm::boundary::BoundaryScheme::GuoExtrapolation;
        if (moving) {
            wall.profile = lbm::boundary::BoundaryProfileKind::Uniform;
            wall.prescribedULb = {0.1, 0.0, 0.0};
            wall.hasPrescribedULb = true;
        }
        program.domainBoundaries.push_back(std::move(wall));
    };
    add_wall("left_wall", lbm::boundary::DomainSide::XLo, false);
    add_wall("right_wall", lbm::boundary::DomainSide::XHi, false);
    add_wall("bottom_wall", lbm::boundary::DomainSide::YLo, false);
    add_wall("moving_lid", lbm::boundary::DomainSide::YHi, true);
    return program;
}

void run_negative_checks(Program const& base)
{
    {
        Program invalid = base;
        invalid.mesh.nCell[0] = 0;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::Mesh,
                       "non_positive_cell_count");
    }
    {
        Program invalid = base;
        invalid.lbm.reynolds = -1.0;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::LbmClosure,
                       "invalid_lbm_intent");
    }
    {
        Program invalid = base;
        invalid.run.maxStep = 0;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::RunControl,
                       "invalid_run_control");
    }
    {
        Program invalid = base;
        invalid.initialization.intent.rhoLb = -1.0;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::Initialization,
                       "invalid_initialization");
    }
    if (!base.internalBoundaries.empty()) {
        Program invalid = base;
        invalid.internalBoundaries.front().radiusPhys = 10.0;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::PhysicalBoundary,
                       "invalid_physical_boundary");
    }
}

void run_thickness_preparation_checks()
{
    auto thick = make_direct_periodic_program();
    thick.mesh.envelopeHi[2] *= 2.0;
    if constexpr (ActiveDescriptor::d == 2) {
        expect_success(thick);

        auto nonperiodic = thick;
        nonperiodic.mesh.isPeriodic[2] = 0;
        expect_failure(std::move(nonperiodic), lbm::runtime::SingleLevelSetupStage::Mesh,
                       "invalid_d2q9_thickness_layout");

        auto multiplePlanes = thick;
        multiplePlanes.mesh.nCell[2] = 2;
        multiplePlanes.mesh.envelopeHi[2] *= 2.0;
        expect_failure(std::move(multiplePlanes), lbm::runtime::SingleLevelSetupStage::Mesh,
                       "invalid_d2q9_thickness_layout");

        auto legacy = make_direct_periodic_program();
        legacy.mesh.nCell[2] = 3;
        legacy.mesh.envelopeHi[2] *= 3.0;
        expect_success(std::move(legacy));

        auto cavity = make_direct_guo_cavity_program();
        cavity.mesh.envelopeHi[2] *= 2.0;
        expect_success(std::move(cavity));
    } else {
        expect_failure(thick, lbm::runtime::SingleLevelSetupStage::LbmClosure,
                       "invalid_lbm_intent");
    }

    for (amrex::Real height : {amrex::Real(0.0), amrex::Real(-0.1),
             std::numeric_limits<amrex::Real>::infinity(),
             std::numeric_limits<amrex::Real>::quiet_NaN()}) {
        auto invalid = thick;
        invalid.mesh.envelopeHi[2] = height;
        expect_failure(std::move(invalid), lbm::runtime::SingleLevelSetupStage::Mesh,
                       "invalid_envelope");
    }
    auto overflow = thick;
    overflow.mesh.envelopeLo[2] = -std::numeric_limits<amrex::Real>::max();
    overflow.mesh.envelopeHi[2] = std::numeric_limits<amrex::Real>::max();
    expect_failure(std::move(overflow), lbm::runtime::SingleLevelSetupStage::Mesh,
                   "invalid_envelope");

    auto wrongXY = make_direct_periodic_program();
    wrongXY.mesh.envelopeHi[1] *= 2.0;
    expect_failure(std::move(wrongXY), lbm::runtime::SingleLevelSetupStage::LbmClosure,
                   "invalid_lbm_intent");

    for (int scale : {1, 2}) {
        if (ActiveDescriptor::d == 3 && scale == 2) {
            continue;
        }
        auto base = make_direct_periodic_program();
        lbm::runtime::SingleLevelProgram<ActiveDescriptor, lbm::BGKDynamicsSelection,
            lbm::initialization::ConfiguredInitializationSelection, ThicknessPreparation> observed;
        observed.mesh = base.mesh;
        observed.mesh.envelopeHi[2] *= scale;
        observed.lbm = base.lbm;
        observed.run = base.run;
        observed.initialization = base.initialization;
        observed.casePreparation = {observed.mesh.envelopeHi[2], observed.mesh.nCell[2]};
        auto prepared = lbm::runtime::prepare_single_level(std::move(observed));
        if (!prepared.ok()) {
            print_diagnostics(prepared.diagnostics());
        }
        AMREX_ALWAYS_ASSERT(prepared.ok());
        auto report = lbm::runtime::run_single_level(std::move(prepared).take());
        AMREX_ALWAYS_ASSERT(report.ok() && report.finalStep == 1);
    }
    amrex::Print() << "Fixed-thickness preparation checks: PASS\n";
}

void assert_channel_input_facts(
    bool re20,
    int expectedDiameterIntervals)
{
    std::string variant;
    int diameterIntervals = 0;
    int depthCells = 0;
    amrex::Real uMeanPhys = 0.0;
    amrex::Real reynoldsHeight = 0.0;
    amrex::ParmParse channel("channel_cylinder");
    channel.get("variant", variant);
    channel.get("diameter_intervals", diameterIntervals);
    channel.get("depth_cells", depthCells);
    channel.get("u_mean_phys", uMeanPhys);
    channel.get("reynolds_height", reynoldsHeight);
    AMREX_ALWAYS_ASSERT(
        diameterIntervals == expectedDiameterIntervals && depthCells == 1 &&
        (re20 ? (variant == "re20_yms" &&
                 (expectedDiameterIntervals == 40 || expectedDiameterIntervals == 80) &&
                 reynoldsHeight == 82.0 && uMeanPhys == 0.2)
              : (variant == "re100_yms" && expectedDiameterIntervals == 80 &&
                 reynoldsHeight == 410.0 && uMeanPhys == 1.0)));
}

Program make_internal_cylinder_fixture_program()
{
    Program program;
    program.mesh.nCell = {24, 16, 1};
    program.mesh.envelopeLo = {0.0, 0.0, 0.0};
    program.mesh.envelopeHi = {1.2, 0.8, 0.05};
    program.mesh.isPeriodic = {0, 0, 1};
    program.mesh.maxGridSize = 16;
    program.lbm.parameterizationKind = lbm::config::LbmParameterizationKind::FromReynolds;
    program.lbm.rhoRefLb = 1.0;
    program.lbm.uRefLb = 0.05;
    program.lbm.uRefPhys = 1.0;
    program.lbm.lengthRefLb = 16.0;
    program.lbm.lengthRefPhys = 0.8;
    program.lbm.reynolds = 20.0;
    program.run.maxStep = 1;
    program.run.finalTimePhys = 1.0;
    program.run.kind = "focused_check";
    program.run.name = "internal_cylinder_fixture";
    program.initialization.intent.kind = lbm::initialization::InitializationKind::Uniform;
    program.initialization.intent.rhoLb = 1.0;
    program.initialization.intent.uniformULb =
        amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0};

    auto wall = [](char const* name, lbm::boundary::DomainSide side) {
        lbm::boundary::DomainBoundarySpec result;
        result.name = name;
        result.location = side;
        result.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
        result.scheme = lbm::boundary::BoundaryScheme::HalfwayBounceBack;
        return result;
    };
    program.domainBoundaries = {
        wall("xlo_wall", lbm::boundary::DomainSide::XLo),
        wall("xhi_wall", lbm::boundary::DomainSide::XHi),
        wall("ylo_wall", lbm::boundary::DomainSide::YLo),
        wall("yhi_wall", lbm::boundary::DomainSide::YHi)};

    lbm::boundary::InternalBoundarySpec cylinder;
    cylinder.name = "internal_cylinder";
    cylinder.geometry = "cylinder";
    cylinder.physical = lbm::boundary::BoundaryPhysical::NoSlipWall;
    cylinder.hasPhysical = true;
    cylinder.scheme = lbm::boundary::BoundaryScheme::Yms;
    cylinder.profile = lbm::boundary::BoundaryProfileKind::None;
    cylinder.objectName = "fixture";
    cylinder.centerPhys = {0.6, 0.4, 0.025};
    cylinder.hasCenterPhys = true;
    cylinder.radiusPhys = 0.2;
    cylinder.hasRadiusPhys = true;
    cylinder.axis = "z";
    cylinder.hasAxis = true;
    program.internalBoundaries.push_back(std::move(cylinder));
    return program;
}

bool run_direct_channel_preparation_if_requested()
{
    amrex::ParmParse integration("integration");
    std::string mode;
    if (!integration.query("mode", mode)) {
        return false;
    }
    const bool re20 = mode == "channel_re20_prepare";
    const bool re100 = mode == "channel_re100_prepare";
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(re20 || re100, "Unsupported direct channel preparation mode");
    int expectedDiameterIntervals = 0;
    integration.get("expected_diameter_intervals", expectedDiameterIntervals);

    assert_channel_input_facts(re20, expectedDiameterIntervals);
    expect_success(make_internal_cylinder_fixture_program());
    amrex::Print() << "Tests-owned internal-cylinder preparation: PASS N_D="
                   << expectedDiameterIntervals << '\n';
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    using Preparation = decltype(lbm::runtime::prepare_single_level(std::declval<Program>()));
    using PreparedRun = decltype(std::declval<Preparation&&>().take());
    static_assert(!std::is_copy_constructible_v<PreparedRun>);
    static_assert(std::is_move_constructible_v<PreparedRun>);

    amrex::Initialize(argc, argv);
    {
        if (run_direct_channel_preparation_if_requested()) {
            amrex::Finalize();
            return 0;
        }
        Program program = make_direct_periodic_program();
        run_negative_checks(program);
        run_thickness_preparation_checks();
        expect_success(std::move(program));
        if constexpr (ActiveDescriptor::d == 2) {
            expect_success(make_direct_guo_cavity_program());
            auto shortGuo = make_direct_guo_cavity_program();
            shortGuo.mesh.nCell = {2, 3, 1};
            shortGuo.mesh.envelopeLo = {-0.5, -0.5, 0.0};
            shortGuo.mesh.envelopeHi = {1.5, 2.5, 1.0};
            shortGuo.lbm.lengthRefLb = 1.0;
            shortGuo.lbm.lengthRefPhys = 1.0;
            expect_failure(std::move(shortGuo),
                           lbm::runtime::SingleLevelSetupStage::PhysicalBoundary,
                           "invalid_physical_boundary");
        }
        amrex::Print() << "Single-level preparation integration tests: PASS\n";
    }
    amrex::Finalize();
    return 0;
}
