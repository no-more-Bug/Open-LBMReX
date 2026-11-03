#ifndef LBM_TESTS_INTEGRATION_LBM_LBMDOMAININTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_LBMDOMAININTEGRATIONTESTS_H

#include "lbm/boundary/core/BoundaryDiagnostics.H"
#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/core/BoundarySystem.H"
#include "lbm/boundary/domain/BoundarySamplingPosition.H"
#include "lbm/boundary/domain/DomainBoundaryPrescription.H"
#include "lbm/boundary/domain/DomainBoundarySystem.H"
#include "lbm/boundary/profiles/BuiltinProfiles.H"
#include "lbm/boundary/schemes/GuoNonEquilibriumExtrapolationScheme.H"
#include "lbm/dynamics/BGKDynamics.H"
#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"
#include "lbm/fields/IntField.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/fields/MacroField.H"
#include "lbm/fields/RealField.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/kernels/StreamingKernels.H"
#include "lbm/operators/level/CollisionOperator.H"
#include "lbm/operators/level/EquilibriumPopulationInitializationOperator.H"
#include "lbm/operators/level/MacroscopicOperator.H"
#include "lbm/operators/level/PullStreamingOperator.H"
#include "lbm/solvers/BoundaryStepContext.H"
#include "integration/lbm/BoundarySystemTestSetup.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lbm::tests::integration {

inline void expect(bool condition, char const* message)
{
    if (!condition) {
        amrex::Abort(message);
    }
}

struct ConstantPopulationInitializationProfile {
    initialization::MacroscopicState state;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
        amrex::Real) const noexcept
    {
        return state;
    }
};

struct CoordinatePopulationInitializationProfile {
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    initialization::MacroscopicState operator()(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real time) const noexcept
    {
        return initialization::MacroscopicState{
            1.0 + AMREX_D_TERM(
                1.0e-3 * position[0],
                + 2.0e-3 * position[1],
                + 3.0e-3 * position[2]) +
                4.0e-3 * time,
            1.0e-2 + 1.0e-4 * position[0],
            -8.0e-3 + 1.0e-4 * position[1],
            0.0};
    }
};

template <class Descriptor>
amrex::Geometry make_lbm_integration_geometry(
    int periodicX,
    int periodicY,
    int periodicZ,
    int nx = 4,
    int ny = 4,
    int nzRequested = -1)
{
    const int nz = nzRequested > 0 ? nzRequested : (Descriptor::d == 3 ? 4 : 1);

    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);

    amrex::RealBox realBox(
        {AMREX_D_DECL(0.0, 0.0, 0.0)},
        {AMREX_D_DECL(
            0.25 * static_cast<amrex::Real>(nx),
            0.25 * static_cast<amrex::Real>(ny),
            0.25 * static_cast<amrex::Real>(nz))});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(periodicX, periodicY, periodicZ)};
    return amrex::Geometry(domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());
}

inline bool contains_error(
    boundary::BoundaryResolveResult const& result,
    std::string const& needle)
{
    for (std::string const& error : result.errors) {
        if (error.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

template <class Descriptor>
inline void run_poiseuille_prescribed_direction_integration_tests()
{
    using boundary::BoundaryPhysical;
    using boundary::DomainSide;

    struct DirectionCase {
        DomainSide side;
        BoundaryPhysical physical;
        int expectedSign;
    };

    constexpr std::array<DirectionCase, 4> cases{{
        {DomainSide::XLo, BoundaryPhysical::VelocityInlet, 1},
        {DomainSide::XHi, BoundaryPhysical::VelocityInlet, -1},
        {DomainSide::XLo, BoundaryPhysical::VelocityOutlet, -1},
        {DomainSide::XHi, BoundaryPhysical::VelocityOutlet, 1}}};
    constexpr amrex::Real peakSpeed = 0.08;
    const amrex::Geometry geometry =
        make_lbm_integration_geometry<Descriptor>(0, 0, 1);
    const int profileAxis = 1;
    const amrex::Real profileLo =
        geometry.ProbLo(profileAxis) +
        (static_cast<amrex::Real>(geometry.Domain().smallEnd(profileAxis)) + 0.5) *
            geometry.CellSize(profileAxis);
    const amrex::Real profileHi =
        geometry.ProbLo(profileAxis) +
        (static_cast<amrex::Real>(geometry.Domain().bigEnd(profileAxis)) + 0.5) *
            geometry.CellSize(profileAxis);
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> center{
        AMREX_D_DECL(0.0, 0.5 * (profileLo + profileHi), 0.0)};
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lowEndpoint{
        AMREX_D_DECL(0.0, profileLo, 0.0)};
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> highEndpoint{
        AMREX_D_DECL(0.0, profileHi, 0.0)};

    for (DirectionCase const& directionCase : cases) {
        boundary::ResolvedDomainBoundary resolved;
        resolved.side = directionCase.side;
        resolved.physical = directionCase.physical;
        resolved.scheme = boundary::BoundaryScheme::GuoExtrapolation;
        resolved.stage = boundary::BoundaryExecutionStage::PostStreamingValidNode;
        resolved.profile = boundary::BoundaryProfileKind::Poiseuille;
        resolved.profileAxis = profileAxis;
        resolved.uMaxLb = peakSpeed;
        const boundary::DomainFaceTable table =
            boundary::build_domain_face_table({resolved}, geometry);
        const boundary::DomainFaceTableView faces = table.view();
        const int sideIndex =
            boundary::domain_side_index(directionCase.side);
        const boundary::PoiseuilleVelocityProfile profile =
            boundary::make_poiseuille_velocity_profile(
                faces,
                sideIndex,
                directionCase.side,
                directionCase.physical);
        const auto velocity = profile(center, 2.0);
        const auto lowEndpointVelocity = profile(lowEndpoint, 2.0);
        const auto highEndpointVelocity = profile(highEndpoint, 2.0);
        const amrex::Real expected =
            static_cast<amrex::Real>(directionCase.expectedSign) * peakSpeed;
        const std::string message =
            "Poiseuille prescribed normal direction mismatch for side='" +
            std::string(boundary::to_string(directionCase.side)) +
            "', physical='" +
            std::string(boundary::to_string(directionCase.physical)) + "'";
        expect(
            std::abs(velocity[0] - expected) < 1.0e-14 &&
                std::abs(velocity[1]) < 1.0e-14 &&
                std::abs(velocity[2]) < 1.0e-14 &&
                std::abs(lowEndpointVelocity[0]) < 1.0e-14 &&
                std::abs(highEndpointVelocity[0]) < 1.0e-14,
            message.c_str());
    }
}

template <class Descriptor>
void run_boundary_resolver_integration_tests()
{
    run_poiseuille_prescribed_direction_integration_tests<Descriptor>();

    auto const geom = make_lbm_integration_geometry<Descriptor>(0, 0, 0);

    auto make_wall = [](
        std::string name,
        boundary::DomainSide side,
        boundary::BoundaryPhysical physical = boundary::BoundaryPhysical::NoSlipWall) {
        boundary::DomainBoundarySpec spec;
        spec.name = std::move(name);
        spec.location = side;
        spec.physical = physical;
        return spec;
    };

    std::vector<boundary::DomainBoundarySpec> walls{
        make_wall("left_wall", boundary::DomainSide::XLo),
        make_wall("right_wall", boundary::DomainSide::XHi),
        make_wall("bottom_wall", boundary::DomainSide::YLo),
        make_wall("top_wall", boundary::DomainSide::YHi)};
    if constexpr (Descriptor::d == 3) {
        walls.push_back(make_wall("front_wall", boundary::DomainSide::ZLo));
        walls.push_back(make_wall("back_wall", boundary::DomainSide::ZHi));
    }
    auto noSlip = boundary::resolve_domain_boundary_specs<Descriptor>(walls, geom);
    const std::string noSlipError = noSlip.error_message();
    expect(noSlip.ok(), noSlipError.c_str());
    expect(
        noSlip.resolved.domain.size() == static_cast<std::size_t>(2 * Descriptor::d),
        "No-slip domain wall resolve count mismatch");
    for (auto const& resolved : noSlip.resolved.domain) {
        expect(
            resolved.physical == boundary::BoundaryPhysical::NoSlipWall,
            "No-slip wall physical resolve mismatch");
        expect(
            resolved.scheme == boundary::BoundaryScheme::HalfwayBounceBack,
            "No-slip wall default scheme mismatch");
        expect(
            resolved.schemeOrigin == boundary::BoundarySchemeOrigin::Default,
            "No-slip wall scheme origin mismatch");
        expect(
            resolved.stage == boundary::BoundaryExecutionStage::PreStreamingPhysicalGhost,
            "No-slip wall execution stage mismatch");
        expect(
            resolved.profile == boundary::BoundaryProfileKind::None,
            "No-slip wall default profile mismatch");
    }

    boundary::DomainBoundarySpec lid =
        make_wall("moving_lid", boundary::DomainSide::YHi, boundary::BoundaryPhysical::MovingWall);
    lid.profile = boundary::BoundaryProfileKind::Uniform;
    lid.prescribedULb = amrex::Array<amrex::Real, 3>{0.1, 0.0, 0.0};
    lid.hasPrescribedULb = true;
    auto moving = boundary::resolve_domain_boundary_specs<Descriptor>({lid}, geom);
    const std::string movingError = moving.error_message();
    expect(moving.ok(), movingError.c_str());
    expect(moving.resolved.domain.size() == 1, "Moving wall resolve count mismatch");
    expect(
        moving.resolved.domain[0].profile == boundary::BoundaryProfileKind::Uniform,
        "Moving wall profile resolve mismatch");
    expect(
        moving.resolved.domain[0].scheme ==
                boundary::BoundaryScheme::HalfwayBounceBack &&
            moving.resolved.domain[0].stage ==
                boundary::BoundaryExecutionStage::PreStreamingPhysicalGhost,
        "Moving wall default must remain pre-stream HBB");
    expect(
        std::abs(moving.resolved.domain[0].prescribedULb[0] - 0.1) < 1.0e-14,
        "Moving wall velocity resolve mismatch");

    boundary::DomainBoundarySpec guoLid = lid;
    guoLid.name = "guo_moving_lid";
    guoLid.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    auto guoMoving =
        boundary::resolve_domain_boundary_specs<Descriptor>({guoLid}, geom);
    expect(guoMoving.ok(), guoMoving.error_message().c_str());
    expect(
        guoMoving.resolved.domain[0].scheme ==
                boundary::BoundaryScheme::GuoExtrapolation &&
            guoMoving.resolved.domain[0].stage ==
                boundary::BoundaryExecutionStage::PostStreamingValidNode,
        "Explicit Guo moving wall must resolve to the post-stream valid-node stage");

    boundary::DomainBoundarySpec nonFiniteGuoLid = guoLid;
    nonFiniteGuoLid.name = "non_finite_guo_moving_lid";
    nonFiniteGuoLid.prescribedULb[0] =
        std::numeric_limits<amrex::Real>::infinity();
    auto nonFiniteMoving = boundary::resolve_domain_boundary_specs<Descriptor>(
        {nonFiniteGuoLid}, geom);
    expect(!nonFiniteMoving.ok(), "Non-finite Guo moving-wall velocity must fail");
    expect(
        contains_error(nonFiniteMoving, "finite u_lb components"),
        "Non-finite Guo moving-wall error mismatch");

    if constexpr (Descriptor::d == 2) {
        boundary::DomainBoundarySpec unsupportedMovingZ = guoLid;
        unsupportedMovingZ.name = "unsupported_guo_moving_lid_z";
        unsupportedMovingZ.prescribedULb[2] = 0.01;
        auto unsupportedMoving =
            boundary::resolve_domain_boundary_specs<Descriptor>(
                {unsupportedMovingZ}, geom);
        expect(
            !unsupportedMoving.ok() &&
                contains_error(
                    unsupportedMoving,
                    "does not support moving_wall u_lb component"),
            "D2 moving_wall must reject a non-zero z velocity");
    }

    boundary::DomainBoundarySpec normalLid = lid;
    normalLid.name = "normal_lid";
    normalLid.prescribedULb =
        amrex::Array<amrex::Real, 3>{0.0, 0.02, 0.0};
    auto normalMoving = boundary::resolve_domain_boundary_specs<Descriptor>({normalLid}, geom);
    const std::string normalMovingError = normalMoving.error_message();
    expect(normalMoving.ok(), normalMovingError.c_str());
    expect(
        normalMoving.resolved.warnings.size() == 1,
        "Moving wall normal velocity warning mismatch");

    auto periodicGeom = make_lbm_integration_geometry<Descriptor>(1, 0, 0);
    auto periodicConflict =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {make_wall("left_wall", boundary::DomainSide::XLo)}, periodicGeom);
    expect(!periodicConflict.ok(), "Periodic domain boundary conflict must fail");
    expect(
        contains_error(periodicConflict, "periodic direction"),
        "Periodic domain boundary conflict error mismatch");

    auto duplicateName =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {make_wall("dup", boundary::DomainSide::XLo),
             make_wall("dup", boundary::DomainSide::XHi)}, geom);
    expect(!duplicateName.ok(), "Duplicate boundary names must fail");
    expect(contains_error(duplicateName, "duplicate boundary name"), "Duplicate name error mismatch");

    auto duplicateLocation =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {make_wall("wall_a", boundary::DomainSide::XLo),
             make_wall("wall_b", boundary::DomainSide::XLo)}, geom);
    expect(!duplicateLocation.ok(), "Duplicate domain locations must fail");
    expect(
        contains_error(duplicateLocation, "duplicate domain location"),
        "Duplicate location error mismatch");

    boundary::DomainBoundarySpec inlet =
        make_wall("inlet", boundary::DomainSide::XLo, boundary::BoundaryPhysical::VelocityInlet);
    inlet.profile = boundary::BoundaryProfileKind::Uniform;
    inlet.prescribedULb = amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0};
    inlet.hasPrescribedULb = true;
    auto velocityInlet = boundary::resolve_domain_boundary_specs<Descriptor>({inlet}, geom);
    const std::string velocityInletError = velocityInlet.error_message();
    expect(velocityInlet.ok(), velocityInletError.c_str());
    expect(
        velocityInlet.resolved.domain[0].scheme == boundary::BoundaryScheme::HalfwayBounceBack,
        "Velocity inlet default scheme mismatch");
    expect(
        velocityInlet.resolved.domain[0].schemeOrigin == boundary::BoundarySchemeOrigin::Default,
        "Velocity inlet default scheme origin mismatch");

    boundary::DomainBoundarySpec nonFiniteVelocityInlet = inlet;
    nonFiniteVelocityInlet.name = "non_finite_velocity_inlet";
    nonFiniteVelocityInlet.prescribedULb[0] =
        std::numeric_limits<amrex::Real>::infinity();
    auto nonFiniteVelocity = boundary::resolve_domain_boundary_specs<Descriptor>(
        {nonFiniteVelocityInlet}, geom);
    expect(
        !nonFiniteVelocity.ok() &&
            contains_error(nonFiniteVelocity, "finite u_lb components"),
        "Velocity inlet must reject non-finite u_lb components");

    if constexpr (Descriptor::d == 2) {
        boundary::DomainBoundarySpec unsupportedVelocityZ = inlet;
        unsupportedVelocityZ.name = "unsupported_velocity_z";
        unsupportedVelocityZ.prescribedULb[2] = 0.01;
        auto unsupportedVelocity =
            boundary::resolve_domain_boundary_specs<Descriptor>(
                {unsupportedVelocityZ}, geom);
        expect(
            !unsupportedVelocity.ok() &&
                contains_error(
                    unsupportedVelocity,
                    "does not support velocity u_lb component"),
            "D2 velocity boundary must reject a non-zero z component");
    }

    boundary::DomainBoundarySpec guoVelocityInlet = inlet;
    guoVelocityInlet.name = "guo_velocity_inlet";
    guoVelocityInlet.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    auto resolvedGuoVelocity =
        boundary::resolve_domain_boundary_specs<Descriptor>({guoVelocityInlet}, geom);
    expect(resolvedGuoVelocity.ok(), resolvedGuoVelocity.error_message().c_str());
    expect(
        resolvedGuoVelocity.resolved.domain[0].stage ==
            boundary::BoundaryExecutionStage::PostStreamingValidNode,
        "Guo velocity execution stage mismatch");

    boundary::DomainBoundarySpec velocityOutlet =
        make_wall("outlet", boundary::DomainSide::XHi, boundary::BoundaryPhysical::VelocityOutlet);
    velocityOutlet.profile = boundary::BoundaryProfileKind::Uniform;
    velocityOutlet.prescribedULb =
        amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0};
    velocityOutlet.hasPrescribedULb = true;
    auto resolvedVelocityOutlet =
        boundary::resolve_domain_boundary_specs<Descriptor>({velocityOutlet}, geom);
    const std::string velocityOutletError = resolvedVelocityOutlet.error_message();
    expect(resolvedVelocityOutlet.ok(), velocityOutletError.c_str());
    expect(
        resolvedVelocityOutlet.resolved.domain[0].scheme == boundary::BoundaryScheme::HalfwayBounceBack,
        "Velocity outlet default scheme mismatch");

    boundary::DomainBoundarySpec poiseuilleInlet = inlet;
    poiseuilleInlet.name = "poiseuille_inlet";
    poiseuilleInlet.profile = boundary::BoundaryProfileKind::Poiseuille;
    poiseuilleInlet.hasPrescribedULb = false;
    poiseuilleInlet.profileAxis = 1;
    poiseuilleInlet.hasProfileAxis = true;
    poiseuilleInlet.uMaxLb = 0.1;
    poiseuilleInlet.hasUMaxLb = true;
    auto resolvedPoiseuilleVelocity =
        boundary::resolve_domain_boundary_specs<Descriptor>({poiseuilleInlet}, geom);
    const std::string poiseuilleVelocityError = resolvedPoiseuilleVelocity.error_message();
    expect(resolvedPoiseuilleVelocity.ok(), poiseuilleVelocityError.c_str());
    expect(
        resolvedPoiseuilleVelocity.resolved.domain[0].uMaxLb == 0.1,
        "Poiseuille velocity u_max resolve mismatch");
    expect(
        resolvedPoiseuilleVelocity.resolved.domain[0].profileAxis == 1,
        "Poiseuille velocity profile_axis resolve mismatch");

    boundary::DomainBoundarySpec negativePoiseuilleAxis = poiseuilleInlet;
    negativePoiseuilleAxis.name = "negative_poiseuille_axis";
    negativePoiseuilleAxis.profileAxis = -1;
    auto negativePoiseuilleAxisResolve =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {negativePoiseuilleAxis}, geom);
    expect(
        !negativePoiseuilleAxisResolve.ok() &&
            contains_error(
                negativePoiseuilleAxisResolve,
                "does not support poiseuille profile_axis=-1"),
        "Negative Poiseuille profile_axis must fail before geometry compilation");

    boundary::DomainBoundarySpec outOfRangePoiseuilleAxis = poiseuilleInlet;
    outOfRangePoiseuilleAxis.name = "out_of_range_poiseuille_axis";
    outOfRangePoiseuilleAxis.profileAxis = Descriptor::d;
    auto outOfRangePoiseuilleAxisResolve =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {outOfRangePoiseuilleAxis}, geom);
    const std::string outOfRangeAxisError =
        "does not support poiseuille profile_axis=" +
        std::to_string(Descriptor::d);
    expect(
        !outOfRangePoiseuilleAxisResolve.ok() &&
            contains_error(
                outOfRangePoiseuilleAxisResolve, outOfRangeAxisError),
        "Out-of-range Poiseuille profile_axis must fail before geometry compilation");

    boundary::DomainBoundarySpec infinitePoiseuilleScale = poiseuilleInlet;
    infinitePoiseuilleScale.name = "infinite_poiseuille_scale";
    infinitePoiseuilleScale.uMaxLb =
        std::numeric_limits<amrex::Real>::infinity();
    auto infinitePoiseuilleScaleResolve =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {infinitePoiseuilleScale}, geom);
    expect(
        !infinitePoiseuilleScaleResolve.ok() &&
            contains_error(
                infinitePoiseuilleScaleResolve,
                "velocity scale must be finite and positive"),
        "Poiseuille velocity scale must reject positive infinity");

    boundary::DomainBoundarySpec zeroHeightPoiseuille = poiseuilleInlet;
    zeroHeightPoiseuille.name = "zero_height_guo_poiseuille";
    zeroHeightPoiseuille.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    auto oneCellProfileGeometry =
        make_lbm_integration_geometry<Descriptor>(0, 1, 1, 4, 1);
    auto zeroHeightPoiseuilleResolve =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {zeroHeightPoiseuille}, oneCellProfileGeometry);
    expect(
        !zeroHeightPoiseuilleResolve.ok() &&
            contains_error(
                zeroHeightPoiseuilleResolve,
                "requires at least 2 valid nodes"),
        "Guo Poiseuille must reject a zero-height tangential interval");

    boundary::DomainBoundarySpec invalidPoiseuilleAxis = poiseuilleInlet;
    invalidPoiseuilleAxis.name = "invalid_poiseuille_axis";
    invalidPoiseuilleAxis.profileAxis = 0;
    auto invalidPoiseuilleAxisResolve =
        boundary::resolve_domain_boundary_specs<Descriptor>({invalidPoiseuilleAxis}, geom);
    expect(
        !invalidPoiseuilleAxisResolve.ok(),
        "Poiseuille profile_axis equal to face normal must fail");
    expect(
        contains_error(invalidPoiseuilleAxisResolve, "profile_axis must be tangential"),
        "Poiseuille profile_axis tangential error mismatch");

    boundary::DomainBoundarySpec pressureInlet =
        make_wall("pressure_inlet", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::PressureInlet);
    pressureInlet.profile = boundary::BoundaryProfileKind::Constant;
    pressureInlet.rhoLb = 1.001;
    pressureInlet.hasRhoLb = true;
    pressureInlet.scheme = boundary::BoundaryScheme::AntiBounceBack;
    auto unsupportedPressureInlet =
        boundary::resolve_domain_boundary_specs<Descriptor>({pressureInlet}, geom);
    expect(!unsupportedPressureInlet.ok(), "Pressure inlet must fail until Stage 4.4-C");
    expect(
        contains_error(unsupportedPressureInlet, "scheme='anti_bounce_back'"),
        "Pressure inlet staged anti-bounce-back error mismatch");

    pressureInlet.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    auto guoPressureInlet =
        boundary::resolve_domain_boundary_specs<Descriptor>({pressureInlet}, geom);
    expect(guoPressureInlet.ok(), guoPressureInlet.error_message().c_str());
    expect(
        guoPressureInlet.resolved.domain[0].stage ==
            boundary::BoundaryExecutionStage::PostStreamingValidNode,
        "Guo pressure execution stage mismatch");
    expect(
        guoPressureInlet.resolved.domain[0].hasRhoLb &&
            guoPressureInlet.resolved.domain[0].rhoLb == pressureInlet.rhoLb,
        "Guo pressure density metadata mismatch");
    const std::string guoPressureDiagnostics =
        boundary::format_resolved_boundary_table(
            guoPressureInlet.resolved, geom);
    expect(
        guoPressureDiagnostics.find("constant density rho_lb=1.001") !=
            std::string::npos,
        "Resolved pressure diagnostics must report rho_lb");

    boundary::DomainBoundarySpec prescribedState =
        make_wall(
            "prescribed_state",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::PrescribedState);
    prescribedState.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    prescribedState.profile = boundary::BoundaryProfileKind::Poiseuille;
    prescribedState.profileAxis = 1;
    prescribedState.hasProfileAxis = true;
    prescribedState.rhoLb = 1.02;
    prescribedState.hasRhoLb = true;
    prescribedState.prescribedULb =
        amrex::Array<amrex::Real, 3>{0.08, 0.0, 0.0};
    prescribedState.hasPrescribedULb = true;
    auto resolvedPrescribedState =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {prescribedState}, geom);
    expect(
        resolvedPrescribedState.ok(),
        resolvedPrescribedState.error_message().c_str());
    expect(
        resolvedPrescribedState.resolved.domain[0].physical ==
                boundary::BoundaryPhysical::PrescribedState &&
            resolvedPrescribedState.resolved.domain[0].stage ==
                boundary::BoundaryExecutionStage::PostStreamingValidNode &&
            resolvedPrescribedState.resolved.domain[0].hasRhoLb &&
            resolvedPrescribedState.resolved.domain[0].hasUMaxLb &&
            std::abs(
                resolvedPrescribedState.resolved.domain[0].uMaxLb - 0.08) <
                1.0e-14,
        "Guo prescribed-state metadata mismatch");
    const std::string prescribedStateDiagnostics =
        boundary::format_resolved_boundary_table(
            resolvedPrescribedState.resolved, geom);
    expect(
        prescribedStateDiagnostics.find("prescribed_state") !=
                std::string::npos &&
            prescribedStateDiagnostics.find("constant density rho_lb=1.02") !=
                std::string::npos,
        "Resolved prescribed-state diagnostics must report physical type and rho_lb");

    boundary::DomainBoundarySpec missingPrescribedRho = prescribedState;
    missingPrescribedRho.name = "missing_prescribed_rho";
    missingPrescribedRho.hasRhoLb = false;
    auto missingPrescribedRhoResult =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {missingPrescribedRho}, geom);
    expect(
        !missingPrescribedRhoResult.ok() &&
            contains_error(missingPrescribedRhoResult, "requires rho_lb"),
        "prescribed_state without rho_lb must fail resolve");

    boundary::DomainBoundarySpec missingPrescribedVelocity = prescribedState;
    missingPrescribedVelocity.name = "missing_prescribed_velocity";
    missingPrescribedVelocity.hasPrescribedULb = false;
    auto missingPrescribedVelocityResult =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {missingPrescribedVelocity}, geom);
    expect(
        !missingPrescribedVelocityResult.ok() &&
            contains_error(missingPrescribedVelocityResult, "requires u_lb"),
        "prescribed_state without u_lb must fail resolve");

    boundary::DomainBoundarySpec ambiguousPrescribedVelocity = prescribedState;
    ambiguousPrescribedVelocity.name = "ambiguous_prescribed_velocity";
    ambiguousPrescribedVelocity.uMaxLb = 0.08;
    ambiguousPrescribedVelocity.hasUMaxLb = true;
    auto ambiguousPrescribedVelocityResult =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {ambiguousPrescribedVelocity}, geom);
    expect(
        !ambiguousPrescribedVelocityResult.ok() &&
            contains_error(ambiguousPrescribedVelocityResult, "omit u_max_lb"),
        "prescribed_state with both u_lb and u_max_lb must fail resolve");

    if constexpr (Descriptor::d == 2) {
        boundary::DomainBoundarySpec unsupportedPrescribedZ = prescribedState;
        unsupportedPrescribedZ.name = "unsupported_prescribed_z";
        unsupportedPrescribedZ.prescribedULb[2] = 0.01;
        auto unsupportedPrescribedZResult =
            boundary::resolve_domain_boundary_specs<Descriptor>(
                {unsupportedPrescribedZ}, geom);
        expect(
            !unsupportedPrescribedZResult.ok() &&
                contains_error(
                    unsupportedPrescribedZResult,
                    "does not support prescribed u_lb component"),
            "D2 prescribed_state must reject a non-zero z velocity");
    }

    boundary::DomainBoundarySpec invalidPressureDensity = pressureInlet;
    invalidPressureDensity.name = "invalid_pressure_density";
    invalidPressureDensity.rhoLb = 0.0;
    auto zeroPressureDensity =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {invalidPressureDensity}, geom);
    expect(
        !zeroPressureDensity.ok() &&
            contains_error(zeroPressureDensity, "finite positive rho_lb"),
        "Zero pressure density must fail resolve");
    invalidPressureDensity.rhoLb =
        std::numeric_limits<amrex::Real>::infinity();
    auto infinitePressureDensity =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            {invalidPressureDensity}, geom);
    expect(
        !infinitePressureDensity.ok() &&
            contains_error(infinitePressureDensity, "finite positive rho_lb"),
        "Infinite pressure density must fail resolve");

    boundary::DomainBoundarySpec freeOutflow =
        make_wall("free_outflow", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::FreeOutflow);
    auto resolvedFreeOutflow =
        boundary::resolve_domain_boundary_specs<Descriptor>({freeOutflow}, geom);
    expect(resolvedFreeOutflow.ok(), resolvedFreeOutflow.error_message().c_str());
    expect(
        resolvedFreeOutflow.resolved.domain[0].scheme ==
            boundary::BoundaryScheme::GuoExtrapolation &&
        resolvedFreeOutflow.resolved.domain[0].schemeOrigin ==
            boundary::BoundarySchemeOrigin::Default,
        "Free outflow default Guo scheme mismatch");

    boundary::DomainBoundarySpec guoNoSlip =
        make_wall("guo_wall", boundary::DomainSide::YLo);
    guoNoSlip.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    auto resolvedGuoNoSlip =
        boundary::resolve_domain_boundary_specs<Descriptor>({guoNoSlip}, geom);
    expect(resolvedGuoNoSlip.ok(), resolvedGuoNoSlip.error_message().c_str());

    auto mixedStageIntersection = boundary::resolve_domain_boundary_specs<Descriptor>(
        {guoVelocityInlet, make_wall("pre_stream_wall", boundary::DomainSide::YLo)},
        geom);
    expect(!mixedStageIntersection.ok(), "Mixed-stage domain corner must fail");
    expect(
        contains_error(mixedStageIntersection, "cannot intersect stage"),
        "Mixed-stage domain corner error mismatch");

    boundary::DomainBoundarySpec guoYVelocity = guoVelocityInlet;
    guoYVelocity.name = "guo_y_velocity";
    guoYVelocity.location = boundary::DomainSide::YLo;
    guoYVelocity.prescribedULb =
        amrex::Array<amrex::Real, 3>{0.0, 0.05, 0.0};
    auto ambiguousGuoCorner = boundary::resolve_domain_boundary_specs<Descriptor>(
        {guoVelocityInlet, guoYVelocity}, geom);
    expect(!ambiguousGuoCorner.ok(), "Non-wall Guo corner must fail");
    expect(
        contains_error(ambiguousGuoCorner, "requires wall-wins"),
        "Non-wall Guo corner error mismatch");

    const std::string guoDiagnostics = boundary::format_resolved_boundary_table(
        resolvedGuoVelocity.resolved, geom);
    expect(
        guoDiagnostics.find("post_stream_valid") != std::string::npos &&
            guoDiagnostics.find("geometry=valid_node_center") != std::string::npos,
        "Resolved Guo diagnostics must report stage and valid-node geometry");

    boundary::DomainBoundarySpec unsupportedScheme =
        make_wall("bouzidi_wall", boundary::DomainSide::XLo);
    unsupportedScheme.scheme = boundary::BoundaryScheme::Bouzidi;
    auto explicitUnsupported =
        boundary::resolve_domain_boundary_specs<Descriptor>({unsupportedScheme}, geom);
    expect(!explicitUnsupported.ok(), "Explicit unsupported domain scheme must fail");
    expect(
        contains_error(explicitUnsupported, "scheme='bouzidi'"),
        "Explicit unsupported domain scheme error mismatch");

    auto zFace = boundary::resolve_domain_boundary_specs<Descriptor>(
        {make_wall("zlo_wall", boundary::DomainSide::ZLo)}, geom);
    if constexpr (Descriptor::d == 2) {
        expect(!zFace.ok(), "D2Q9 must reject z domain faces");
        expect(contains_error(zFace, "does not support domain location"), "D2Q9 z-face error mismatch");
    } else {
        const std::string zFaceError = zFace.error_message();
        expect(zFace.ok(), zFaceError.c_str());
    }

    boundary::BoundarySystem<Descriptor> system;
    amrex::BoxArray ba(geom.Domain());
    amrex::DistributionMapping dm(ba);
    define_boundary_system_for_test(system, walls, geom, ba, dm);
    expect(
        system.resolved().domain.size() == static_cast<std::size_t>(2 * Descriptor::d),
        "BoundarySystem initialize mismatch");

    amrex::Print() << Descriptor::name() << " boundary resolver integration tests passed.\n";
}

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real boundary_fill_pattern(int i, int j, int k, int iPop) noexcept
{
    return 100000.0 * static_cast<amrex::Real>(iPop) +
           1000.0 * static_cast<amrex::Real>(i) +
           10.0 * static_cast<amrex::Real>(j) +
           static_cast<amrex::Real>(k);
}

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real boundary_fill_pattern_density(int i, int j, int k) noexcept
{
    amrex::Real rho = 0.0;
    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
        rho += boundary_fill_pattern<Descriptor>(i, j, k, iPop);
    }
    return rho;
}

template <class Descriptor>
void initialize_boundary_fill_pattern(LatticeField<Descriptor>& populations)
{
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& grown = mfi.fabbox();
        amrex::Box const& valid = mfi.validbox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(grown, Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) {
                if (valid.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                    f(i, j, k, iPop) =
                        boundary_fill_pattern<Descriptor>(i, j, k, iPop);
                } else {
                    f(i, j, k, iPop) = -999999.0 - static_cast<amrex::Real>(iPop);
                }
            });
    }
    amrex::Gpu::streamSynchronize();
}

inline boundary::DomainBoundarySpec make_domain_wall_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical = boundary::BoundaryPhysical::NoSlipWall)
{
    boundary::DomainBoundarySpec spec;
    spec.name = std::move(name);
    spec.location = side;
    spec.physical = physical;
    return spec;
}

inline boundary::DomainBoundarySpec make_uniform_velocity_domain_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical,
    amrex::Array<amrex::Real, 3> velocity)
{
    boundary::DomainBoundarySpec spec =
        make_domain_wall_spec(std::move(name), side, physical);
    spec.profile = boundary::BoundaryProfileKind::Uniform;
    spec.prescribedULb = velocity;
    spec.hasPrescribedULb = true;
    return spec;
}

inline boundary::DomainBoundarySpec make_poiseuille_velocity_domain_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical,
    int profileAxis,
    amrex::Real uMaxLb)
{
    boundary::DomainBoundarySpec spec =
        make_domain_wall_spec(std::move(name), side, physical);
    spec.profile = boundary::BoundaryProfileKind::Poiseuille;
    spec.profileAxis = profileAxis;
    spec.hasProfileAxis = true;
    spec.uMaxLb = uMaxLb;
    spec.hasUMaxLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec make_guo_no_slip_domain_spec(
    std::string name,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec =
        make_domain_wall_spec(std::move(name), side);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    return spec;
}

inline boundary::DomainBoundarySpec make_guo_uniform_velocity_domain_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical,
    amrex::Array<amrex::Real, 3> velocity)
{
    boundary::DomainBoundarySpec spec = make_uniform_velocity_domain_spec(
        std::move(name), side, physical, velocity);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    return spec;
}

inline boundary::DomainBoundarySpec make_guo_moving_wall_domain_spec(
    std::string name,
    boundary::DomainSide side,
    amrex::Array<amrex::Real, 3> velocity)
{
    return make_guo_uniform_velocity_domain_spec(
        std::move(name), side, boundary::BoundaryPhysical::MovingWall,
        velocity);
}

inline boundary::DomainBoundarySpec make_guo_poiseuille_velocity_domain_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical,
    int profileAxis,
    amrex::Real uMaxLb)
{
    boundary::DomainBoundarySpec spec = make_poiseuille_velocity_domain_spec(
        std::move(name), side, physical, profileAxis, uMaxLb);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    return spec;
}

inline boundary::DomainBoundarySpec make_guo_pressure_domain_spec(
    std::string name,
    boundary::DomainSide side,
    boundary::BoundaryPhysical physical,
    amrex::Real rhoLb)
{
    boundary::DomainBoundarySpec spec =
        make_domain_wall_spec(std::move(name), side, physical);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::Constant;
    spec.rhoLb = rhoLb;
    spec.hasRhoLb = true;
    return spec;
}

inline boundary::DomainBoundarySpec make_guo_prescribed_state_domain_spec(
    std::string name,
    boundary::DomainSide side,
    amrex::Real rhoLb,
    amrex::Array<amrex::Real, 3> velocity,
    boundary::BoundaryProfileKind profile = boundary::BoundaryProfileKind::Uniform,
    int profileAxis = -1)
{
    boundary::DomainBoundarySpec spec = make_domain_wall_spec(
        std::move(name), side, boundary::BoundaryPhysical::PrescribedState);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = profile;
    spec.rhoLb = rhoLb;
    spec.hasRhoLb = true;
    spec.prescribedULb = velocity;
    spec.hasPrescribedULb = true;
    if (profile == boundary::BoundaryProfileKind::Poiseuille) {
        spec.profileAxis = profileAxis;
        spec.hasProfileAxis = true;
    }
    return spec;
}

template <class Descriptor>
boundary::DomainBoundarySystem<Descriptor> make_domain_boundary_system(
    amrex::Geometry const& geom,
    std::vector<boundary::DomainBoundarySpec> const& specs)
{
    auto resolved = boundary::resolve_domain_boundary_specs<Descriptor>(specs, geom);
    const std::string error = resolved.error_message();
    expect(resolved.ok(), error.c_str());
    boundary::DomainBoundarySystem<Descriptor> system;
    std::string defineError;
    expect(
        system.tryDefine(geom, resolved.resolved.domain, defineError),
        defineError.c_str());
    return system;
}

template <class Descriptor>
void run_post_stream_boundary_geometry_integration_tests()
{
    auto const periodicZGeom =
        make_lbm_integration_geometry<Descriptor>(0, 0, 1);
    std::vector<boundary::DomainBoundarySpec> xySpecs{
        make_guo_uniform_velocity_domain_spec(
            "guo_xlo", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0}),
        make_guo_uniform_velocity_domain_spec(
            "guo_xhi", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0}),
        make_guo_no_slip_domain_spec("guo_ylo", boundary::DomainSide::YLo),
        make_guo_no_slip_domain_spec("guo_yhi", boundary::DomainSide::YHi)};

    auto xyResolved =
        boundary::resolve_domain_boundary_specs<Descriptor>(xySpecs, periodicZGeom);
    expect(xyResolved.ok(), xyResolved.error_message().c_str());
    auto xyPlanResult = boundary::compile_post_streaming_boundary_plan<Descriptor>(
        periodicZGeom, xyResolved.resolved.domain);
    expect(xyPlanResult.ok(), xyPlanResult.error_message().c_str());
    boundary::PostStreamingBoundaryPlan const& xyPlan = xyPlanResult.plan;
    expect(xyPlan.hasPostStreamingFaces, "Guo plan must report post-stream faces");
    expect(!xyPlan.hasPreStreamingFaces, "All-Guo plan must not report pre-stream faces");
    expect(xyPlan.regions.size() == 8, "Four-face Guo plan must contain 4 faces and 4 corners");
    expect(
        xyPlan.axisGeometry[0].validNodes == 4 &&
            xyPlan.axisGeometry[0].latticeIntervals == 3,
        "Guo N intervals / N+1 valid nodes geometry mismatch");
    expect(
        boundary::matching_on_lattice_reference_axis(xyPlan, 3.0, 0.75) == 0,
        "Guo physical interval reference lengths must match an active axis");
    expect(
        boundary::matching_on_lattice_reference_axis(xyPlan, 4.0, 1.0) < 0,
        "Guo reference lengths must reject the N+1-cell AMReX envelope");

    {
        constexpr int validNodes = 129;
        constexpr int physicalIntervals = 128;
        constexpr amrex::Real physicalLength = 1.0;
        constexpr amrex::Real dx = physicalLength / physicalIntervals;
        const int nz = Descriptor::d == 3 ? 4 : 1;
        const amrex::Box cavityDomain(
            amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
            amrex::IntVect(AMREX_D_DECL(
                validNodes - 1, validNodes - 1, nz - 1)));
        const amrex::RealBox cavityEnvelope(
            {AMREX_D_DECL(-0.5 * dx, -0.5 * dx, 0.0)},
            {AMREX_D_DECL(
                physicalLength + 0.5 * dx,
                physicalLength + 0.5 * dx,
                static_cast<amrex::Real>(nz) * dx)});
        const amrex::Array<int, AMREX_SPACEDIM> periodic{
            AMREX_D_DECL(0, 0, 1)};
        const amrex::Geometry cavityGeometry(
            cavityDomain,
            &cavityEnvelope,
            amrex::CoordSys::cartesian,
            periodic.data());
        const auto cavityResolved =
            boundary::resolve_domain_boundary_specs<Descriptor>(
                xySpecs, cavityGeometry);
        expect(cavityResolved.ok(), cavityResolved.error_message().c_str());
        const auto cavityPlan =
            boundary::compile_post_streaming_boundary_plan<Descriptor>(
                cavityGeometry, cavityResolved.resolved.domain);
        expect(cavityPlan.ok(), cavityPlan.error_message().c_str());
        for (int axis = 0; axis < 2; ++axis) {
            const auto& geometry = cavityPlan.plan.axisGeometry[axis];
            expect(
                geometry.active && geometry.validNodes == validNodes &&
                    geometry.latticeIntervals == physicalIntervals &&
                    std::abs(geometry.dx - dx) < 1.0e-15 &&
                    std::abs(geometry.physicalLo) < 1.0e-15 &&
                    std::abs(geometry.physicalHi - physicalLength) < 1.0e-15 &&
                    std::abs(geometry.envelopeLo + 0.5 * dx) < 1.0e-15 &&
                    std::abs(
                        geometry.envelopeHi -
                        (physicalLength + 0.5 * dx)) < 1.0e-15,
                "Guo cavity must preserve 129 valid nodes, 128 physical "
                "intervals and the half-cell AMReX envelope");
        }
        expect(
            boundary::matching_on_lattice_reference_axis(
                cavityPlan.plan, physicalIntervals, physicalLength) == 0,
            "Guo cavity reference must use physical intervals, not valid nodes");
    }

    amrex::BoxArray localDonorLayout(periodicZGeom.Domain());
    localDonorLayout.maxSize(2);
    expect(
        !boundary::post_stream_donor_fill_boundary_required(
            xyPlan, localDonorLayout),
        "Wide boundary FABs must use same-FAB post-stream donors");

    {
        const int nz = Descriptor::d == 3 ? 4 : 1;
        const amrex::IntVect domainLo(AMREX_D_DECL(4, 7, 2));
        const amrex::IntVect domainHi(
            AMREX_D_DECL(7, 10, 2 + nz - 1));
        const amrex::Box offsetDomain(domainLo, domainHi);
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probLo{
            AMREX_D_DECL(-1.25, 2.0, -0.75)};
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellSize{
            AMREX_D_DECL(0.5, 0.75, 1.25)};
        amrex::RealBox realBox(
            {AMREX_D_DECL(probLo[0], probLo[1], probLo[2])},
            {AMREX_D_DECL(
                probLo[0] + offsetDomain.length(0) * cellSize[0],
                probLo[1] + offsetDomain.length(1) * cellSize[1],
                probLo[2] + offsetDomain.length(2) * cellSize[2])});
        const amrex::Array<int, AMREX_SPACEDIM> periodic{
            AMREX_D_DECL(0, 1, 1)};
        const amrex::Geometry offsetGeometry(
            offsetDomain,
            &realBox,
            amrex::CoordSys::cartesian,
            periodic.data());
        std::vector<boundary::DomainBoundarySpec> offsetSpecs{
            make_guo_uniform_velocity_domain_spec(
                "offset_xlo",
                boundary::DomainSide::XLo,
                boundary::BoundaryPhysical::VelocityInlet,
                amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0}),
            make_guo_uniform_velocity_domain_spec(
                "offset_xhi",
                boundary::DomainSide::XHi,
                boundary::BoundaryPhysical::VelocityOutlet,
                amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0})};
        auto offsetResolved =
            boundary::resolve_domain_boundary_specs<Descriptor>(
                offsetSpecs, offsetGeometry);
        expect(
            offsetResolved.ok(),
            offsetResolved.error_message().c_str());
        auto offsetPlan =
            boundary::compile_post_streaming_boundary_plan<Descriptor>(
                offsetGeometry, offsetResolved.resolved.domain);
        expect(offsetPlan.ok(), offsetPlan.error_message().c_str());

        const amrex::Real expectedPhysicalLo =
            probLo[0] +
            (static_cast<amrex::Real>(offsetDomain.smallEnd(0)) + 0.5) *
                cellSize[0];
        const amrex::Real expectedPhysicalHi =
            probLo[0] +
            (static_cast<amrex::Real>(offsetDomain.bigEnd(0)) + 0.5) *
                cellSize[0];
        expect(
            std::abs(
                offsetPlan.plan.axisGeometry[0].physicalLo -
                expectedPhysicalLo) < 1.0e-14 &&
                std::abs(
                    offsetPlan.plan.axisGeometry[0].physicalHi -
                    expectedPhysicalHi) < 1.0e-14,
            "Guo plan physical bounds must use the actual outer valid "
            "indices when the domain small end is nonzero");

        std::ostringstream expectedBounds;
        expectedBounds
            << "physical_bounds=[" << expectedPhysicalLo << ','
            << expectedPhysicalHi << ']';
        const std::string diagnostics =
            boundary::format_resolved_boundary_table(
                offsetResolved.resolved, offsetGeometry);
        expect(
            diagnostics.find(expectedBounds.str()) != std::string::npos,
            "Resolved boundary diagnostics must report the same nonzero-index "
            "valid-node bounds as the compiled plan");
    }

    std::vector<boundary::DomainBoundarySpec> cavitySpecs{
        make_guo_no_slip_domain_spec("cavity_xlo", boundary::DomainSide::XLo),
        make_guo_no_slip_domain_spec("cavity_xhi", boundary::DomainSide::XHi),
        make_guo_no_slip_domain_spec("cavity_ylo", boundary::DomainSide::YLo),
        make_guo_moving_wall_domain_spec(
            "cavity_yhi", boundary::DomainSide::YHi,
            amrex::Array<amrex::Real, 3>{0.1, 0.0, 0.0})};
    auto cavityResolved = boundary::resolve_domain_boundary_specs<Descriptor>(
        cavitySpecs, periodicZGeom);
    expect(cavityResolved.ok(), cavityResolved.error_message().c_str());
    auto cavityPlanResult =
        boundary::compile_post_streaming_boundary_plan<Descriptor>(
            periodicZGeom, cavityResolved.resolved.domain);
    expect(cavityPlanResult.ok(), cavityPlanResult.error_message().c_str());
    int movingLidFaces = 0;
    int stationaryTopCorners = 0;
    for (boundary::PostStreamBoundaryRegion const& region :
         cavityPlanResult.plan.regions) {
        const bool touchesTop = [&] {
            for (int n = 0; n < region.touchedSideCount; ++n) {
                if (region.touchedSides[n] == boundary::DomainSide::YHi) {
                    return true;
                }
            }
            return false;
        }();
        if (!touchesTop) {
            continue;
        }
        if (region.touchedSideCount == 1) {
            ++movingLidFaces;
            expect(
                region.effectivePhysical == boundary::BoundaryPhysical::MovingWall &&
                    region.inward[0] == 0 && region.inward[1] == -1,
                "Strict top Guo cavity face must use moving-wall reconstruction");
        } else {
            ++stationaryTopCorners;
            expect(
                region.effectivePhysical == boundary::BoundaryPhysical::NoSlipWall &&
                    region.inward[1] == -1 && region.inward[0] != 0,
                "Guo cavity top corners must use stationary side-wall-wins donors");
        }
    }
    expect(
        movingLidFaces == 1 && stationaryTopCorners == 2,
        "Guo cavity plan must contain one moving-lid face and two stationary top corners");

    amrex::Box const& domain = periodicZGeom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    std::vector<int> visits(static_cast<std::size_t>(domain.numPts()), 0);
    int faceCount = 0;
    int cornerCount = 0;
    bool foundXLoYLo = false;
    for (boundary::PostStreamBoundaryRegion const& region : xyPlan.regions) {
        faceCount += region.touchedSideCount == 1 ? 1 : 0;
        cornerCount += region.touchedSideCount == 2 ? 1 : 0;
        for (int n = 0; n < region.touchedSideCount; ++n) {
            const boundary::DomainSide side = region.touchedSides[n];
            const int axis = boundary::domain_side_axis(side);
            const int expectedInward = boundary::domain_side_is_low(side) ? 1 : -1;
            expect(
                region.inward[axis] == expectedInward,
                "Guo region inward sign does not match its touched side");
            const int boundaryIndex = boundary::domain_side_is_low(side) ?
                domain.smallEnd(axis) : domain.bigEnd(axis);
            expect(
                region.box.smallEnd(axis) == boundaryIndex &&
                    region.box.bigEnd(axis) == boundaryIndex,
                "Guo region slab does not lie on its touched valid boundary layer");
        }
        if (region.touchedSideCount == 2 &&
            region.box.smallEnd(0) == domain.smallEnd(0) &&
            region.box.smallEnd(1) == domain.smallEnd(1)) {
            foundXLoYLo = true;
            expect(
                region.inward[0] == 1 && region.inward[1] == 1 &&
                    region.inward[2] == 0,
                "(xlo,ylo) Guo corner inward vector must be (+1,+1,0)");
            expect(
                region.effectivePhysical == boundary::BoundaryPhysical::NoSlipWall,
                "Guo inlet/wall corner must use wall-wins no-slip");
        }
        amrex::Box donorBox = region.box;
        donorBox.shift(region.inward);
        for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
            if (!xyPlan.axisGeometry[axis].active) {
                continue;
            }
            expect(
                donorBox.smallEnd(axis) > domain.smallEnd(axis) &&
                    donorBox.bigEnd(axis) < domain.bigEnd(axis),
                "Every Guo first-interior donor must lie inside the physical "
                "boundary shell");
        }
        for (int k = region.box.smallEnd(2); k <= region.box.bigEnd(2); ++k) {
            for (int j = region.box.smallEnd(1); j <= region.box.bigEnd(1); ++j) {
                for (int i = region.box.smallEnd(0); i <= region.box.bigEnd(0); ++i) {
                    const int linear =
                        ((k - domain.smallEnd(2)) * ny + (j - domain.smallEnd(1))) * nx +
                        (i - domain.smallEnd(0));
                    ++visits[static_cast<std::size_t>(linear)];
                }
            }
        }
    }
    expect(faceCount == 4 && cornerCount == 4, "D2 Guo face/corner region counts mismatch");
    expect(foundXLoYLo, "D2 Guo plan is missing the (xlo,ylo) corner");
    for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
        for (int j = domain.smallEnd(1); j <= domain.bigEnd(1); ++j) {
            for (int i = domain.smallEnd(0); i <= domain.bigEnd(0); ++i) {
                const bool boundaryNode =
                    i == domain.smallEnd(0) || i == domain.bigEnd(0) ||
                    j == domain.smallEnd(1) || j == domain.bigEnd(1);
                const int linear =
                    ((k - domain.smallEnd(2)) * ny + (j - domain.smallEnd(1))) * nx +
                    (i - domain.smallEnd(0));
                expect(
                    visits[static_cast<std::size_t>(linear)] ==
                        (boundaryNode ? 1 : 0),
                    "Guo regions must cover the boundary shell exactly once");
            }
        }
    }
    amrex::ignore_unused(nz);

    auto const narrowGeom =
        make_lbm_integration_geometry<Descriptor>(0, 1, 1, 2, 4);
    std::vector<boundary::DomainBoundarySpec> narrowSpecs{
        make_guo_uniform_velocity_domain_spec(
            "narrow_xlo", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0}),
        make_guo_uniform_velocity_domain_spec(
            "narrow_xhi", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0})};
    auto narrowResolved =
        boundary::resolve_domain_boundary_specs<Descriptor>(narrowSpecs, narrowGeom);
    expect(narrowResolved.ok(), narrowResolved.error_message().c_str());
    auto narrowPlan = boundary::compile_post_streaming_boundary_plan<Descriptor>(
        narrowGeom, narrowResolved.resolved.domain);
    expect(!narrowPlan.ok(), "Guo axis with fewer than three valid nodes must fail");

    auto const sameAxisMixedGeom =
        make_lbm_integration_geometry<Descriptor>(0, 1, 1);
    std::vector<boundary::DomainBoundarySpec> sameAxisMixedSpecs{
        make_guo_uniform_velocity_domain_spec(
            "mixed_xlo_guo", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0}),
        make_uniform_velocity_domain_spec(
            "mixed_xhi_hbb", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            amrex::Array<amrex::Real, 3>{0.05, 0.0, 0.0})};
    auto sameAxisMixedResolved = boundary::resolve_domain_boundary_specs<Descriptor>(
        sameAxisMixedSpecs, sameAxisMixedGeom);
    expect(
        !sameAxisMixedResolved.ok() &&
            contains_error(
                sameAxisMixedResolved,
                "cannot mix valid-node and domain-face geometry on the same axis"),
        "Same-axis Guo/HBB geometry must fail during resolve");

    if constexpr (Descriptor::d == 3) {
        auto const allFaceGeom =
            make_lbm_integration_geometry<Descriptor>(0, 0, 0);
        std::vector<boundary::DomainBoundarySpec> allWallSpecs{
            make_guo_no_slip_domain_spec("xlo", boundary::DomainSide::XLo),
            make_guo_no_slip_domain_spec("xhi", boundary::DomainSide::XHi),
            make_guo_no_slip_domain_spec("ylo", boundary::DomainSide::YLo),
            make_guo_no_slip_domain_spec("yhi", boundary::DomainSide::YHi),
            make_guo_no_slip_domain_spec("zlo", boundary::DomainSide::ZLo),
            make_guo_no_slip_domain_spec("zhi", boundary::DomainSide::ZHi)};
        auto allResolved = boundary::resolve_domain_boundary_specs<Descriptor>(
            allWallSpecs, allFaceGeom);
        expect(allResolved.ok(), allResolved.error_message().c_str());
        auto allPlan = boundary::compile_post_streaming_boundary_plan<Descriptor>(
            allFaceGeom, allResolved.resolved.domain);
        expect(allPlan.ok(), allPlan.error_message().c_str());
        int d3Faces = 0;
        int d3Edges = 0;
        int d3Corners = 0;
        amrex::Box const& d3Domain = allFaceGeom.Domain();
        const int d3Nx = d3Domain.length(0);
        const int d3Ny = d3Domain.length(1);
        std::vector<int> d3Visits(
            static_cast<std::size_t>(d3Domain.numPts()), 0);
        for (boundary::PostStreamBoundaryRegion const& region : allPlan.plan.regions) {
            d3Faces += region.touchedSideCount == 1 ? 1 : 0;
            d3Edges += region.touchedSideCount == 2 ? 1 : 0;
            d3Corners += region.touchedSideCount == 3 ? 1 : 0;
            expect(
                region.effectivePhysical == boundary::BoundaryPhysical::NoSlipWall,
                "All-wall D3 Guo regions must resolve to no-slip");
            for (int n = 0; n < region.touchedSideCount; ++n) {
                const boundary::DomainSide side = region.touchedSides[n];
                const int axis = boundary::domain_side_axis(side);
                const int expectedInward = boundary::domain_side_is_low(side) ? 1 : -1;
                expect(
                    region.inward[axis] == expectedInward,
                    "D3 Guo region inward sign mismatch");
            }
            amrex::Box donorBox = region.box;
            donorBox.shift(region.inward);
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
                expect(
                    donorBox.smallEnd(axis) > d3Domain.smallEnd(axis) &&
                        donorBox.bigEnd(axis) < d3Domain.bigEnd(axis),
                    "D3 Guo first-interior donor must lie inside the physical "
                    "boundary shell");
            }
            for (int k = region.box.smallEnd(2); k <= region.box.bigEnd(2); ++k) {
                for (int j = region.box.smallEnd(1); j <= region.box.bigEnd(1); ++j) {
                    for (int i = region.box.smallEnd(0); i <= region.box.bigEnd(0); ++i) {
                        const int linear =
                            ((k - d3Domain.smallEnd(2)) * d3Ny +
                             (j - d3Domain.smallEnd(1))) * d3Nx +
                            (i - d3Domain.smallEnd(0));
                        ++d3Visits[static_cast<std::size_t>(linear)];
                    }
                }
            }
        }
        expect(
            allPlan.plan.regions.size() == 26 && d3Faces == 6 &&
                d3Edges == 12 && d3Corners == 8,
            "D3 Guo plan must contain 6 faces, 12 edges and 8 corners");
        for (int k = d3Domain.smallEnd(2); k <= d3Domain.bigEnd(2); ++k) {
            for (int j = d3Domain.smallEnd(1); j <= d3Domain.bigEnd(1); ++j) {
                for (int i = d3Domain.smallEnd(0); i <= d3Domain.bigEnd(0); ++i) {
                    const bool boundaryNode =
                        i == d3Domain.smallEnd(0) || i == d3Domain.bigEnd(0) ||
                        j == d3Domain.smallEnd(1) || j == d3Domain.bigEnd(1) ||
                        k == d3Domain.smallEnd(2) || k == d3Domain.bigEnd(2);
                    const int linear =
                        ((k - d3Domain.smallEnd(2)) * d3Ny +
                         (j - d3Domain.smallEnd(1))) * d3Nx +
                        (i - d3Domain.smallEnd(0));
                    expect(
                        d3Visits[static_cast<std::size_t>(linear)] ==
                            (boundaryNode ? 1 : 0),
                        "D3 Guo regions must cover the boundary shell exactly once");
                }
            }
        }
        for (boundary::PostStreamBoundaryRegion const& region : xyPlan.regions) {
            expect(region.inward[2] == 0, "Periodic z must not create a Guo inward component");
        }
    }

    amrex::Print() << Descriptor::name()
                   << " post-stream boundary geometry integration tests passed.\n";
}


template <class Descriptor>
void run_guo_pressure_executor_integration_tests()
{
    constexpr amrex::Real sentinel = -4.567890123e20;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real rhoIn = 1.03;
    constexpr amrex::Real rhoOut = 0.97;

    const int nx = 7;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::Box domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    constexpr amrex::Real dx = 0.25;
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.5 * dx, 0.0, 0.0)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * dx,
            static_cast<amrex::Real>(ny) * dx,
            static_cast<amrex::Real>(nz) * dx)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(0, 1, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_pressure_domain_spec(
            "pressure_executor_xlo",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::PressureInlet,
            rhoIn),
        make_guo_pressure_domain_spec(
            "pressure_executor_xhi",
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::PressureOutlet,
            rhoOut)};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    amrex::BoxArray ba(domain);
    amrex::DistributionMapping dm(ba);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    LatticeField<Descriptor> populations("pressure_executor", ba, dm);
    populations.multifab().setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                if (i == domain.smallEnd(0) || i == domain.bigEnd(0)) {
                    f(i, j, k, iPop) = sentinel;
                    return;
                }
                const amrex::Real rho =
                    1.0 + 8.0e-4 * static_cast<amrex::Real>(i) +
                    2.0e-4 * static_cast<amrex::Real>(j) +
                    3.0e-5 * static_cast<amrex::Real>(k);
                const amrex::Real ux =
                    0.025 + 1.0e-4 * static_cast<amrex::Real>(j);
                const amrex::Real uy =
                    -4.0e-5 * static_cast<amrex::Real>(i);
                const amrex::Real uz = Descriptor::d == 3 ?
                    2.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                const amrex::Real perturbation =
                    (iPop % 2 == 0 ? 1.0 : -1.0) * 1.0e-7 *
                    static_cast<amrex::Real>((iPop + 1) * (j + 1));
                f(i, j, k, iPop) = Equilibrium<Descriptor>::compute(
                    iPop, rho, ux, uy, uz) + perturbation;
            });
    }
    amrex::Gpu::streamSynchronize();

    amrex::MultiFab snapshot(ba, dm, Descriptor::q, 0);
    amrex::MultiFab expected(ba, dm, Descriptor::q, 0);
    expected.setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const source = populations.multifab().const_array(mfi);
        auto const saved = snapshot.array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                saved(i, j, k, iPop) = source(i, j, k, iPop);
            });

        auto const expectedData = expected.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (i != domain.smallEnd(0) && i != domain.bigEnd(0)) {
                return;
            }
            const int donorI = i == domain.smallEnd(0) ? i + 1 : i - 1;
            const amrex::Real rhoTarget =
                i == domain.smallEnd(0) ? rhoIn : rhoOut;
            ConstPopulationCell donor{source, donorI, j, k};
            PopulationCell boundaryCell{expectedData, i, j, k};
            boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::pressure(
                donor, boundaryCell, rhoTarget);
        });
    }
    amrex::Gpu::streamSynchronize();

    boundarySystem.applyPostStreamingValidNodes(
        populations, populations, 0.0);

    amrex::MultiFab error(ba, dm, 2, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const saved = snapshot.const_array(mfi);
        auto const reference = expected.const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool isBoundary =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            amrex::Real maxPopulationError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expectedValue = isBoundary ?
                    reference(i, j, k, iPop) : saved(i, j, k, iPop);
                amrex::Real diff = actual(i, j, k, iPop) - expectedValue;
                diff = diff < 0.0 ? -diff : diff;
                maxPopulationError =
                    diff > maxPopulationError ? diff : maxPopulationError;
            }
            err(i, j, k, 0) = maxPopulationError;
            if (isBoundary) {
                ConstPopulationCell boundaryCell{actual, i, j, k};
                const amrex::Real rhoActual =
                    Moments<Descriptor>::rho(boundaryCell);
                const amrex::Real rhoTarget =
                    i == domain.smallEnd(0) ? rhoIn : rhoOut;
                amrex::Real rhoError = rhoActual - rhoTarget;
                err(i, j, k, 1) = rhoError < 0.0 ? -rhoError : rhoError;
            }
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        error.max(0) < tolerance,
        "Guo pressure executor must dispatch prescribed rho, preserve donor "
        "velocity/non-equilibrium, overwrite all q, and preserve the interior");
    expect(
        error.max(1) < tolerance,
        "Guo pressure executor boundary density moment mismatch");

    amrex::Print() << Descriptor::name()
                   << " Guo pressure executor integration tests passed.\n";
}

template <class Descriptor>
void run_guo_prescribed_state_executor_integration_tests()
{
    constexpr amrex::Real sentinel = -7.654321e20;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real rhoTarget = 1.025;
    const amrex::Array<amrex::Real, 3> prescribedVelocity{
        0.043, -0.002, Descriptor::d == 3 ? 0.001 : 0.0};
    const amrex::GpuArray<amrex::Real, 3> prescribedVelocityDevice{
        prescribedVelocity[0], prescribedVelocity[1], prescribedVelocity[2]};
    const int nx = 7;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::Box domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    const amrex::Real dy =
        1.0 / static_cast<amrex::Real>(ny - 1);
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.125, -0.5 * dy, 0.0)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * 0.25,
            (static_cast<amrex::Real>(ny) - 0.5) * dy,
            static_cast<amrex::Real>(nz) * 0.25)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(0, 0, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_prescribed_state_domain_spec(
            "prescribed_state_executor_xlo",
            boundary::DomainSide::XLo,
            rhoTarget,
            prescribedVelocity,
            boundary::BoundaryProfileKind::Poiseuille,
            1),
        make_guo_prescribed_state_domain_spec(
            "prescribed_state_executor_xhi",
            boundary::DomainSide::XHi,
            rhoTarget,
            prescribedVelocity,
            boundary::BoundaryProfileKind::Poiseuille,
            1),
        make_guo_no_slip_domain_spec(
            "prescribed_state_executor_ylo", boundary::DomainSide::YLo),
        make_guo_no_slip_domain_spec(
            "prescribed_state_executor_yhi", boundary::DomainSide::YHi)};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    amrex::BoxArray ba(domain);
    amrex::DistributionMapping dm(ba);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    LatticeField<Descriptor> populations("prescribed_state_executor", ba, dm);
    populations.multifab().setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                if (i == domain.smallEnd(0) || i == domain.bigEnd(0) ||
                    j == domain.smallEnd(1) || j == domain.bigEnd(1)) {
                    f(i, j, k, iPop) = sentinel;
                    return;
                }
                const amrex::Real rhoDonor =
                    0.99 + 5.0e-4 * static_cast<amrex::Real>(i + j + k);
                const amrex::Real uxDonor =
                    0.02 + 2.0e-4 * static_cast<amrex::Real>(j);
                const amrex::Real uyDonor =
                    -1.0e-4 * static_cast<amrex::Real>(i);
                const amrex::Real uzDonor = Descriptor::d == 3 ?
                    8.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                const amrex::Real sign =
                    ((iPop + i + j + k) & 1) == 0 ? 1.0 : -1.0;
                const amrex::Real nonEquilibrium =
                    sign * 2.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                    static_cast<amrex::Real>(iPop + 2) *
                    (1.0 + 1.0e-2 * static_cast<amrex::Real>(i + j + k));
                f(i, j, k, iPop) =
                    Equilibrium<Descriptor>::compute(
                        iPop, rhoDonor, uxDonor, uyDonor, uzDonor) +
                    nonEquilibrium;
            });
    }
    amrex::Gpu::streamSynchronize();

    boundarySystem.applyPostStreamingValidNodes(
        populations, populations, 0.0);

    // Components: strict-x full-q error, rho/ux/uy/uz errors, interior
    // preservation error, and any physical-boundary sentinel left behind.
    amrex::MultiFab error(ba, dm, 7, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool onXBoundary =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            const bool onYBoundary =
                j == domain.smallEnd(1) || j == domain.bigEnd(1);
            const bool onPhysicalBoundary = onXBoundary || onYBoundary;

            if (onPhysicalBoundary) {
                amrex::Real sentinelLeft = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    if (actual(i, j, k, iPop) == sentinel) {
                        sentinelLeft = 1.0;
                    }
                }
                err(i, j, k, 6) = sentinelLeft;
            }

            if (!onPhysicalBoundary) {
                amrex::Real maxInteriorError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    const amrex::Real rhoDonor =
                        0.99 + 5.0e-4 * static_cast<amrex::Real>(i + j + k);
                    const amrex::Real uxDonor =
                        0.02 + 2.0e-4 * static_cast<amrex::Real>(j);
                    const amrex::Real uyDonor =
                        -1.0e-4 * static_cast<amrex::Real>(i);
                    const amrex::Real uzDonor = Descriptor::d == 3 ?
                        8.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                    const amrex::Real sign =
                        ((iPop + i + j + k) & 1) == 0 ? 1.0 : -1.0;
                    const amrex::Real nonEquilibrium =
                        sign * 2.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                        static_cast<amrex::Real>(iPop + 2) *
                        (1.0 + 1.0e-2 * static_cast<amrex::Real>(i + j + k));
                    const amrex::Real expected =
                        Equilibrium<Descriptor>::compute(
                            iPop, rhoDonor, uxDonor, uyDonor, uzDonor) +
                        nonEquilibrium;
                    const amrex::Real difference =
                        actual(i, j, k, iPop) - expected;
                    const amrex::Real absoluteDifference =
                        difference < 0.0 ? -difference : difference;
                    maxInteriorError = absoluteDifference > maxInteriorError ?
                        absoluteDifference : maxInteriorError;
                }
                err(i, j, k, 5) = maxInteriorError;
                return;
            }

            // y wall wins at edges/corners.  The strict x-face nodes below
            // directly exercise the prescribed-state Poiseuille dispatch.
            if (!onXBoundary || onYBoundary) {
                return;
            }

            const int donorI =
                i == domain.smallEnd(0) ? i + 1 : i - 1;
            amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real rhoDonor =
                    0.99 + 5.0e-4 *
                        static_cast<amrex::Real>(donorI + j + k);
                const amrex::Real uxDonor =
                    0.02 + 2.0e-4 * static_cast<amrex::Real>(j);
                const amrex::Real uyDonor =
                    -1.0e-4 * static_cast<amrex::Real>(donorI);
                const amrex::Real uzDonor = Descriptor::d == 3 ?
                    8.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                const amrex::Real sign =
                    ((iPop + donorI + j + k) & 1) == 0 ? 1.0 : -1.0;
                const amrex::Real nonEquilibrium =
                    sign * 2.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                    static_cast<amrex::Real>(iPop + 2) *
                    (1.0 + 1.0e-2 *
                        static_cast<amrex::Real>(donorI + j + k));
                donor[iPop] = Equilibrium<Descriptor>::compute(
                    iPop, rhoDonor, uxDonor, uyDonor, uzDonor) +
                    nonEquilibrium;
            }

            const amrex::Real eta =
                static_cast<amrex::Real>(j - domain.smallEnd(1)) /
                static_cast<amrex::Real>(domain.length(1) - 1);
            const amrex::Real profileScale = 4.0 * eta * (1.0 - eta);
            const amrex::GpuArray<amrex::Real, 3> targetVelocity{
                profileScale * prescribedVelocityDevice[0],
                profileScale * prescribedVelocityDevice[1],
                profileScale * prescribedVelocityDevice[2]};
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::
                prescribed_state(
                    donor, expected, rhoTarget, targetVelocity);

            amrex::Real maxPopulationError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real difference =
                    actual(i, j, k, iPop) - expected[iPop];
                const amrex::Real absoluteDifference =
                    difference < 0.0 ? -difference : difference;
                maxPopulationError = absoluteDifference > maxPopulationError ?
                    absoluteDifference : maxPopulationError;
            }
            err(i, j, k, 0) = maxPopulationError;

            ConstPopulationCell boundaryCell{actual, i, j, k};
            const amrex::Real rho = Moments<Descriptor>::rho(boundaryCell);
            amrex::Real ux = 0.0;
            amrex::Real uy = 0.0;
            amrex::Real uz = 0.0;
            Moments<Descriptor>::velocity(boundaryCell, rho, ux, uy, uz);
            const amrex::Real rhoError = rho - rhoTarget;
            const amrex::Real uxError = ux - targetVelocity[0];
            const amrex::Real uyError = uy - targetVelocity[1];
            const amrex::Real uzError = uz - targetVelocity[2];
            err(i, j, k, 1) = rhoError < 0.0 ? -rhoError : rhoError;
            err(i, j, k, 2) = uxError < 0.0 ? -uxError : uxError;
            err(i, j, k, 3) = uyError < 0.0 ? -uyError : uyError;
            err(i, j, k, 4) = uzError < 0.0 ? -uzError : uzError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        error.max(0) < tolerance,
        "Guo prescribed-state Poiseuille executor full-q reconstruction mismatch");
    expect(
        error.max(1) < tolerance && error.max(2) < tolerance &&
            error.max(3) < tolerance && error.max(4) < tolerance,
        "Guo prescribed-state Poiseuille executor macroscopic moment mismatch");
    expect(
        error.max(5) < tolerance,
        "Guo prescribed-state executor must preserve strict interior populations");
    expect(
        error.max(6) == 0.0,
        "Guo prescribed-state/wall-wins executor left a boundary population unwritten");

    amrex::Print() << Descriptor::name()
                   << " Guo prescribed-state executor integration tests passed.\n";
}

template <class Descriptor>
void run_guo_moving_wall_executor_integration_tests()
{
    constexpr amrex::Real sentinel = -8.7654321e20;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real lidSpeed = 0.1;
    const int nx = 7;
    const int ny = 7;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::Box domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    constexpr amrex::Real dx = 0.25;
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.5 * dx, -0.5 * dx, 0.0)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * dx,
            (static_cast<amrex::Real>(ny) - 0.5) * dx,
            static_cast<amrex::Real>(nz) * dx)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(0, 0, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_no_slip_domain_spec(
            "moving_wall_executor_xlo", boundary::DomainSide::XLo),
        make_guo_no_slip_domain_spec(
            "moving_wall_executor_xhi", boundary::DomainSide::XHi),
        make_guo_no_slip_domain_spec(
            "moving_wall_executor_ylo", boundary::DomainSide::YLo),
        make_guo_moving_wall_domain_spec(
            "moving_wall_executor_yhi", boundary::DomainSide::YHi,
            amrex::Array<amrex::Real, 3>{lidSpeed, 0.0, 0.0})};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    amrex::DistributionMapping dm(ba);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    LatticeField<Descriptor> populations("moving_wall_executor", ba, dm);
    populations.multifab().setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                const bool onBoundary =
                    i == domain.smallEnd(0) || i == domain.bigEnd(0) ||
                    j == domain.smallEnd(1) || j == domain.bigEnd(1);
                if (onBoundary) {
                    f(i, j, k, iPop) = sentinel;
                    return;
                }
                const amrex::Real rho =
                    1.0 + 4.0e-4 * static_cast<amrex::Real>(i) +
                    7.0e-4 * static_cast<amrex::Real>(j) +
                    3.0e-5 * static_cast<amrex::Real>(k);
                const amrex::Real ux =
                    0.018 + 2.0e-4 * static_cast<amrex::Real>(j);
                const amrex::Real uy =
                    -0.009 + 1.0e-4 * static_cast<amrex::Real>(i);
                const amrex::Real uz = Descriptor::d == 3 ?
                    6.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                const amrex::Real sign =
                    ((iPop + i + 2 * j + k) & 1) == 0 ? 1.0 : -1.0;
                const amrex::Real nonEquilibrium =
                    sign * 1.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                    (1.0 + 1.0e-2 * static_cast<amrex::Real>(i + j + k));
                f(i, j, k, iPop) = Equilibrium<Descriptor>::compute(
                    iPop, rho, ux, uy, uz) + nonEquilibrium;
            });
    }
    amrex::Gpu::streamSynchronize();

    boundarySystem.applyPostStreamingValidNodes(
        populations, populations, 0.0);

    // Components: full-q reconstruction, boundary velocity moment, preserved
    // strict interior, and any sentinel left on the physical boundary shell.
    amrex::MultiFab error(ba, dm, 4, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool onLeft = i == domain.smallEnd(0);
            const bool onRight = i == domain.bigEnd(0);
            const bool onBottom = j == domain.smallEnd(1);
            const bool onTop = j == domain.bigEnd(1);
            const bool onSide = onLeft || onRight;
            const bool onBoundary = onSide || onBottom || onTop;

            if (!onBoundary) {
                amrex::Real maxInteriorError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    const amrex::Real rho =
                        1.0 + 4.0e-4 * static_cast<amrex::Real>(i) +
                        7.0e-4 * static_cast<amrex::Real>(j) +
                        3.0e-5 * static_cast<amrex::Real>(k);
                    const amrex::Real ux =
                        0.018 + 2.0e-4 * static_cast<amrex::Real>(j);
                    const amrex::Real uy =
                        -0.009 + 1.0e-4 * static_cast<amrex::Real>(i);
                    const amrex::Real uz = Descriptor::d == 3 ?
                        6.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                    const amrex::Real sign =
                        ((iPop + i + 2 * j + k) & 1) == 0 ? 1.0 : -1.0;
                    const amrex::Real nonEquilibrium =
                        sign * 1.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                        (1.0 + 1.0e-2 * static_cast<amrex::Real>(i + j + k));
                    const amrex::Real expected =
                        Equilibrium<Descriptor>::compute(
                            iPop, rho, ux, uy, uz) + nonEquilibrium;
                    amrex::Real difference = actual(i, j, k, iPop) - expected;
                    difference = difference < 0.0 ? -difference : difference;
                    maxInteriorError = difference > maxInteriorError ?
                        difference : maxInteriorError;
                }
                err(i, j, k, 2) = maxInteriorError;
                return;
            }

            int rx = 0;
            int ry = 0;
            if (onLeft) {
                rx = 1;
            } else if (onRight) {
                rx = -1;
            }
            if (onBottom) {
                ry = 1;
            } else if (onTop) {
                ry = -1;
            }
            const int donorI = i + rx;
            const int donorJ = j + ry;
            amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real rho =
                    1.0 + 4.0e-4 * static_cast<amrex::Real>(donorI) +
                    7.0e-4 * static_cast<amrex::Real>(donorJ) +
                    3.0e-5 * static_cast<amrex::Real>(k);
                const amrex::Real ux =
                    0.018 + 2.0e-4 * static_cast<amrex::Real>(donorJ);
                const amrex::Real uy =
                    -0.009 + 1.0e-4 * static_cast<amrex::Real>(donorI);
                const amrex::Real uz = Descriptor::d == 3 ?
                    6.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                const amrex::Real sign =
                    ((iPop + donorI + 2 * donorJ + k) & 1) == 0 ? 1.0 : -1.0;
                const amrex::Real nonEquilibrium =
                    sign * 1.0e-7 * static_cast<amrex::Real>(iPop + 1) *
                    (1.0 + 1.0e-2 *
                        static_cast<amrex::Real>(donorI + donorJ + k));
                donor[iPop] = Equilibrium<Descriptor>::compute(
                    iPop, rho, ux, uy, uz) + nonEquilibrium;
            }

            const bool movingLidNode = onTop && !onSide;
            const amrex::GpuArray<amrex::Real, 3> targetVelocity{
                movingLidNode ? lidSpeed : 0.0, 0.0, 0.0};
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            if (movingLidNode) {
                boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::
                    moving_wall(donor, expected, targetVelocity);
            } else {
                boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::
                    no_slip(donor, expected);
            }

            amrex::Real maxPopulationError = 0.0;
            amrex::Real sentinelLeft = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                amrex::Real difference =
                    actual(i, j, k, iPop) - expected[iPop];
                difference = difference < 0.0 ? -difference : difference;
                maxPopulationError = difference > maxPopulationError ?
                    difference : maxPopulationError;
                if (actual(i, j, k, iPop) == sentinel) {
                    sentinelLeft = 1.0;
                }
            }
            err(i, j, k, 0) = maxPopulationError;
            err(i, j, k, 3) = sentinelLeft;

            ConstPopulationCell boundaryCell{actual, i, j, k};
            const amrex::Real rho = Moments<Descriptor>::rho(boundaryCell);
            amrex::Real ux = 0.0;
            amrex::Real uy = 0.0;
            amrex::Real uz = 0.0;
            Moments<Descriptor>::velocity(boundaryCell, rho, ux, uy, uz);
            const amrex::Real uxError = ux - targetVelocity[0];
            const amrex::Real velocityErrorSqr =
                uxError * uxError + uy * uy + uz * uz;
            err(i, j, k, 1) = std::sqrt(velocityErrorSqr);
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        error.max(0) < tolerance,
        "Guo moving-wall executor must use first-interior donors, reconstruct "
        "all q, and apply stationary wall-wins at top corners");
    expect(
        error.max(1) < tolerance,
        "Guo moving-wall/static-wall boundary velocity moments mismatch");
    expect(
        error.max(2) < tolerance,
        "Guo moving-wall executor must preserve strict interior populations");
    expect(
        error.max(3) == 0.0,
        "Guo moving-wall executor left a physical boundary population unwritten");

    amrex::Print() << Descriptor::name()
                   << " Guo moving-wall executor integration tests passed.\n";
}

template <class Descriptor>
struct BoundaryPositionTimeReconstructionProbe {
    template <class DonorCell, class BoundaryCell>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void operator()(
        DonorCell const&,
        BoundaryCell& boundary,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real physicalTime) const noexcept
    {
        const amrex::Real sample =
            position[0] + 2.0 * position[1] + 3.0 * position[2] +
            4.0 * physicalTime;
        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            boundary[iPop] =
                1000.0 * static_cast<amrex::Real>(iPop + 1) + sample;
        }
    }
};

template <class Descriptor>
void run_boundary_sampling_position_integration_tests()
{
    constexpr amrex::Real tolerance = 1.0e-14;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    const amrex::IntVect domainLo(AMREX_D_DECL(4, 7, 2));
    const amrex::IntVect domainHi(
        AMREX_D_DECL(7, 10, 2 + nz - 1));
    const amrex::Box domain(domainLo, domainHi);
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> probLo{
        AMREX_D_DECL(-1.25, 2.0, -0.75)};
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> cellSize{
        AMREX_D_DECL(0.5, 0.75, 1.25)};
    amrex::RealBox realBox(
        {AMREX_D_DECL(probLo[0], probLo[1], probLo[2])},
        {AMREX_D_DECL(
            probLo[0] + domain.length(0) * cellSize[0],
            probLo[1] + domain.length(1) * cellSize[1],
            probLo[2] + domain.length(2) * cellSize[2])});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(1, 1, 1)};
    const amrex::Geometry geometry(
        domain,
        &realBox,
        amrex::CoordSys::cartesian,
        isPeriodic.data());

    const int i = domain.smallEnd(0) + 1;
    const int j = domain.smallEnd(1) + 1;
    const int k = domain.smallEnd(2);
    const auto cellCenter = boundary::boundary_cell_center_position(
        geometry.data(), i, j, k);
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
        const amrex::Real expected =
            probLo[axis] +
            (static_cast<amrex::Real>(index[axis]) + 0.5) *
                cellSize[axis];
        expect(
            std::abs(cellCenter[axis] - expected) < tolerance,
            "Guo boundary-node physical cell-center position mismatch");
    }

    int diagonalPopulation = -1;
    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
        if (Descriptor::cx(iPop) == 1 &&
            Descriptor::cy(iPop) == 1 &&
            Descriptor::cz(iPop) == 0) {
            diagonalPopulation = iPop;
            break;
        }
    }
    expect(
        diagonalPopulation >= 0,
        "Boundary position probe requires an xy diagonal population");
    const auto linkPosition =
        boundary::halfway_boundary_link_position<Descriptor>(
            geometry.data(), i, j, k, diagonalPopulation);
    for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
        const amrex::Real expected =
            cellCenter[axis] -
            0.5 * static_cast<amrex::Real>(
                      Descriptor::c(diagonalPopulation, axis)) *
                cellSize[axis];
        expect(
            std::abs(linkPosition[axis] - expected) < tolerance,
            "HBB halfway-link physical position/sign mismatch");
    }

    amrex::BoxArray boxArray(domain);
    amrex::DistributionMapping distribution(boxArray);
    LatticeField<Descriptor> populations(
        "boundary_position_time_probe",
        boxArray,
        distribution);
    populations.multifab().setVal(-9.87654321e20);
    boundary::PostStreamBoundaryRegion region;
    region.box = domain;
    region.box.setBig(0, domain.smallEnd(0));
    region.box.growLo(1, -1);
    region.box.growHi(1, -1);
    if constexpr (Descriptor::d == 3) {
        region.box.growLo(2, -1);
        region.box.growHi(2, -1);
    }
    region.inward[0] = 1;
    const solvers::BoundaryStepContext boundaryStep(1.25, 0.04);
    boundaryStep.runPostStreaming([&](amrex::Real physicalTime) {
        boundary::LbmPostStreamingBoundaryExecutor<Descriptor>::applyRegion(
            populations,
            region,
            geometry.data(),
            physicalTime,
            BoundaryPositionTimeReconstructionProbe<Descriptor>{});
    });

    amrex::MultiFab error(boxArray, distribution, 1, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(
             populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox() & region.box;
        if (!bx.ok()) {
            continue;
        }
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                const amrex::Real x =
                    probLo[0] +
                    (static_cast<amrex::Real>(ii) + 0.5) *
                        cellSize[0];
                const amrex::Real y =
                    probLo[1] +
                    (static_cast<amrex::Real>(jj) + 0.5) *
                        cellSize[1];
                const amrex::Real z =
                    probLo[2] +
                    (static_cast<amrex::Real>(kk) + 0.5) *
                        cellSize[2];
                const amrex::Real sample =
                    x + 2.0 * y + 3.0 * z + 4.0 * 1.29;
                amrex::Real maxError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    amrex::Real difference =
                        actual(ii, jj, kk, iPop) -
                        (1000.0 * static_cast<amrex::Real>(iPop + 1) +
                         sample);
                    difference =
                        difference < 0.0 ? -difference : difference;
                    maxError =
                        difference > maxError ? difference : maxError;
                }
                err(ii, jj, kk, 0) = maxError;
            });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        error.max(0) < tolerance,
        "Post-stream executor must forward valid-center position/current time");
}

template <class Descriptor>
void run_domain_boundary_fill_integration_tests()
{
    run_boundary_sampling_position_integration_tests<Descriptor>();

    {
        constexpr int validCenters = 129;
        constexpr amrex::Real physicalLength = 1.0;
        constexpr amrex::Real dx = physicalLength / validCenters;
        const int nz = Descriptor::d == 3 ? 4 : 1;
        const amrex::Box hbbDomain(
            amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
            amrex::IntVect(AMREX_D_DECL(
                validCenters - 1, validCenters - 1, nz - 1)));
        const amrex::RealBox hbbEnvelope(
            {AMREX_D_DECL(0.0, 0.0, 0.0)},
            {AMREX_D_DECL(
                physicalLength,
                physicalLength,
                static_cast<amrex::Real>(nz) * dx)});
        const amrex::Array<int, AMREX_SPACEDIM> periodic{
            AMREX_D_DECL(0, 0, 1)};
        const amrex::Geometry hbbGeometry(
            hbbDomain,
            &hbbEnvelope,
            amrex::CoordSys::cartesian,
            periodic.data());
        const amrex::Real lowValidCenter =
            hbbGeometry.ProbLo(0) + 0.5 * dx;
        const amrex::Real lowGhostCenter =
            hbbGeometry.ProbLo(0) - 0.5 * dx;
        const amrex::Real highValidCenter =
            hbbGeometry.ProbHi(0) - 0.5 * dx;
        const amrex::Real highGhostCenter =
            hbbGeometry.ProbHi(0) + 0.5 * dx;
        expect(
            hbbGeometry.Domain().length(0) == validCenters &&
                std::abs(hbbGeometry.CellSize(0) - dx) < 1.0e-15 &&
                std::abs(0.5 * (lowValidCenter + lowGhostCenter)) < 1.0e-15 &&
                std::abs(
                    0.5 * (highValidCenter + highGhostCenter) -
                    physicalLength) < 1.0e-15,
            "HBB cavity must preserve 129 valid centers, 129 physical "
            "intervals and midpoint walls at the AMReX envelope");
    }

    auto const faceGeom = make_lbm_integration_geometry<Descriptor>(0, 1, 1);

    const int nx = 4;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::Box domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    amrex::DistributionMapping dm(ba);

    LatticeField<Descriptor> populations("domain_boundary_fill_populations", ba, dm);
    boundary::DomainFaceTable emptyTable;
    std::vector<std::string> emptyCoverageErrors =
        boundary::validate_domain_face_coverage<Descriptor>(emptyTable.view(), faceGeom);
    expect(
        !emptyCoverageErrors.empty(),
        "Missing all non-periodic domain face specs must fail validation");

    auto missingCoverage = boundary::resolve_domain_boundary_specs<Descriptor>(
        {make_domain_wall_spec("left_wall", boundary::DomainSide::XLo)}, faceGeom);
    const std::string missingCoverageResolveError = missingCoverage.error_message();
    expect(missingCoverage.ok(), missingCoverageResolveError.c_str());
    boundary::DomainFaceTable missingTable =
        boundary::build_domain_face_table(
            missingCoverage.resolved.domain, faceGeom);
    std::vector<std::string> coverageErrors =
        boundary::validate_domain_face_coverage<Descriptor>(missingTable.view(), faceGeom);
    expect(!coverageErrors.empty(), "Missing non-periodic domain face spec must fail validation");

    boundary::DomainBoundarySystem<Descriptor> noSlipSystem =
        make_domain_boundary_system<Descriptor>(
            faceGeom,
            {make_domain_wall_spec("left_wall", boundary::DomainSide::XLo),
             make_domain_wall_spec("right_wall", boundary::DomainSide::XHi)});

    initialize_boundary_fill_pattern(populations);
    noSlipSystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab faceErr(ba, dm, 1, 0);
    faceErr.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellLo(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = faceErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    boundary_fill_pattern<Descriptor>(fi, fj, fk, opp);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(faceErr.max(0) < 1.0e-14, "No-slip domain ghost fill mismatch");

    boundary::DomainBoundarySpec movingWall =
        make_domain_wall_spec("moving_left_wall", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::MovingWall);
    movingWall.profile = boundary::BoundaryProfileKind::Uniform;
    movingWall.prescribedULb =
        amrex::Array<amrex::Real, 3>{0.0, 0.02, 0.0};
    movingWall.hasPrescribedULb = true;
    boundary::DomainBoundarySystem<Descriptor> movingSystem =
        make_domain_boundary_system<Descriptor>(
            faceGeom,
            {movingWall, make_domain_wall_spec("right_wall", boundary::DomainSide::XHi)});

    initialize_boundary_fill_pattern(populations);
    movingSystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab movingErr(ba, dm, 1, 0);
    movingErr.setVal(0.0);
    const amrex::GpuArray<amrex::Real, 3> wallULb{0.0, 0.02, 0.0};
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellLo(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = movingErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                const amrex::Real correction =
                    boundary::HalfwayBounceBackScheme<Descriptor>::moving_wall_correction(
                        iPop, wallULb);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    (boundary_fill_pattern<Descriptor>(fi, fj, fk, opp) + correction);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(movingErr.max(0) < 1.0e-14, "Moving-wall domain ghost fill mismatch");

    const amrex::Array<amrex::Real, 3> velocityULb{0.03, 0.0, 0.0};
    boundary::DomainBoundarySystem<Descriptor> velocitySystem =
        make_domain_boundary_system<Descriptor>(
            faceGeom,
            {make_uniform_velocity_domain_spec(
                 "velocity_inlet", boundary::DomainSide::XLo,
                 boundary::BoundaryPhysical::VelocityInlet, velocityULb),
             make_uniform_velocity_domain_spec(
                 "velocity_outlet", boundary::DomainSide::XHi,
                 boundary::BoundaryPhysical::VelocityOutlet, velocityULb)});

    initialize_boundary_fill_pattern(populations);
    velocitySystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab velocityErr(ba, dm, 1, 0);
    velocityErr.setVal(0.0);
    const amrex::GpuArray<amrex::Real, 3> velocityULbGpu{
        velocityULb[0], velocityULb[1], velocityULb[2]};
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellLo(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = velocityErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                const amrex::Real rho =
                    boundary_fill_pattern_density<Descriptor>(fi, fj, fk);
                const amrex::Real correction =
                    boundary::HalfwayBounceBackScheme<Descriptor>::velocity_correction(
                        iPop, rho, velocityULbGpu);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    (boundary_fill_pattern<Descriptor>(fi, fj, fk, opp) + correction);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j, k, 0) = maxErr;
        });
    }
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellHi(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = velocityErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                const amrex::Real rho =
                    boundary_fill_pattern_density<Descriptor>(fi, fj, fk);
                const amrex::Real correction =
                    boundary::HalfwayBounceBackScheme<Descriptor>::velocity_correction(
                        iPop, rho, velocityULbGpu);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    (boundary_fill_pattern<Descriptor>(fi, fj, fk, opp) + correction);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i - 1, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        velocityErr.max(0) < 1.0e-8,
        "Velocity domain ghost fill mismatch");

    constexpr amrex::Real poiseuilleUMax = 0.1;
    boundary::DomainBoundarySystem<Descriptor> poiseuilleSystem =
        make_domain_boundary_system<Descriptor>(
            faceGeom,
            {make_poiseuille_velocity_domain_spec(
                 "poiseuille_inlet", boundary::DomainSide::XLo,
                 boundary::BoundaryPhysical::VelocityInlet, 1, poiseuilleUMax),
             make_poiseuille_velocity_domain_spec(
                 "poiseuille_outlet", boundary::DomainSide::XHi,
                 boundary::BoundaryPhysical::VelocityOutlet, 1, poiseuilleUMax)});

    initialize_boundary_fill_pattern(populations);
    poiseuilleSystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab poiseuilleErr(ba, dm, 1, 0);
    poiseuilleErr.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellLo(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = poiseuilleErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                const amrex::Real rhoWall = Descriptor::rho0_lb();
                const amrex::Real yWall =
                    static_cast<amrex::Real>(fj - domainBox.smallEnd(1)) + 0.5 -
                    0.5 * static_cast<amrex::Real>(Descriptor::cy(iPop));
                const amrex::Real eta =
                    yWall / static_cast<amrex::Real>(domainBox.length(1));
                const amrex::Real ux =
                    4.0 * poiseuilleUMax * eta * (1.0 - eta);
                const amrex::GpuArray<amrex::Real, 3> wallU{ux, 0.0, 0.0};
                const amrex::Real correction =
                    boundary::HalfwayBounceBackScheme<Descriptor>::velocity_correction(
                        iPop, rhoWall, wallU);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    (boundary_fill_pattern<Descriptor>(fi, fj, fk, opp) + correction);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j, k, 0) = maxErr;
        });
    }
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellHi(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = poiseuilleErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                const amrex::Real rhoWall = Descriptor::rho0_lb();
                const amrex::Real yWall =
                    static_cast<amrex::Real>(fj - domainBox.smallEnd(1)) + 0.5 -
                    0.5 * static_cast<amrex::Real>(Descriptor::cy(iPop));
                const amrex::Real eta =
                    yWall / static_cast<amrex::Real>(domainBox.length(1));
                const amrex::Real ux =
                    4.0 * poiseuilleUMax * eta * (1.0 - eta);
                const amrex::GpuArray<amrex::Real, 3> wallU{ux, 0.0, 0.0};
                const amrex::Real correction =
                    boundary::HalfwayBounceBackScheme<Descriptor>::velocity_correction(
                        iPop, rhoWall, wallU);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    (boundary_fill_pattern<Descriptor>(fi, fj, fk, opp) + correction);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i - 1, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        poiseuilleErr.max(0) < 1.0e-8,
        "Poiseuille velocity ghost fill must use link-wall profile and reference rho correction");

    auto const cornerGeom = make_lbm_integration_geometry<Descriptor>(0, 0, 1);
    boundary::DomainBoundarySystem<Descriptor> cornerSystem =
        make_domain_boundary_system<Descriptor>(
            cornerGeom,
            {make_domain_wall_spec("left_wall", boundary::DomainSide::XLo),
             make_domain_wall_spec("right_wall", boundary::DomainSide::XHi),
             make_domain_wall_spec("bottom_wall", boundary::DomainSide::YLo),
             make_domain_wall_spec("top_wall", boundary::DomainSide::YHi)});
    initialize_boundary_fill_pattern(populations);
    cornerSystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab cornerErr(ba, dm, 1, 0);
    cornerErr.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box corner = amrex::adjCellLo(amrex::adjCellLo(mfi.validbox(), 0, 1), 1, 1);
        corner &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = cornerErr.array(mfi);
        amrex::Box const domainBox = cornerGeom.Domain();
        amrex::ParallelFor(corner, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                if (Descriptor::cx(iPop) != 1 || Descriptor::cy(iPop) != 1) {
                    continue;
                }
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    boundary_fill_pattern<Descriptor>(fi, fj, fk, opp);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j + 1, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(cornerErr.max(0) < 1.0e-14, "Corner domain ghost fill mismatch");

    boundary::DomainBoundarySystem<Descriptor> cornerWallWinsSystem =
        make_domain_boundary_system<Descriptor>(
            cornerGeom,
            {make_uniform_velocity_domain_spec(
                 "corner_velocity_inlet", boundary::DomainSide::XLo,
                 boundary::BoundaryPhysical::VelocityInlet, velocityULb),
             make_domain_wall_spec("right_wall", boundary::DomainSide::XHi),
             make_domain_wall_spec("bottom_wall", boundary::DomainSide::YLo),
             make_domain_wall_spec("top_wall", boundary::DomainSide::YHi)});
    initialize_boundary_fill_pattern(populations);
    cornerWallWinsSystem.fillPhysicalGhosts(populations, 0.0);

    amrex::MultiFab wallWinsErr(ba, dm, 1, 0);
    wallWinsErr.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box corner = amrex::adjCellLo(amrex::adjCellLo(mfi.validbox(), 0, 1), 1, 1);
        corner &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = wallWinsErr.array(mfi);
        amrex::Box const domainBox = cornerGeom.Domain();
        amrex::ParallelFor(corner, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                if (Descriptor::cx(iPop) != 1 || Descriptor::cy(iPop) != 1) {
                    continue;
                }
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    boundary_fill_pattern<Descriptor>(fi, fj, fk, opp);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j + 1, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        wallWinsErr.max(0) < 1.0e-14,
        "Velocity/no-slip corner must use wall-wins stationary bounce-back");

    boundary::BoundarySystem<Descriptor> boundarySystem;
    define_boundary_system_for_test(
        boundarySystem,
        {make_domain_wall_spec("left_wall", boundary::DomainSide::XLo),
         make_domain_wall_spec("right_wall", boundary::DomainSide::XHi)},
        faceGeom,
        ba,
        dm);
    initialize_boundary_fill_pattern(populations);
    boundarySystem.fillPreStreaming(populations, 0.0);

    amrex::MultiFab preStreamErr(ba, dm, 1, 0);
    preStreamErr.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box bndry = amrex::adjCellLo(mfi.validbox(), 0, 1);
        bndry &= mfi.fabbox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = preStreamErr.array(mfi);
        amrex::Box const domainBox = faceGeom.Domain();
        amrex::ParallelFor(bndry, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int fi = i + Descriptor::cx(iPop);
                const int fj = j + Descriptor::cy(iPop);
                const int fk = k + Descriptor::cz(iPop);
                if (!boundary::domain_contains(domainBox, fi, fj, fk)) {
                    continue;
                }
                const int opp = Descriptor::opposite(iPop);
                amrex::Real diff =
                    f(i, j, k, iPop) -
                    boundary_fill_pattern<Descriptor>(fi, fj, fk, opp);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i + 1, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        preStreamErr.max(0) < 1.0e-14,
        "BoundarySystem pre-stream domain ghost fill mismatch");

    amrex::Print() << Descriptor::name() << " domain boundary fill integration tests passed.\n";
}

template <class Descriptor>
void run_hbb_multibox_read_integration_tests()
{
    const auto geometry = make_lbm_integration_geometry<Descriptor>(
        0, 0, Descriptor::d == 2 ? 1 : 0, 8, 8,
        Descriptor::d == 3 ? 8 : 1);
    const auto domain = geometry.Domain();
    std::vector<boundary::DomainBoundarySpec> specs{
        make_domain_wall_spec("wall_xlo", boundary::DomainSide::XLo),
        make_domain_wall_spec("wall_xhi", boundary::DomainSide::XHi),
        make_domain_wall_spec("wall_ylo", boundary::DomainSide::YLo),
        make_domain_wall_spec("wall_yhi", boundary::DomainSide::YHi)};
    if constexpr (Descriptor::d == 3) {
        specs.push_back(make_domain_wall_spec("wall_zlo", boundary::DomainSide::ZLo));
        specs.push_back(make_domain_wall_spec("wall_zhi", boundary::DomainSide::ZHi));
    }

    for (int maxBoxSize : {8, 4}) {
        amrex::BoxArray boxArray(domain);
        boxArray.maxSize(maxBoxSize);
        amrex::DistributionMapping distribution(boxArray);
        boundary::BoundarySystem<Descriptor> boundaries;
        define_boundary_system_for_test(
            boundaries, specs, geometry, boxArray, distribution);
        LatticeField<Descriptor> source("hbb_source", boxArray, distribution);
        LatticeField<Descriptor> destination("hbb_destination", boxArray, distribution);
        initialize_boundary_fill_pattern(source);

        if (maxBoxSize == 4) {
            // Ghost (4,-1,0), direction (1,1,0): source (5,0,0) is
            // inside the domain but outside the one-ghost allocation of
            // the box containing (3,0,0). AMREX_BOUND_CHECK guards this read.
            bool hasCorner = false;
            for (int grid = 0; grid < boxArray.size(); ++grid) {
                if (boxArray[grid].contains(amrex::IntVect(3, 0, 0))) {
                    hasCorner = domain.contains(amrex::IntVect(5, 0, 0)) &&
                        !amrex::grow(boxArray[grid], 1).contains(amrex::IntVect(5, 0, 0));
                }
            }
            expect(hasCorner, "HBB split fixture must expose the outer-FAB diagonal");
        }

        boundaries.fillPreStreaming(source, 1.25);
        operators::PullStreamingOperator<Descriptor>{}.apply(source, destination);
        amrex::MultiFab mismatch(boxArray, distribution, 1, 0);
        for (amrex::MFIter mfi(destination.multifab(), amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const auto valid = mfi.tilebox();
            const auto populations = destination.multifab().const_array(mfi);
            const auto error = mismatch.array(mfi);
            amrex::ParallelFor(valid, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real wrong = 0.0;
                for (int q = 0; q < Descriptor::q; ++q) {
                    const int si = i - Descriptor::cx(q);
                    const int sj = j - Descriptor::cy(q);
                    const int sk = k - Descriptor::cz(q);
                    const bool inside = domain.contains(amrex::IntVect(si, sj, sk));
                    // Independent stationary-wall pull expectation; every
                    // value is an exactly representable integer in Real.
                    const amrex::Real expected = inside ?
                        boundary_fill_pattern<Descriptor>(si, sj, sk, q) :
                        boundary_fill_pattern<Descriptor>(i, j, k, Descriptor::opposite(q));
                    if (populations(i, j, k, q) != expected) {
                        wrong = 1.0;
                    }
                }
                error(i, j, k) = wrong;
            });
        }
        amrex::Gpu::streamSynchronize();
        expect(mismatch.max(0) == 0.0,
            "HBB valid pull result must agree for single and split Boxes");
    }
    amrex::Print() << Descriptor::name() << " HBB multi-Box read tests passed.\n";
}

template <class Descriptor>
void run_post_stream_boundary_lifecycle_integration_tests()
{
    constexpr amrex::Real sentinel = -1.23456789e30;
    constexpr amrex::Real corruptSource = -9.87654321e20;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real sourcePhysicalTime = 1.25;
    constexpr amrex::Real dtPhys = 0.04;
    std::array<amrex::Real, 2> observedPreTimes{-1.0, -1.0};
    std::array<amrex::Real, 2> observedPostTimes{-1.0, -1.0};

    const int nx = 7;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);

    constexpr amrex::Real dx = 0.25;
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.5 * dx, 0.0, 0.0)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * dx,
            static_cast<amrex::Real>(ny) * dx,
            static_cast<amrex::Real>(nz) * dx)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(0, 1, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    const amrex::Array<amrex::Real, 3> prescribedVelocity{0.04, 0.0, 0.0};
    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_uniform_velocity_domain_spec(
            "lifecycle_xlo",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            prescribedVelocity),
        make_guo_uniform_velocity_domain_spec(
            "lifecycle_xhi",
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            prescribedVelocity)};

    boundary::BoundarySystem<Descriptor> boundarySystem;
    amrex::BoxArray ba(domain);
    ba.maxSize(3);
    amrex::DistributionMapping dm(ba);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    LatticeField<Descriptor> src("lifecycle_src", ba, dm);
    LatticeField<Descriptor> dst("lifecycle_dst", ba, dm);

    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = src.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                const amrex::Real rho =
                    1.0 + 1.0e-3 * static_cast<amrex::Real>(i) +
                    2.0e-4 * static_cast<amrex::Real>(j) +
                    3.0e-5 * static_cast<amrex::Real>(k);
                const amrex::Real ux =
                    prescribedVelocity[0] +
                    1.0e-4 * static_cast<amrex::Real>(j);
                const amrex::Real uy =
                    -5.0e-5 * static_cast<amrex::Real>(i);
                const amrex::Real uz = Descriptor::d == 3 ?
                    2.0e-5 * static_cast<amrex::Real>(k) : 0.0;
                f(i, j, k, iPop) = Equilibrium<Descriptor>::compute(
                    iPop, rho, ux, uy, uz);
                if (i == domain.smallEnd(0) || i == domain.bigEnd(0)) {
                    f(i, j, k, iPop) = sentinel;
                }
            });
    }

    // Initial reconstruction must make boundary populations legal before any
    // macroscopic diagnostic or plotfile can observe them.
    boundarySystem.initializeValidBoundaries(src, sourcePhysicalTime);
    amrex::MultiFab t0SentinelMatches(ba, dm, 1, 0);
    t0SentinelMatches.setVal(0.0);
    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = src.multifab().const_array(mfi);
        auto const matches = t0SentinelMatches.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (i != domain.smallEnd(0) && i != domain.bigEnd(0)) {
                return;
            }
            amrex::Real found = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                found = found > (f(i, j, k, iPop) == sentinel ? 1.0 : 0.0) ?
                    found : (f(i, j, k, iPop) == sentinel ? 1.0 : 0.0);
            }
            matches(i, j, k, 0) = found;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        t0SentinelMatches.max(0) == 0.0,
        "Initial Guo reconstruction must overwrite all q boundary populations");

    BGKDynamics<Descriptor> dynamics(0.9);
    operators::CollisionOperator<Descriptor, BGKDynamics<Descriptor>> collision(
        dynamics);
    operators::PullStreamingOperator<Descriptor> streaming;
    const solvers::BoundaryStepContext firstBoundaryStep(
        sourcePhysicalTime, dtPhys);
    collision.apply(src);
    firstBoundaryStep.runPreStreaming(
        [&](amrex::Real physicalTime) {
            observedPreTimes[0] = physicalTime;
            boundarySystem.fillPreStreaming(src, physicalTime);
        });

    dst.multifab().setVal(sentinel);
    streaming.apply(src, dst);

    amrex::MultiFab streamLifecycleError(ba, dm, 3, 0);
    streamLifecycleError.setVal(0.0);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const source = src.multifab().const_array(mfi);
        auto const streamed = dst.multifab().const_array(mfi);
        auto const error = streamLifecycleError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real maxError = 0.0;
            amrex::Real foundXLoGhostValue = 0.0;
            amrex::Real foundXHiGhostValue = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expected =
                    kernels::PullStreamingKernel<Descriptor>::population(
                        source, i, j, k, iPop);
                amrex::Real diff = streamed(i, j, k, iPop) - expected;
                diff = diff < 0.0 ? -diff : diff;
                maxError = diff > maxError ? diff : maxError;
                const int sourceI = i - Descriptor::cx(iPop);
                if (i == domain.smallEnd(0) &&
                    sourceI < domain.smallEnd(0) &&
                    streamed(i, j, k, iPop) == 0.0) {
                    foundXLoGhostValue = 1.0;
                }
                if (i == domain.bigEnd(0) &&
                    sourceI > domain.bigEnd(0) &&
                    streamed(i, j, k, iPop) == 0.0) {
                    foundXHiGhostValue = 1.0;
                }
            }
            error(i, j, k, 0) = maxError;
            error(i, j, k, 1) = foundXLoGhostValue;
            error(i, j, k, 2) = foundXHiGhostValue;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        streamLifecycleError.max(0) < tolerance,
        "pull streaming must update every valid destination");
    expect(
        streamLifecycleError.max(1) == 1.0 &&
            streamLifecycleError.max(2) == 1.0,
        "full-domain streaming check must observe finite disposable values "
        "pulled from both non-periodic physical ghost layers");

    // Preserve streamed valid data and independently reconstruct the expected
    // xlo/xhi values from dst donors at the post-stream time layer.
    amrex::MultiFab streamedSnapshot(ba, dm, Descriptor::q, 0);
    amrex::MultiFab expectedBoundary(ba, dm, Descriptor::q, 0);
    expectedBoundary.setVal(sentinel);
    dst.fillBoundary(geom, false);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const streamed = dst.multifab().const_array(mfi);
        auto const snapshot = streamedSnapshot.array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                snapshot(i, j, k, iPop) = streamed(i, j, k, iPop);
            });
    }

    auto buildExpectedBoundary = [&](amrex::Box const& boundaryBox, int rx) {
        for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box bx = mfi.tilebox() & boundaryBox;
            if (!bx.ok()) {
                continue;
            }
            auto const donorData = dst.multifab().const_array(mfi);
            auto const expectedData = expectedBoundary.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    ConstPopulationCell donor{donorData, i + rx, j, k};
                    PopulationCell boundaryCell{expectedData, i, j, k};
                    boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::velocity(
                        donor, boundaryCell,
                        amrex::GpuArray<amrex::Real, 3>{
                            prescribedVelocity[0],
                            prescribedVelocity[1],
                            prescribedVelocity[2]});
                });
        }
    };
    amrex::Box xloBox = domain;
    xloBox.setBig(0, domain.smallEnd(0));
    amrex::Box xhiBox = domain;
    xhiBox.setSmall(0, domain.bigEnd(0));
    buildExpectedBoundary(xloBox, 1);
    buildExpectedBoundary(xhiBox, -1);
    amrex::Gpu::streamSynchronize();

    // Deliberately destroy src after streaming. Domain-only Guo reconstruction
    // must continue to use synchronized dst donors rather than collided src.
    src.multifab().setVal(corruptSource);
    boundary::PostStreamingTimingSample postStreamingTiming;
    firstBoundaryStep.runPostStreaming(
        [&](amrex::Real physicalTime) {
            observedPostTimes[0] = physicalTime;
            boundarySystem.applyPostStreamingValidNodes(
                src, dst, physicalTime, &postStreamingTiming);
        });
    expect(
        std::isfinite(postStreamingTiming.domain) &&
            std::isfinite(postStreamingTiming.internalSolid) &&
            postStreamingTiming.domain >= 0.0 &&
            postStreamingTiming.internalSolid >= 0.0,
        "post-stream timing must report finite non-negative sub-stages");

    amrex::MultiFab postGuoError(ba, dm, 1, 0);
    postGuoError.setVal(0.0);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = dst.multifab().const_array(mfi);
        auto const snapshot = streamedSnapshot.const_array(mfi);
        auto const expected = expectedBoundary.const_array(mfi);
        auto const error = postGuoError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool isBoundary =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real reference = isBoundary ?
                    expected(i, j, k, iPop) : snapshot(i, j, k, iPop);
                amrex::Real diff = actual(i, j, k, iPop) - reference;
                diff = diff < 0.0 ? -diff : diff;
                maxError = diff > maxError ? diff : maxError;
            }
            error(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        postGuoError.max(0) < tolerance,
        "post-stream Guo must read dst donors, overwrite all q boundary values, "
        "and preserve streamed interior values");

    LatticeField<Descriptor>* stepSource = &src;
    LatticeField<Descriptor>* stepDestination = &dst;
    std::swap(stepSource, stepDestination);
    expect(stepSource == &dst, "Lifecycle swap must promote reconstructed dst to src");
    collision.apply(*stepSource);

    amrex::MultiFab invalidAfterSwap(ba, dm, 1, 0);
    invalidAfterSwap.setVal(0.0);
    for (amrex::MFIter mfi(stepSource->multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = stepSource->multifab().const_array(mfi);
        auto const invalid = invalidAfterSwap.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real found = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real value = f(i, j, k, iPop);
                if (value != value || value == sentinel || value == corruptSource) {
                    found = 1.0;
                }
            }
            invalid(i, j, k, 0) = found;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        invalidAfterSwap.max(0) == 0.0,
        "collision after swap must consume legal reconstructed boundary state");

    // The following step samples the next physical interval. Callbacks observe
    // the values supplied by the same production context used in the solver;
    // no time-dependent profile is added solely for this test.
    const solvers::BoundaryStepContext secondBoundaryStep(
        sourcePhysicalTime + dtPhys, dtPhys);
    secondBoundaryStep.runPreStreaming(
        [&](amrex::Real physicalTime) {
            observedPreTimes[1] = physicalTime;
            boundarySystem.fillPreStreaming(
                *stepSource, physicalTime);
        });
    stepDestination->multifab().setVal(sentinel);
    streaming.apply(*stepSource, *stepDestination);
    secondBoundaryStep.runPostStreaming(
        [&](amrex::Real physicalTime) {
            observedPostTimes[1] = physicalTime;
            boundarySystem.applyPostStreamingValidNodes(
                *stepSource, *stepDestination, physicalTime);
        });

    amrex::MultiFab secondStepInvalid(ba, dm, 1, 0);
    secondStepInvalid.setVal(0.0);
    for (amrex::MFIter mfi(
             stepDestination->multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = stepDestination->multifab().const_array(mfi);
        auto const invalid = secondStepInvalid.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real found = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    const amrex::Real value = f(i, j, k, iPop);
                    if (value != value || value == sentinel ||
                        value == corruptSource) {
                        found = 1.0;
                    }
                }
                invalid(i, j, k, 0) = found;
            });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        secondStepInvalid.max(0) == 0.0,
        "second lifecycle step must publish a legal reconstructed destination");
    expect(
        std::abs(observedPreTimes[0] - 1.25) < tolerance &&
            std::abs(observedPostTimes[0] - 1.29) < tolerance &&
            std::abs(observedPreTimes[1] - 1.29) < tolerance &&
            std::abs(observedPostTimes[1] - 1.33) < tolerance,
        "boundary lifecycle must sample physical (pre,post)=(1.25,1.29) "
        "then (1.29,1.33) for dtPhys=0.04");

    amrex::Print() << Descriptor::name()
                   << " post-stream boundary lifecycle integration tests passed.\n";
}

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real decomposition_population(
    int i,
    int j,
    int k,
    int iPop) noexcept
{
    const amrex::Real rho =
        1.0 + 7.0e-4 * static_cast<amrex::Real>(i) +
        3.0e-4 * static_cast<amrex::Real>(j) +
        2.0e-4 * static_cast<amrex::Real>(k);
    const amrex::Real ux =
        0.025 + 2.0e-4 * static_cast<amrex::Real>(j);
    const amrex::Real uy =
        -0.004 + 1.0e-4 * static_cast<amrex::Real>(i);
    const amrex::Real uz = Descriptor::d == 3 ?
        0.003 - 8.0e-5 * static_cast<amrex::Real>(k) : 0.0;
    const amrex::Real perturbation =
        1.0e-7 * static_cast<amrex::Real>((iPop % 7) - 3) *
        (1.0 + 0.01 * static_cast<amrex::Real>(i + 2 * j + 3 * k));
    return Equilibrium<Descriptor>::compute(iPop, rho, ux, uy, uz) +
        perturbation;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real decomposition_absolute_error(
    amrex::Real actual,
    amrex::Real expected) noexcept
{
    constexpr amrex::Real failureError = 1.0e30;
    if (!(actual == actual) || !(expected == expected)) {
        return failureError;
    }
    const amrex::Real difference = actual - expected;
    if (!(difference == difference)) {
        return failureError;
    }
    const amrex::Real absoluteDifference =
        difference < 0.0 ? -difference : difference;
    return absoluteDifference < failureError ? absoluteDifference : failureError;
}

inline amrex::BoxArray make_explicit_box_array(
    std::vector<amrex::Box> const& boxes)
{
    expect(!boxes.empty(), "Explicit decomposition requires at least one box");
    return amrex::BoxArray(boxes.data(), static_cast<int>(boxes.size()));
}

template <class Descriptor>
void run_guo_x_slab_decomposition_integration_case()
{
    constexpr amrex::Real sentinel = -1.23456789e30;
    constexpr amrex::Real tolerance = 5.0e-13;
    const int nx = 7;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);

    constexpr amrex::Real dx = 0.25;
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.5 * dx, 0.0, 0.0)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * dx,
            static_cast<amrex::Real>(ny) * dx,
            static_cast<amrex::Real>(nz) * dx)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 1, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    const amrex::Array<amrex::Real, 3> prescribedVelocity{0.04, 0.0, 0.0};
    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_uniform_velocity_domain_spec(
            "decomposition_xlo",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            prescribedVelocity),
        make_guo_uniform_velocity_domain_spec(
            "decomposition_xhi",
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            prescribedVelocity)};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    std::vector<amrex::Box> boxes;
    for (auto const& range : std::array<std::array<int, 2>, 5>{
             std::array<int, 2>{0, 0},
             std::array<int, 2>{1, 1},
             std::array<int, 2>{2, 4},
             std::array<int, 2>{5, 5},
             std::array<int, 2>{6, 6}}) {
        amrex::Box box = domain;
        box.setSmall(0, range[0]);
        box.setBig(0, range[1]);
        boxes.push_back(box);
    }
    amrex::BoxArray ba = make_explicit_box_array(boxes);
    expect(
        ba[0].length(0) == 1 && ba[4].length(0) == 1,
        "Physical-boundary FABs must be exactly one valid cell thick");
    const int rankCount = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<int> processorMap(5, 0);
    if (rankCount >= 2) {
        // xlo donor crosses a same-rank FAB boundary; xhi donor crosses rank 0->1.
        processorMap = amrex::Vector<int>{0, 0, 1, 0, 1};
    }
    amrex::DistributionMapping dm(processorMap);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    expect(
        boundary::post_stream_donor_fill_boundary_required(
            boundarySystem.postStreamingPlan(), ba),
        "One-cell-thick boundary FABs must require donor FillBoundary fallback");
    expect(
        dm[0] == dm[1],
        "xlo first donor must cross a same-rank FAB boundary in focused decomposition");
    if (rankCount >= 2) {
        expect(
            dm[3] != dm[4],
            "xhi first donor must cross MPI ranks in focused decomposition");
    }

    LatticeField<Descriptor> populations("x_slab_decomposition", ba, dm);
    populations.multifab().setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                if (i != domain.smallEnd(0) && i != domain.bigEnd(0)) {
                    f(i, j, k, iPop) =
                        decomposition_population<Descriptor>(i, j, k, iPop);
                }
            });
    }

    const amrex::GpuArray<amrex::Real, 3> targetVelocity{
        prescribedVelocity[0], prescribedVelocity[1], prescribedVelocity[2]};
    boundarySystem.initializeValidBoundaries(populations, 0.0);
    amrex::MultiFab initializationError(ba, dm, 1, 0);
    initializationError.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = initializationError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool boundaryNode =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            if (boundaryNode) {
                const int rx = i == domain.smallEnd(0) ? 1 : -1;
                amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    donor[iPop] = decomposition_population<Descriptor>(
                        i + rx, j, k, iPop);
                }
                const boundary::GuoDonorState donorState =
                    boundary::GuoNonEquilibriumExtrapolationScheme<
                        Descriptor>::donor_state(donor);
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] = Equilibrium<Descriptor>::compute(
                        iPop,
                        donorState.rho,
                        targetVelocity[0],
                        targetVelocity[1],
                        targetVelocity[2]);
                }
            } else {
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] =
                        decomposition_population<Descriptor>(
                            i, j, k, iPop);
                }
            }

            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real difference =
                    decomposition_absolute_error(
                        actual(i, j, k, iPop), expected[iPop]);
                maxError = difference > maxError ? difference : maxError;
            }
            err(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        initializationError.max(0) < tolerance,
        "Guo valid-boundary initialization failed same-rank or cross-rank donor synchronization");

    boundarySystem.applyPostStreamingValidNodes(
        populations, populations, 1.0);

    amrex::MultiFab error(ba, dm, 1, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool boundaryNode =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            if (boundaryNode) {
                const int rx = i == domain.smallEnd(0) ? 1 : -1;
                amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    donor[iPop] = decomposition_population<Descriptor>(
                        i + rx, j, k, iPop);
                }
                boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::velocity(
                    donor, expected, targetVelocity);
            } else {
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] =
                        decomposition_population<Descriptor>(i, j, k, iPop);
                }
            }

            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real diff = decomposition_absolute_error(
                    actual(i, j, k, iPop), expected[iPop]);
                maxError = diff > maxError ? diff : maxError;
            }
            err(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real maxError = error.max(0);
    expect(
        maxError < tolerance,
        "Guo x-slab executor failed same-rank or cross-rank donor synchronization");

    amrex::Print()
        << Descriptor::name()
        << " Guo x-slab decomposition passed: ranks=" << rankCount
        << " boundary_fab_width=1"
        << " xlo_boundary_owner=" << dm[0]
        << " xlo_donor_owner=" << dm[1]
        << " xhi_donor_owner=" << dm[3]
        << " xhi_boundary_owner=" << dm[4]
        << " max_error=" << maxError << "\n";
}

template <class Descriptor>
void run_guo_d2_diagonal_corner_decomposition_integration_case()
{
    if constexpr (Descriptor::d != 2) {
        return;
    }

    constexpr amrex::Real sentinel = -1.23456789e30;
    constexpr amrex::Real tolerance = 5.0e-13;
    const int nx = 5;
    const int ny = 5;
    const int nz = 1;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.125, -0.125, 0.0)},
        {AMREX_D_DECL(1.125, 1.125, 0.25)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    const amrex::Array<amrex::Real, 3> prescribedVelocity{0.04, 0.0, 0.0};
    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_uniform_velocity_domain_spec(
            "corner_xlo", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet, prescribedVelocity),
        make_guo_uniform_velocity_domain_spec(
            "corner_xhi", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet, prescribedVelocity),
        make_guo_no_slip_domain_spec("corner_ylo", boundary::DomainSide::YLo),
        make_guo_no_slip_domain_spec("corner_yhi", boundary::DomainSide::YHi)};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    const std::array<std::array<int, 2>, 3> ranges{
        std::array<int, 2>{0, 0},
        std::array<int, 2>{1, 3},
        std::array<int, 2>{4, 4}};
    std::vector<amrex::Box> boxes;
    for (auto const& yRange : ranges) {
        for (auto const& xRange : ranges) {
            amrex::Box box = domain;
            box.setSmall(0, xRange[0]);
            box.setBig(0, xRange[1]);
            box.setSmall(1, yRange[0]);
            box.setBig(1, yRange[1]);
            boxes.push_back(box);
        }
    }
    amrex::BoxArray ba = make_explicit_box_array(boxes);
    const int rankCount = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<int> processorMap(9, 0);
    if (rankCount >= 2) {
        for (int n = 0; n < 9; ++n) {
            processorMap[n] = n % 2;
        }
        processorMap[0] = 0;
        processorMap[4] = 1;
    }
    amrex::DistributionMapping dm(processorMap);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    expect(
        boundary::post_stream_donor_fill_boundary_required(
            boundarySystem.postStreamingPlan(), ba),
        "D2 diagonal corner donors must require FillBoundary fallback");
    if (rankCount >= 2) {
        expect(
            dm[0] != dm[4],
            "D2 (xlo,ylo) diagonal donor must cross MPI ranks");
    }

    LatticeField<Descriptor> populations("d2_corner_decomposition", ba, dm);
    populations.multifab().setVal(sentinel);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = populations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                const bool boundaryNode =
                    i == domain.smallEnd(0) || i == domain.bigEnd(0) ||
                    j == domain.smallEnd(1) || j == domain.bigEnd(1);
                if (!boundaryNode) {
                    f(i, j, k, iPop) =
                        decomposition_population<Descriptor>(i, j, k, iPop);
                }
            });
    }

    const amrex::GpuArray<amrex::Real, 3> targetVelocity{
        prescribedVelocity[0], prescribedVelocity[1], prescribedVelocity[2]};
    boundarySystem.initializeValidBoundaries(populations, 0.0);
    amrex::MultiFab initializationError(ba, dm, 1, 0);
    initializationError.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = initializationError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool xBoundary =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            const bool yBoundary =
                j == domain.smallEnd(1) || j == domain.bigEnd(1);
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            if (xBoundary || yBoundary) {
                const int rx = i == domain.smallEnd(0) ? 1 :
                    (i == domain.bigEnd(0) ? -1 : 0);
                const int ry = j == domain.smallEnd(1) ? 1 :
                    (j == domain.bigEnd(1) ? -1 : 0);
                amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    donor[iPop] = decomposition_population<Descriptor>(
                        i + rx, j + ry, k, iPop);
                }
                const boundary::GuoDonorState donorState =
                    boundary::GuoNonEquilibriumExtrapolationScheme<
                        Descriptor>::donor_state(donor);
                const amrex::GpuArray<amrex::Real, 3> velocity =
                    yBoundary ?
                        amrex::GpuArray<amrex::Real, 3>{
                            0.0, 0.0, 0.0} :
                        targetVelocity;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] = Equilibrium<Descriptor>::compute(
                        iPop,
                        donorState.rho,
                        velocity[0],
                        velocity[1],
                        velocity[2]);
                }
            } else {
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] =
                        decomposition_population<Descriptor>(
                            i, j, k, iPop);
                }
            }

            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real difference =
                    decomposition_absolute_error(
                        actual(i, j, k, iPop), expected[iPop]);
                maxError = difference > maxError ? difference : maxError;
            }
            err(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        initializationError.max(0) < tolerance,
        "D2 valid-boundary initialization diagonal donor or wall-wins mismatch");

    boundarySystem.applyPostStreamingValidNodes(
        populations, populations, 1.0);

    amrex::MultiFab error(ba, dm, 1, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(populations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool xBoundary =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            const bool yBoundary =
                j == domain.smallEnd(1) || j == domain.bigEnd(1);
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};
            if (xBoundary || yBoundary) {
                const int rx = i == domain.smallEnd(0) ? 1 :
                    (i == domain.bigEnd(0) ? -1 : 0);
                const int ry = j == domain.smallEnd(1) ? 1 :
                    (j == domain.bigEnd(1) ? -1 : 0);
                amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    donor[iPop] = decomposition_population<Descriptor>(
                        i + rx, j + ry, k, iPop);
                }
                if (yBoundary) {
                    boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::no_slip(
                        donor, expected);
                } else {
                    boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::velocity(
                        donor, expected, targetVelocity);
                }
            } else {
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    expected[iPop] =
                        decomposition_population<Descriptor>(i, j, k, iPop);
                }
            }

            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real diff = decomposition_absolute_error(
                    actual(i, j, k, iPop), expected[iPop]);
                maxError = diff > maxError ? diff : maxError;
            }
            err(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real maxError = error.max(0);
    expect(
        maxError < tolerance,
        "D2 diagonal corner donor or wall-wins reconstruction mismatch");

    amrex::Print()
        << Descriptor::name()
        << " Guo diagonal-corner decomposition passed: ranks=" << rankCount
        << " corner_owner=" << dm[0]
        << " diagonal_donor_owner=" << dm[4]
        << " max_error=" << maxError << "\n";
}

template <class Descriptor>
void run_guo_d3_periodic_donor_decomposition_integration_case()
{
    if constexpr (Descriptor::d != 3) {
        return;
    }

    constexpr amrex::Real sentinel = -1.23456789e30;
    constexpr amrex::Real tolerance = 5.0e-13;
    const int nx = 7;
    const int ny = 4;
    const int nz = 4;
    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.125, 0.0, 0.0)},
        {AMREX_D_DECL(1.625, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 1, 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    const amrex::Array<amrex::Real, 3> prescribedVelocity{0.04, 0.0, 0.0};
    std::vector<boundary::DomainBoundarySpec> specs{
        make_guo_uniform_velocity_domain_spec(
            "periodic_donor_xlo", boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet, prescribedVelocity),
        make_guo_uniform_velocity_domain_spec(
            "periodic_donor_xhi", boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet, prescribedVelocity)};
    boundary::BoundarySystem<Descriptor> boundarySystem;
    const std::array<std::array<int, 2>, 3> xRanges{
        std::array<int, 2>{0, 0},
        std::array<int, 2>{1, 5},
        std::array<int, 2>{6, 6}};
    const std::array<std::array<int, 2>, 3> zRanges{
        std::array<int, 2>{0, 0},
        std::array<int, 2>{1, 2},
        std::array<int, 2>{3, 3}};
    std::vector<amrex::Box> boxes;
    for (auto const& zRange : zRanges) {
        for (auto const& xRange : xRanges) {
            amrex::Box box = domain;
            box.setSmall(0, xRange[0]);
            box.setBig(0, xRange[1]);
            box.setSmall(2, zRange[0]);
            box.setBig(2, zRange[1]);
            boxes.push_back(box);
        }
    }
    amrex::BoxArray ba = make_explicit_box_array(boxes);
    const int rankCount = amrex::ParallelDescriptor::NProcs();
    amrex::Vector<int> processorMap(9, 0);
    if (rankCount >= 2) {
        for (int n = 0; n < 9; ++n) {
            processorMap[n] = n % 2;
        }
        processorMap[1] = 0;
        processorMap[7] = 1;
    }
    amrex::DistributionMapping dm(processorMap);
    define_boundary_system_for_test(boundarySystem, specs, geom, ba, dm);
    if (rankCount >= 2) {
        expect(
            dm[1] != dm[7],
            "D3 z-periodic source and wrapped donor FABs must cross MPI ranks");
    }

    LatticeField<Descriptor> src("d3_periodic_source", ba, dm);
    LatticeField<Descriptor> dst("d3_periodic_destination", ba, dm);
    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = src.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                f(i, j, k, iPop) =
                    decomposition_population<Descriptor>(i, j, k, iPop);
            });
    }
    boundarySystem.fillPreStreaming(src, 1.0);
    dst.multifab().setVal(sentinel);
    operators::PullStreamingOperator<Descriptor> streaming;
    streaming.apply(src, dst);
    boundarySystem.applyPostStreamingValidNodes(src, dst, 1.0);

    const amrex::GpuArray<amrex::Real, 3> targetVelocity{
        prescribedVelocity[0], prescribedVelocity[1], prescribedVelocity[2]};
    amrex::MultiFab error(ba, dm, 1, 0);
    error.setVal(0.0);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = dst.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const bool boundaryNode =
                i == domain.smallEnd(0) || i == domain.bigEnd(0);
            const int rx = i == domain.smallEnd(0) ? 1 :
                (i == domain.bigEnd(0) ? -1 : 0);
            amrex::GpuArray<amrex::Real, Descriptor::q> expected{};

            if (boundaryNode) {
                amrex::GpuArray<amrex::Real, Descriptor::q> streamedDonor{};
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    const int si = i + rx - Descriptor::cx(iPop);
                    int sj = j - Descriptor::cy(iPop);
                    int sk = k - Descriptor::cz(iPop);
                    sj = sj < domain.smallEnd(1) ? sj + domain.length(1) : sj;
                    sj = sj > domain.bigEnd(1) ? sj - domain.length(1) : sj;
                    sk = sk < domain.smallEnd(2) ? sk + domain.length(2) : sk;
                    sk = sk > domain.bigEnd(2) ? sk - domain.length(2) : sk;
                    streamedDonor[iPop] = decomposition_population<Descriptor>(
                        si, sj, sk, iPop);
                }
                boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>::velocity(
                    streamedDonor, expected, targetVelocity);
            } else {
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    const int si = i - Descriptor::cx(iPop);
                    int sj = j - Descriptor::cy(iPop);
                    int sk = k - Descriptor::cz(iPop);
                    sj = sj < domain.smallEnd(1) ? sj + domain.length(1) : sj;
                    sj = sj > domain.bigEnd(1) ? sj - domain.length(1) : sj;
                    sk = sk < domain.smallEnd(2) ? sk + domain.length(2) : sk;
                    sk = sk > domain.bigEnd(2) ? sk - domain.length(2) : sk;
                    expected[iPop] = decomposition_population<Descriptor>(
                        si, sj, sk, iPop);
                }
            }

            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real diff = decomposition_absolute_error(
                    actual(i, j, k, iPop), expected[iPop]);
                maxError = diff > maxError ? diff : maxError;
            }
            err(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real maxError = error.max(0);
    expect(
        maxError < tolerance,
        "D3 periodic streamed donor reconstruction mismatch");

    amrex::Print()
        << Descriptor::name()
        << " Guo z-periodic donor decomposition passed: ranks=" << rankCount
        << " zlo_donor_owner=" << dm[1]
        << " zhi_wrapped_source_owner=" << dm[7]
        << " max_error=" << maxError << "\n";
}

template <class Descriptor>
void run_post_stream_boundary_decomposition_integration_tests()
{
    run_guo_x_slab_decomposition_integration_case<Descriptor>();
    run_guo_d2_diagonal_corner_decomposition_integration_case<Descriptor>();
    run_guo_d3_periodic_donor_decomposition_integration_case<Descriptor>();
    amrex::Print() << Descriptor::name()
                   << " post-stream boundary decomposition integration tests passed.\n";
}

template <class Descriptor>
void run_field_integration_tests()
{
    const int nx = 8;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;

    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);
    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    amrex::DistributionMapping dm(ba);
    expect(ba.size() >= 2, "Field integration layout must contain multiple FABs");

    LatticeField<Descriptor> populations("integration_populations", ba, dm);
    expect(populations.nComp() == Descriptor::q, "LatticeField nComp must match descriptor q");
    expect(populations.nGrow() == Descriptor::vicinity, "LatticeField default ghost width mismatch");
    expect(
        std::string_view(populations.descriptorName()) == std::string_view(Descriptor::name()),
        "LatticeField descriptor name mismatch");

    ScalarField rho("rho", ba, dm);
    VectorField velocity("velocity", ba, dm);
    MacroField macro("macro", ba, dm);
    MaskField mask("mask", ba, dm);

    expect(
        populations.multifab().boxArray() == ba &&
            rho.multifab().boxArray() == ba &&
            velocity.multifab().boxArray() == ba &&
            macro.multifab().boxArray() == ba &&
            mask.multifab().boxArray() == ba &&
            populations.multifab().DistributionMap() == dm &&
            rho.multifab().DistributionMap() == dm &&
            mask.multifab().DistributionMap() == dm,
        "LBM fields must preserve the supplied BoxArray and DistributionMapping");

    expect(rho.nComp() == 1, "ScalarField must have one component");
    expect(velocity.nComp() == AMREX_SPACEDIM, "VectorField component count mismatch");
    expect(macro.nComp() == MacroField::NumComp, "MacroField component count mismatch");
    expect(mask.nComp() == 1, "MaskField must have one component");
    expect(MacroField::Rho == 0 && MacroField::Ux == 1 &&
           MacroField::Uy == 2 && MacroField::Uz == 3,
           "MacroField component ordering mismatch");

    rho.multifab().setVal(1.0);
    velocity.multifab().setVal(0.0);
    macro.multifab().setVal(0.0);
    mask.multifab().setVal(1);
    expect(
        rho.multifab().min(0) == 1.0 && rho.multifab().max(0) == 1.0 &&
            velocity.multifab().max(0) == 0.0 &&
            macro.multifab().max(0) == 0.0 &&
            mask.multifab().min(0) == 1 && mask.multifab().max(0) == 1,
        "MultiFab/iMultiFab values must remain observable across the distributed layout");

    amrex::Print() << Descriptor::name() << " field integration tests passed.\n";
}


template <class Descriptor>
void run_operator_integration_tests()
{
    const int nx = 4;
    const int ny = 4;
    const int nz = Descriptor::d == 3 ? 4 : 1;

    amrex::IntVect domLo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect domHi(AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box domain(domLo, domHi);

    amrex::RealBox realBox(
        {AMREX_D_DECL(0.0, 0.0, 0.0)},
        {AMREX_D_DECL(1.0, 1.0, 1.0)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Geometry geom(domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    const amrex::Real collisionOmega = 0.7;
    BGKDynamics<Descriptor> dynamics(collisionOmega);
    operators::CollisionOperator<Descriptor, BGKDynamics<Descriptor>> collision(dynamics);
    operators::PullStreamingOperator<Descriptor> streaming;
    operators::MacroscopicOperator<Descriptor> macroscopic;
    ConstantPopulationInitializationProfile initializationProfile{
        initialization::MacroscopicState{1.125, 0.02, -0.01, 0.0}};
    operators::EquilibriumPopulationInitializationOperator<
        Descriptor,
        ConstantPopulationInitializationProfile> initialization(initializationProfile, geom);

    expect(collision.name() == std::string_view("collision"), "CollisionOperator name mismatch");
    expect(collision.scope() == operators::OperatorScope::Level, "CollisionOperator scope mismatch");
    expect(streaming.name() == std::string_view("pull_streaming"), "PullStreamingOperator name mismatch");
    expect(streaming.scope() == operators::OperatorScope::Level, "PullStreamingOperator scope mismatch");
    expect(macroscopic.name() == std::string_view("macroscopic"), "MacroscopicOperator name mismatch");
    expect(macroscopic.scope() == operators::OperatorScope::Level, "MacroscopicOperator scope mismatch");
    expect(
        initialization.name() == std::string_view("equilibrium_population_initialization"),
        "EquilibriumPopulationInitializationOperator name mismatch");
    expect(
        initialization.scope() == operators::OperatorScope::Level,
        "EquilibriumPopulationInitializationOperator scope mismatch");

    amrex::BoxArray ba(domain);
    ba.maxSize(4);
    amrex::DistributionMapping dm(ba);

    LatticeField<Descriptor> src(ba, dm);
    LatticeField<Descriptor> dst(ba, dm);
    MacroField macro("operator_macro", ba, dm);
    LatticeField<Descriptor> initialized("initialized_populations", ba, dm);
    MacroField initializedMacro("initialized_macro", ba, dm);
    LatticeField<Descriptor> ghostInitialized("ghost_initialized_populations", ba, dm);

    initialization.apply(initialized, 0.0);
    macroscopic.apply(initialized, initializedMacro);

    amrex::MultiFab initializationErr(ba, dm, 1, 0);
    const auto expectedInitState = initializationProfile.state;
    for (amrex::MFIter mfi(initialized.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = initialized.multifab().const_array(mfi);
        auto const m = initializedMacro.multifab().const_array(mfi);
        auto const err = initializationErr.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expected =
                    Equilibrium<Descriptor>::compute(
                        iPop,
                        expectedInitState.rho,
                        expectedInitState.ux,
                        expectedInitState.uy,
                        expectedInitState.uz);
                amrex::Real diff = f(i, j, k, iPop) - expected;
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }

            amrex::Real e0 = m(i, j, k, MacroField::Rho) - expectedInitState.rho;
            amrex::Real e1 = m(i, j, k, MacroField::Ux) - expectedInitState.ux;
            amrex::Real e2 = m(i, j, k, MacroField::Uy) - expectedInitState.uy;
            amrex::Real e3 = m(i, j, k, MacroField::Uz) - expectedInitState.uz;
            e0 = e0 < 0.0 ? -e0 : e0;
            e1 = e1 < 0.0 ? -e1 : e1;
            e2 = e2 < 0.0 ? -e2 : e2;
            e3 = e3 < 0.0 ? -e3 : e3;
            maxErr = e0 > maxErr ? e0 : maxErr;
            maxErr = e1 > maxErr ? e1 : maxErr;
            maxErr = e2 > maxErr ? e2 : maxErr;
            maxErr = e3 > maxErr ? e3 : maxErr;
            err(i, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        initializationErr.max(0) < 1.0e-14,
        "EquilibriumPopulationInitializationOperator mismatch");

    CoordinatePopulationInitializationProfile ghostProfile;
    operators::EquilibriumPopulationInitializationOperator<
        Descriptor,
        CoordinatePopulationInitializationProfile> ghostInitialization(ghostProfile, geom);
    ghostInitialization.apply(ghostInitialized, 0.0);

    amrex::MultiFab ghostErr(ba, dm, 1, Descriptor::vicinity);
    ghostErr.setVal(0.0);
    for (amrex::MFIter mfi(ghostInitialized.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& grown = mfi.fabbox();
        amrex::Box const& valid = mfi.validbox();
        auto const f = ghostInitialized.multifab().const_array(mfi);
        auto const err = ghostErr.array(mfi);
        amrex::ParallelFor(grown, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (valid.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) {
                return;
            }
            const int wi = (i + nx) % nx;
            const int wj = (j + ny) % ny;
            const int wk = (k + nz) % nz;
            const amrex::GpuArray<
                amrex::Real,
                AMREX_SPACEDIM> position{
                AMREX_D_DECL(
                    (static_cast<amrex::Real>(wi) + 0.5) /
                        static_cast<amrex::Real>(nx),
                    (static_cast<amrex::Real>(wj) + 0.5) /
                        static_cast<amrex::Real>(ny),
                    (static_cast<amrex::Real>(wk) + 0.5) /
                        static_cast<amrex::Real>(nz))};
            const auto expectedState = ghostProfile(position, 0.0);

            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expected =
                    Equilibrium<Descriptor>::compute(
                        iPop,
                        expectedState.rho,
                        expectedState.ux,
                        expectedState.uy,
                        expectedState.uz);
                amrex::Real diff = f(i, j, k, iPop) - expected;
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        ghostErr.max(0, Descriptor::vicinity) < 1.0e-14,
        "EquilibriumPopulationInitializationOperator ghost fill mismatch");

    const amrex::Real rho0 = 1.2;
    const amrex::Real ux0 = 0.015;
    const amrex::Real uy0 = -0.01;
    const amrex::Real uz0 = Descriptor::d == 3 ? 0.012 : 0.0;

    amrex::MultiFab expectedPostCollision(ba, dm, Descriptor::q, 0);
    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = src.multifab().array(mfi);
        auto const expected = expectedPostCollision.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::GpuArray<amrex::Real, Descriptor::q> local{};
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real perturb =
                    1.0e-4 * static_cast<amrex::Real>((iPop % 5) - 2) +
                    1.0e-6 * static_cast<amrex::Real>(i + 2 * j + 3 * k);
                local[iPop] =
                    Equilibrium<Descriptor>::compute(iPop, rho0, ux0, uy0, uz0) +
                    perturb;
                f(i, j, k, iPop) = local[iPop];
            }

            const amrex::Real rho = Moments<Descriptor>::rho(local);
            amrex::Real ux = 0.0;
            amrex::Real uy = 0.0;
            amrex::Real uz = 0.0;
            Moments<Descriptor>::velocity(local, rho, ux, uy, uz);

            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real feq =
                    Equilibrium<Descriptor>::compute(iPop, rho, ux, uy, uz);
                expected(i, j, k, iPop) =
                    local[iPop] - collisionOmega * (local[iPop] - feq);
            }
        });
    }

    collision.apply(src);

    amrex::MultiFab collisionErr(ba, dm, 1, 0);
    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = src.multifab().const_array(mfi);
        auto const expected = expectedPostCollision.const_array(mfi);
        auto const err = collisionErr.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                amrex::Real diff = f(i, j, k, iPop) - expected(i, j, k, iPop);
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(collisionErr.max(0) < 1.0e-14, "CollisionOperator population mismatch");

    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = src.multifab().array(mfi);
        amrex::ParallelFor(bx, Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) {
                f(i, j, k, iPop) =
                    Equilibrium<Descriptor>::compute(iPop, rho0, ux0, uy0, uz0);
            });
    }

    macroscopic.apply(src, macro);

    amrex::MultiFab macroErr(ba, dm, 1, 0);
    for (amrex::MFIter mfi(macro.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const m = macro.multifab().const_array(mfi);
        auto const err = macroErr.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real e0 = m(i, j, k, MacroField::Rho) - rho0;
            amrex::Real e1 = m(i, j, k, MacroField::Ux) - ux0;
            amrex::Real e2 = m(i, j, k, MacroField::Uy) - uy0;
            amrex::Real e3 = m(i, j, k, MacroField::Uz) - uz0;
            e0 = e0 < 0.0 ? -e0 : e0;
            e1 = e1 < 0.0 ? -e1 : e1;
            e2 = e2 < 0.0 ? -e2 : e2;
            e3 = e3 < 0.0 ? -e3 : e3;
            amrex::Real e = e0 > e1 ? e0 : e1;
            e = e > e2 ? e : e2;
            e = e > e3 ? e : e3;
            err(i, j, k, 0) = e;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(macroErr.max(0) < 1.0e-14, "Collision/Macroscopic operator mismatch");

    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = src.multifab().array(mfi);
        amrex::ParallelFor(bx, Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) {
                f(i, j, k, iPop) =
                    100000.0 * iPop + 1000.0 * i + 10.0 * j + k;
            });
    }

    src.fillBoundary(geom);
    streaming.apply(src, dst);

    amrex::MultiFab streamErr(ba, dm, 1, 0);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = dst.multifab().const_array(mfi);
        auto const err = streamErr.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real maxErr = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const int si = (i - Descriptor::cx(iPop) + nx) % nx;
                const int sj = (j - Descriptor::cy(iPop) + ny) % ny;
                const int sk = (k - Descriptor::cz(iPop) + nz) % nz;
                const amrex::Real expected =
                    100000.0 * iPop + 1000.0 * si + 10.0 * sj + sk;
                amrex::Real diff = f(i, j, k, iPop) - expected;
                diff = diff < 0.0 ? -diff : diff;
                maxErr = diff > maxErr ? diff : maxErr;
            }
            err(i, j, k, 0) = maxErr;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(streamErr.max(0) < 1.0e-14, "PullStreamingOperator mismatch");

    amrex::Print() << Descriptor::name() << " operator integration tests passed.\n";
}


} // namespace lbm::tests::integration

#endif
