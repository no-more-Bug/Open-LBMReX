#ifndef LBM_TESTS_UNIT_LBM_BOUNDARYFORMULATESTS_H
#define LBM_TESTS_UNIT_LBM_BOUNDARYFORMULATESTS_H

#include "unit/TestAssertions.H"

#include "lbm/boundary/core/DomainBoundaryCompatibility.H"
#include "lbm/boundary/domain/DomainBoundaryPrescription.H"
#include "lbm/boundary/internal/InternalPopulationReconstruction.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/boundary/profiles/BuiltinProfiles.H"
#include "lbm/boundary/schemes/GuoNonEquilibriumExtrapolationScheme.H"
#include "lbm/boundary/schemes/HalfwayBounceBackScheme.H"
#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <limits>

namespace lbm::tests {

inline void test_domain_boundary_compatibility()
{
    using boundary::BoundaryPhysical;
    using boundary::BoundaryProfileKind;
    using boundary::BoundaryScheme;
    struct Combination {
        BoundaryPhysical physical;
        BoundaryScheme scheme;
        BoundaryProfileKind profile;
    };
    constexpr std::array physicals{
        BoundaryPhysical::NoSlipWall,
        BoundaryPhysical::MovingWall,
        BoundaryPhysical::VelocityInlet,
        BoundaryPhysical::VelocityOutlet,
        BoundaryPhysical::PressureInlet,
        BoundaryPhysical::PressureOutlet,
        BoundaryPhysical::PrescribedState,
        BoundaryPhysical::Outflow,
        BoundaryPhysical::FreeOutflow,
        BoundaryPhysical::Symmetry};
    constexpr std::array schemes{
        BoundaryScheme::HalfwayBounceBack,
        BoundaryScheme::AntiBounceBack,
        BoundaryScheme::Bouzidi,
        BoundaryScheme::Yms,
        BoundaryScheme::ZouHe,
        BoundaryScheme::GuoExtrapolation};
    constexpr std::array profiles{
        BoundaryProfileKind::None,
        BoundaryProfileKind::Constant,
        BoundaryProfileKind::Uniform,
        BoundaryProfileKind::Poiseuille,
        BoundaryProfileKind::UserDef};
    constexpr std::array<Combination, 25> supported{{
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::None},
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Constant},
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::UserDef},
        {BoundaryPhysical::MovingWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::None},
        {BoundaryPhysical::MovingWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Constant},
        {BoundaryPhysical::MovingWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::MovingWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::MovingWall, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::UserDef},
        {BoundaryPhysical::VelocityInlet, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::VelocityInlet, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::VelocityOutlet, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::VelocityOutlet, BoundaryScheme::HalfwayBounceBack, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::NoSlipWall, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::None},
        {BoundaryPhysical::MovingWall, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::VelocityInlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::VelocityInlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::VelocityOutlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::VelocityOutlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::PressureInlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Constant},
        {BoundaryPhysical::PressureOutlet, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Constant},
        {BoundaryPhysical::PrescribedState, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Uniform},
        {BoundaryPhysical::PrescribedState, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::Poiseuille},
        {BoundaryPhysical::FreeOutflow, BoundaryScheme::GuoExtrapolation, BoundaryProfileKind::None}}};
    for (auto physical : physicals) {
        for (auto scheme : schemes) {
            for (auto profile : profiles) {
                bool expected = false;
                for (auto const& combination : supported) {
                    expected = expected ||
                        (combination.physical == physical &&
                         combination.scheme == scheme &&
                         combination.profile == profile);
                }
                require(
                    boundary::is_supported_domain_boundary_combination(
                        physical, scheme, profile) == expected,
                    "Domain physical/scheme/profile compatibility matrix changed");
            }
        }
    }
}

template <class Descriptor>
void test_hbb_and_profile_formulas()
{
    const amrex::GpuArray<amrex::Real, 3> wallVelocity{
        0.1, -0.02, Descriptor::d == 3 ? 0.01 : 0.0};
    constexpr amrex::Real rho = 1.07;
    for (int population = 0; population < Descriptor::q; ++population) {
        const amrex::Real cDotU =
            Descriptor::cx(population) * wallVelocity[0] +
            Descriptor::cy(population) * wallVelocity[1] +
            Descriptor::cz(population) * wallVelocity[2];
        const amrex::Real expected = 2.0 * Descriptor::w(population) * rho *
            Descriptor::invCs2() * cDotU;
        require(
            std::abs(
                boundary::HalfwayBounceBackScheme<Descriptor>::
                    velocity_correction(population, rho, wallVelocity) -
                expected) < 1.0e-15,
            "HBB velocity correction formula changed");
    }
    require(
        boundary::prescribed_normal_velocity_sign(
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet) == 1 &&
        boundary::prescribed_normal_velocity_sign(
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityInlet) == -1 &&
        boundary::prescribed_normal_velocity_sign(
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityOutlet) == -1 &&
        boundary::prescribed_normal_velocity_sign(
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet) == 1,
        "Prescribed normal inlet/outlet direction changed");

    const boundary::PoiseuilleVelocityProfile profile{
        1,
        amrex::GpuArray<amrex::Real, 3>{0.1, -0.02, 0.01},
        -0.5,
        2.0};
    const auto lo = profile(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(8.0, -0.5, 3.0)},
        9.0);
    const auto center = profile(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(-4.0, 0.5, -7.0)},
        2.0);
    const auto hi = profile(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(2.0, 1.5, 5.0)},
        4.0);
    require(
        lo[0] == 0.0 && lo[1] == 0.0 && hi[0] == 0.0 && hi[1] == 0.0 &&
            std::abs(center[0] - 0.1) < 1.0e-15 &&
            std::abs(center[1] + 0.02) < 1.0e-15 &&
            std::abs(center[2] - 0.01) < 1.0e-15,
        "Poiseuille profile must depend only on the resolved profile coordinate");
}

template <class BaseDescriptor>
struct GuoDescriptorWithoutOpposite {
    static constexpr int d = BaseDescriptor::d;
    static constexpr int q = BaseDescriptor::q;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cx(int population) noexcept
    {
        return BaseDescriptor::cx(population);
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cy(int population) noexcept
    {
        return BaseDescriptor::cy(population);
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr int cz(int population) noexcept
    {
        return BaseDescriptor::cz(population);
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real w(int population) noexcept
    {
        return BaseDescriptor::w(population);
    }
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static constexpr amrex::Real invCs2() noexcept
    {
        return BaseDescriptor::invCs2();
    }
};

template <class Descriptor>
void test_guo_local_reconstruction()
{
    using Scheme = boundary::GuoNonEquilibriumExtrapolationScheme<Descriptor>;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real sentinel = -1234567.0;
    constexpr amrex::Real donorRho = 1.07;
    const amrex::GpuArray<amrex::Real, 3> donorVelocity{
        0.031, -0.017, Descriptor::d == 3 ? 0.013 : 0.0};
    amrex::GpuArray<amrex::Real, Descriptor::q> donor{};
    for (int population = 0; population < Descriptor::q; ++population) {
        const amrex::Real sign = population % 2 == 0 ? 1.0 : -1.0;
        donor[population] = Equilibrium<Descriptor>::compute(
            population,
            donorRho,
            donorVelocity[0],
            donorVelocity[1],
            donorVelocity[2]) +
            2.0e-6 * sign *
                static_cast<amrex::Real>((population + 1) * (population + 2));
    }
    const auto donorState = Scheme::donor_state(donor);
    amrex::Real expectedRho = 0.0;
    amrex::Real expectedJ[3]{0.0, 0.0, 0.0};
    for (int population = 0; population < Descriptor::q; ++population) {
        expectedRho += donor[population];
        expectedJ[0] += Descriptor::cx(population) * donor[population];
        expectedJ[1] += Descriptor::cy(population) * donor[population];
        expectedJ[2] += Descriptor::cz(population) * donor[population];
    }
    require(
        std::abs(donorState.rho - expectedRho) < tolerance &&
            std::abs(donorState.ux - expectedJ[0] / expectedRho) < tolerance &&
            std::abs(donorState.uy - expectedJ[1] / expectedRho) < tolerance &&
            std::abs(donorState.uz - expectedJ[2] / expectedRho) < tolerance,
        "Guo donor-state moments changed");

    const amrex::Real targetRho = 0.93;
    const amrex::GpuArray<amrex::Real, 3> targetVelocity{
        0.046, -0.009, Descriptor::d == 3 ? 0.007 : 0.0};
    amrex::GpuArray<amrex::Real, Descriptor::q> reconstructed{};
    Scheme::reconstruct(donor, reconstructed, targetRho, targetVelocity);
    for (int population = 0; population < Descriptor::q; ++population) {
        const amrex::Real donorEquilibrium = Equilibrium<Descriptor>::compute(
            population,
            donorState.rho,
            donorState.ux,
            donorState.uy,
            donorState.uz);
        const amrex::Real targetEquilibrium = Equilibrium<Descriptor>::compute(
            population,
            targetRho,
            targetVelocity[0],
            targetVelocity[1],
            targetVelocity[2]);
        require(
            std::abs(
                reconstructed[population] - targetEquilibrium -
                (donor[population] - donorEquilibrium)) < tolerance,
            "Guo reconstruction must preserve same-population non-equilibrium");
    }

    for (auto& value : reconstructed) {
        value = sentinel;
    }
    Scheme::no_slip(donor, reconstructed);
    amrex::Real rho = Moments<Descriptor>::rho(reconstructed);
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
    Moments<Descriptor>::velocity(reconstructed, rho, ux, uy, uz);
    require(
        std::abs(rho - donorState.rho) < tolerance &&
            std::abs(ux) < tolerance && std::abs(uy) < tolerance &&
            std::abs(uz) < tolerance,
        "Guo no-slip reconstructed moments changed");

    Scheme::velocity(donor, reconstructed, targetVelocity);
    rho = Moments<Descriptor>::rho(reconstructed);
    Moments<Descriptor>::velocity(reconstructed, rho, ux, uy, uz);
    require(
        std::abs(rho - donorState.rho) < tolerance &&
            std::abs(ux - targetVelocity[0]) < tolerance &&
            std::abs(uy - targetVelocity[1]) < tolerance &&
            std::abs(uz - targetVelocity[2]) < tolerance,
        "Guo velocity reconstructed moments changed");
    const auto velocityState = reconstructed;
    Scheme::moving_wall(donor, reconstructed, targetVelocity);
    for (int population = 0; population < Descriptor::q; ++population) {
        require(
            std::abs(reconstructed[population] - velocityState[population]) < tolerance,
            "Guo moving-wall and velocity full-q semantics diverged");
    }

    constexpr amrex::Real pressureRho = 0.97;
    Scheme::pressure(donor, reconstructed, pressureRho);
    const auto pressureState = reconstructed;
    rho = Moments<Descriptor>::rho(reconstructed);
    Moments<Descriptor>::velocity(reconstructed, rho, ux, uy, uz);
    require(
        std::abs(rho - pressureRho) < tolerance &&
            std::abs(ux - donorState.ux) < tolerance &&
            std::abs(uy - donorState.uy) < tolerance &&
            std::abs(uz - donorState.uz) < tolerance,
        "Guo pressure reconstructed moments changed");
    Scheme::prescribed_state(
        donor, reconstructed, targetRho, targetVelocity);
    rho = Moments<Descriptor>::rho(reconstructed);
    Moments<Descriptor>::velocity(reconstructed, rho, ux, uy, uz);
    require(
        std::abs(rho - targetRho) < tolerance &&
            std::abs(ux - targetVelocity[0]) < tolerance &&
            std::abs(uy - targetVelocity[1]) < tolerance &&
            std::abs(uz - targetVelocity[2]) < tolerance,
        "Guo prescribed-state reconstructed moments changed");
    Scheme::free_outflow_copy(donor, reconstructed);
    for (int population = 0; population < Descriptor::q; ++population) {
        require(
            reconstructed[population] == donor[population],
            "Guo free-outflow must remain an exact full-q copy");
    }

    using NoOpposite = GuoDescriptorWithoutOpposite<Descriptor>;
    amrex::GpuArray<amrex::Real, Descriptor::q> noOpposite{};
    boundary::GuoNonEquilibriumExtrapolationScheme<NoOpposite>::pressure(
        donor, noOpposite, pressureRho);
    for (int population = 0; population < Descriptor::q; ++population) {
        require(
            std::abs(noOpposite[population] - pressureState[population]) < tolerance,
            "Guo full-q reconstruction must not depend on opposite mapping");
    }
}

template <class Descriptor>
void test_internal_reconstruction_and_force_values()
{
    const boundary::InternalPopulationDonors donors{1.25, -0.75, 2.5};
    const std::array<amrex::Real, 5> qValues{
        0.0, 0.25, 0.5, 0.75, std::nextafter(1.0, 0.0)};
    for (auto scheme :
         {boundary::BoundaryScheme::Bouzidi, boundary::BoundaryScheme::Yms}) {
        for (amrex::Real q : qValues) {
            boundary::InternalBoundaryExecutionRecord record;
            record.scheme = scheme;
            record.qFromAnchor = q;
            const amrex::Real actual =
                boundary::reconstruct_internal_population(record, donors);
            const amrex::Real expected =
                scheme == boundary::BoundaryScheme::Bouzidi ?
                (q < 0.5 ?
                     2.0 * q * donors.anchorForward +
                         (1.0 - 2.0 * q) * donors.nextFluidForward :
                     donors.anchorForward / (2.0 * q) +
                         (2.0 * q - 1.0) * donors.anchorOpposite / (2.0 * q)) :
                (q * donors.anchorOpposite + q * donors.anchorForward +
                 (1.0 - q) * donors.nextFluidForward) /
                    (1.0 + q);
            require(
                std::abs(actual - expected) <=
                    64.0 * std::numeric_limits<amrex::Real>::epsilon() *
                        (1.0 + std::abs(expected)),
                "Bouzidi/YMS scalar reconstruction formula changed");
        }
    }

    int axis = 0;
    int faceDiagonal = 0;
    int xyz = 0;
    for (int direction = 1; direction < Descriptor::q; ++direction) {
        const amrex::Real outgoing = 0.25 + 0.031 * direction;
        const amrex::Real returned = 0.75 + 0.017 * direction;
        const auto force = boundary::static_internal_solid_link_force<Descriptor>(
            outgoing, returned, direction);
        for (int dimension = 0; dimension < 3; ++dimension) {
            const amrex::Real expected = (outgoing + returned) *
                static_cast<amrex::Real>(Descriptor::c(direction, dimension));
            require(
                std::abs(force[dimension] - expected) < 1.0e-14,
                "Static internal-solid link-force formula or sign changed");
        }
        const int norm = Descriptor::cNormSqr(direction);
        axis += norm == 1;
        faceDiagonal += norm == 2;
        xyz += norm == 3;
    }
    require(
        axis > 0 && faceDiagonal > 0 && (Descriptor::d == 2 || xyz > 0),
        "Internal-solid force direction coverage is incomplete");

    const boundary::BoundaryId boundaryId{7};
    const boundary::ObjectId objectId{3};
    const boundary::InternalObjectForceSample sample{
        {1, 11, 6, 0.4, 0.5},
        {{objectId, {1.0, -2.0, 0.5}}}};
    require(
        boundaryId.value == 7 && objectId.value == 3 &&
            sample.step.populationStep == 11 && sample.step.level == 1 && sample.step.coarsestStep == 6 &&
            std::abs(sample.step.samplePhysicalTime()-0.45) < 1.0e-15 &&
            sample.objectForces.size() == 1 &&
            sample.objectForces.front().objectId.value == 3,
        "BoundaryId/ObjectId value and sample semantics changed");
}

template <class Descriptor>
void run_boundary_formula_tests()
{
    test_domain_boundary_compatibility();
    test_hbb_and_profile_formulas<Descriptor>();
    test_guo_local_reconstruction<Descriptor>();
    test_internal_reconstruction_and_force_values<Descriptor>();
    amrex::Print() << Descriptor::name()
                   << " boundary formula unit tests passed.\n";
}

} // namespace lbm::tests

#endif
