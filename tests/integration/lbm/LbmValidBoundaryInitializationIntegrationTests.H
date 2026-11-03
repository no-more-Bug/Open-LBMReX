#ifndef LBM_TESTS_INTEGRATION_LBM_LBMVALIDBOUNDARYINITIALIZATIONINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_LBMVALIDBOUNDARYINITIALIZATIONINTEGRATIONTESTS_H

#include "lbm/boundary/domain/LbmValidBoundaryInitializationOperator.H"
#include "integration/lbm/LbmDomainIntegrationTests.H"

namespace lbm::tests::integration {

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
boundary::initialization::ValidBoundaryMacroscopicState
valid_boundary_initialization_bulk_state(int i, int j, int k) noexcept
{
    return boundary::initialization::ValidBoundaryMacroscopicState{
        1.0 + 7.0e-4 * static_cast<amrex::Real>(i) +
            3.0e-4 * static_cast<amrex::Real>(j) +
            2.0e-4 * static_cast<amrex::Real>(k),
        0.018 + 2.0e-4 * static_cast<amrex::Real>(j),
        -0.006 + 1.0e-4 * static_cast<amrex::Real>(i),
        Descriptor::d == 3 ?
            0.003 - 8.0e-5 * static_cast<amrex::Real>(k) : 0.0};
}

struct ValidBoundaryInitializationExpectedFace {
    boundary::BoundaryPhysical physical =
        boundary::BoundaryPhysical::NoSlipWall;
    boundary::BoundaryProfileKind profile =
        boundary::BoundaryProfileKind::None;
    amrex::GpuArray<amrex::Real, 3> velocity{0.0, 0.0, 0.0};
    amrex::Real rho = 0.0;
    amrex::Real uMax = 0.0;
};

inline ValidBoundaryInitializationExpectedFace
make_valid_boundary_initialization_expected_face(
    boundary::DomainBoundarySpec const& spec)
{
    return ValidBoundaryInitializationExpectedFace{
        spec.physical,
        spec.profile.value_or(boundary::BoundaryProfileKind::None),
        amrex::GpuArray<amrex::Real, 3>{
            spec.prescribedULb[0],
            spec.prescribedULb[1],
            spec.prescribedULb[2]},
        spec.rhoLb,
        spec.uMaxLb};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
boundary::initialization::ValidBoundaryMacroscopicState
complete_valid_boundary_initialization_state(
    ValidBoundaryInitializationExpectedFace const& face,
    boundary::initialization::ValidBoundaryMacroscopicState const& donor,
    bool lowSide,
    int i,
    int j,
    int k,
    amrex::Box const& domain) noexcept
{
    static_cast<void>(i);
    static_cast<void>(k);
    using State = boundary::initialization::ValidBoundaryMacroscopicState;

    switch (face.physical) {
    case boundary::BoundaryPhysical::NoSlipWall:
        return State{donor.rho, 0.0, 0.0, 0.0};

    case boundary::BoundaryPhysical::MovingWall:
        return State{
            donor.rho,
            face.velocity[0],
            face.velocity[1],
            face.velocity[2]};

    case boundary::BoundaryPhysical::VelocityInlet:
    case boundary::BoundaryPhysical::VelocityOutlet: {
        if (face.profile == boundary::BoundaryProfileKind::Uniform) {
            return State{
                donor.rho,
                face.velocity[0],
                face.velocity[1],
                face.velocity[2]};
        }
        const amrex::Real eta =
            static_cast<amrex::Real>(j - domain.smallEnd(1)) /
            static_cast<amrex::Real>(domain.length(1) - 1);
        const amrex::Real speed = 4.0 * face.uMax * eta * (1.0 - eta);
        const int inwardSign = lowSide ? 1 : -1;
        const int velocitySign =
            face.physical == boundary::BoundaryPhysical::VelocityInlet ?
                inwardSign : -inwardSign;
        return State{
            donor.rho,
            static_cast<amrex::Real>(velocitySign) * speed,
            0.0,
            0.0};
    }

    case boundary::BoundaryPhysical::PressureInlet:
    case boundary::BoundaryPhysical::PressureOutlet:
        return State{face.rho, donor.ux, donor.uy, donor.uz};

    case boundary::BoundaryPhysical::PrescribedState: {
        amrex::Real scale = 1.0;
        if (face.profile == boundary::BoundaryProfileKind::Poiseuille) {
            const amrex::Real eta =
                static_cast<amrex::Real>(j - domain.smallEnd(1)) /
                static_cast<amrex::Real>(domain.length(1) - 1);
            scale = 4.0 * eta * (1.0 - eta);
        }
        return State{
            face.rho,
            scale * face.velocity[0],
            scale * face.velocity[1],
            scale * face.velocity[2]};
    }

    case boundary::BoundaryPhysical::FreeOutflow:
        return donor;

    default:
        return State{};
    }
}

struct TimeDependentBoundaryInitializationProbe {
    amrex::Real zScale = 0.0;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    boundary::initialization::ValidBoundaryMacroscopicState operator()(
        boundary::initialization::ValidBoundaryMacroscopicState const& donor,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
        amrex::Real time) const noexcept
    {
        return boundary::initialization::ValidBoundaryMacroscopicState{
            donor.rho + 1.0e-3 * time,
            2.0e-2 * time + 1.0e-5 * position[0],
            -1.0e-2 * time + 1.0e-5 * position[1],
            zScale * time +
                (zScale != 0.0 ?
                    1.0e-5 * position[2] : 0.0)};
    }
};

inline boundary::DomainBoundarySpec make_guo_free_outflow_domain_spec(
    std::string name,
    boundary::DomainSide side)
{
    boundary::DomainBoundarySpec spec = make_domain_wall_spec(
        std::move(name), side, boundary::BoundaryPhysical::FreeOutflow);
    spec.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    spec.profile = boundary::BoundaryProfileKind::None;
    return spec;
}

/**
 * @brief Verify the equilibrium-only valid-boundary initialization lifecycle.
 *
 * The cases cover every executable Guo physical/profile family, on-lattice
 * Poiseuille coordinates, wall-wins edges/corners, full-q writes, and the
 * roundoff relation to the former runtime-reconstruction initialization path.
 * A separate low-level resolver makes the supplied t_0 observable so steady
 * built-in profiles do not hide a dropped time argument.
 */
template <class Descriptor>
void run_valid_boundary_initialization_integration_tests()
{
    constexpr amrex::Real sentinel = -6.543210987e20;
    constexpr amrex::Real tolerance = 5.0e-13;
    constexpr amrex::Real initializationTime = 2.75;
    const int nx = 7;
    const int ny = 5;
    const int nz = Descriptor::d == 3 ? 5 : 1;
    amrex::Box domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    constexpr amrex::Real dx = 0.25;
    amrex::RealBox realBox(
        {AMREX_D_DECL(-0.5 * dx, -0.5 * dx, -0.5 * dx)},
        {AMREX_D_DECL(
            (static_cast<amrex::Real>(nx) - 0.5) * dx,
            (static_cast<amrex::Real>(ny) - 0.5) * dx,
            (static_cast<amrex::Real>(nz) - 0.5) * dx)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(0, 0, Descriptor::d == 3 ? 0 : 1)};
    amrex::Geometry geom(
        domain, &realBox, amrex::CoordSys::cartesian, isPeriodic.data());

    amrex::BoxArray ba(domain);
    amrex::DistributionMapping dm(ba);

    // HBB lives on valid/ghost faces and must not alter any valid population.
    std::vector<boundary::DomainBoundarySpec> hbbSpecs{
        make_domain_wall_spec("hbb_init_xlo", boundary::DomainSide::XLo),
        make_domain_wall_spec("hbb_init_xhi", boundary::DomainSide::XHi),
        make_domain_wall_spec("hbb_init_ylo", boundary::DomainSide::YLo),
        make_domain_wall_spec("hbb_init_yhi", boundary::DomainSide::YHi)};
    if constexpr (Descriptor::d == 3) {
        hbbSpecs.push_back(
            make_domain_wall_spec("hbb_init_zlo", boundary::DomainSide::ZLo));
        hbbSpecs.push_back(
            make_domain_wall_spec("hbb_init_zhi", boundary::DomainSide::ZHi));
    }
    boundary::BoundarySystem<Descriptor> hbbSystem;
    define_boundary_system_for_test(hbbSystem, hbbSpecs, geom, ba, dm);
    LatticeField<Descriptor> hbbPopulations("hbb_initialization_noop", ba, dm);
    for (amrex::MFIter mfi(hbbPopulations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = hbbPopulations.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int iPop) noexcept {
                f(i, j, k, iPop) =
                    10000.0 * static_cast<amrex::Real>(iPop) +
                    100.0 * static_cast<amrex::Real>(i) +
                    10.0 * static_cast<amrex::Real>(j) +
                    static_cast<amrex::Real>(k);
            });
    }
    hbbSystem.initializeValidBoundaries(
        hbbPopulations, initializationTime);
    amrex::MultiFab hbbError(ba, dm, 1, 0);
    hbbError.setVal(0.0);
    for (amrex::MFIter mfi(hbbPopulations.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const actual = hbbPopulations.multifab().const_array(mfi);
        auto const error = hbbError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expected =
                    10000.0 * static_cast<amrex::Real>(iPop) +
                    100.0 * static_cast<amrex::Real>(i) +
                    10.0 * static_cast<amrex::Real>(j) +
                    static_cast<amrex::Real>(k);
                amrex::Real difference =
                    actual(i, j, k, iPop) - expected;
                difference = difference < 0.0 ? -difference : difference;
                maxError = difference > maxError ? difference : maxError;
            }
            error(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        hbbError.max(0) == 0.0,
        "HBB valid-boundary initialization must be an exact no-op");

    const amrex::Array<amrex::Real, 3> movingVelocity{
        0.031, -0.002, Descriptor::d == 3 ? 0.001 : 0.0};
    const amrex::Array<amrex::Real, 3> uniformVelocity{
        0.04, 0.001, Descriptor::d == 3 ? -0.001 : 0.0};
    const amrex::Array<amrex::Real, 3> prescribedVelocity{
        0.025, -0.003, Descriptor::d == 3 ? 0.002 : 0.0};
    const amrex::Array<amrex::Real, 3> prescribedPeakVelocity{
        0.045, 0.002, Descriptor::d == 3 ? -0.0015 : 0.0};

    std::vector<std::array<boundary::DomainBoundarySpec, 2>> cases;
    cases.push_back({
        make_guo_no_slip_domain_spec(
            "init_no_slip_xlo", boundary::DomainSide::XLo),
        make_guo_moving_wall_domain_spec(
            "init_moving_wall_xhi",
            boundary::DomainSide::XHi,
            movingVelocity)});
    cases.push_back({
        make_guo_uniform_velocity_domain_spec(
            "init_uniform_velocity_xlo",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::VelocityInlet,
            uniformVelocity),
        make_guo_poiseuille_velocity_domain_spec(
            "init_poiseuille_velocity_xhi",
            boundary::DomainSide::XHi,
            boundary::BoundaryPhysical::VelocityOutlet,
            1,
            0.055)});
    cases.push_back({
        make_guo_pressure_domain_spec(
            "init_pressure_xlo",
            boundary::DomainSide::XLo,
            boundary::BoundaryPhysical::PressureInlet,
            1.04),
        make_guo_prescribed_state_domain_spec(
            "init_prescribed_uniform_xhi",
            boundary::DomainSide::XHi,
            0.98,
            prescribedVelocity)});
    cases.push_back({
        make_guo_prescribed_state_domain_spec(
            "init_prescribed_poiseuille_xlo",
            boundary::DomainSide::XLo,
            1.015,
            prescribedPeakVelocity,
            boundary::BoundaryProfileKind::Poiseuille,
            1),
        make_guo_free_outflow_domain_spec(
            "init_free_outflow_xhi", boundary::DomainSide::XHi)});

    for (auto const& testCase : cases) {
        std::vector<boundary::DomainBoundarySpec> specs{
            testCase[0],
            testCase[1],
            make_guo_no_slip_domain_spec(
                "init_wall_ylo", boundary::DomainSide::YLo),
            make_guo_no_slip_domain_spec(
                "init_wall_yhi", boundary::DomainSide::YHi)};
        if constexpr (Descriptor::d == 3) {
            specs.push_back(make_guo_no_slip_domain_spec(
                "init_wall_zlo", boundary::DomainSide::ZLo));
            specs.push_back(make_guo_no_slip_domain_spec(
                "init_wall_zhi", boundary::DomainSide::ZHi));
        }

        boundary::BoundarySystem<Descriptor> boundarySystem;
        define_boundary_system_for_test(
            boundarySystem, specs, geom, ba, dm);
        LatticeField<Descriptor> populations(
            "valid_boundary_initialization_case", ba, dm);
        for (amrex::MFIter mfi(
                 populations.multifab(), amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const f = populations.multifab().array(mfi);
            amrex::ParallelFor(
                bx,
                Descriptor::q,
                [=] AMREX_GPU_DEVICE(
                    int i, int j, int k, int iPop) noexcept {
                    const bool boundaryNode =
                        i == domain.smallEnd(0) ||
                        i == domain.bigEnd(0) ||
                        j == domain.smallEnd(1) ||
                        j == domain.bigEnd(1) ||
                        (Descriptor::d == 3 &&
                         (k == domain.smallEnd(2) ||
                          k == domain.bigEnd(2)));
                    if (boundaryNode) {
                        f(i, j, k, iPop) = sentinel;
                        return;
                    }
                    const auto state =
                        valid_boundary_initialization_bulk_state<Descriptor>(
                            i, j, k);
                    f(i, j, k, iPop) =
                        Equilibrium<Descriptor>::compute(
                            iPop,
                            state.rho,
                            state.ux,
                            state.uy,
                            state.uz);
                });
        }
        amrex::Gpu::streamSynchronize();

        // Preserve the actual initialized bits: recomputing equilibrium in a
        // differently shaped GPU kernel is not an exact no-write oracle.
        amrex::MultiFab initialPopulations(
            ba, dm, Descriptor::q, 0);
        amrex::MultiFab::Copy(
            initialPopulations,
            populations.multifab(),
            0,
            0,
            Descriptor::q,
            0);

        boundarySystem.initializeValidBoundaries(
            populations, initializationTime);

        const ValidBoundaryInitializationExpectedFace xloFace =
            make_valid_boundary_initialization_expected_face(testCase[0]);
        const ValidBoundaryInitializationExpectedFace xhiFace =
            make_valid_boundary_initialization_expected_face(testCase[1]);
        amrex::MultiFab error(ba, dm, 7, 0);
        error.setVal(0.0);
        for (amrex::MFIter mfi(
                 populations.multifab(), amrex::TilingIfNotGPU());
             mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const actual = populations.multifab().const_array(mfi);
            auto const initial = initialPopulations.const_array(mfi);
            auto const err = error.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const bool onXLo = i == domain.smallEnd(0);
                    const bool onXHi = i == domain.bigEnd(0);
                    const bool onYLo = j == domain.smallEnd(1);
                    const bool onYHi = j == domain.bigEnd(1);
                    const bool onZLo =
                        Descriptor::d == 3 && k == domain.smallEnd(2);
                    const bool onZHi =
                        Descriptor::d == 3 && k == domain.bigEnd(2);
                    const bool boundaryNode =
                        onXLo || onXHi || onYLo || onYHi ||
                        onZLo || onZHi;

                    if (!boundaryNode) {
                        amrex::Real maxInteriorError = 0.0;
                        for (int iPop = 0;
                             iPop < Descriptor::q; ++iPop) {
                            amrex::Real difference =
                                actual(i, j, k, iPop) -
                                initial(i, j, k, iPop);
                            difference =
                                difference < 0.0 ? -difference : difference;
                            maxInteriorError =
                                difference > maxInteriorError ?
                                    difference : maxInteriorError;
                        }
                        err(i, j, k, 5) = maxInteriorError;
                        return;
                    }

                    const int rx = onXLo ? 1 : (onXHi ? -1 : 0);
                    const int ry = onYLo ? 1 : (onYHi ? -1 : 0);
                    const int rz = onZLo ? 1 : (onZHi ? -1 : 0);
                    const auto analyticDonor =
                        valid_boundary_initialization_bulk_state<Descriptor>(
                            i + rx, j + ry, k + rz);
                    amrex::GpuArray<amrex::Real, Descriptor::q>
                        donorPopulations{};
                    for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                        donorPopulations[iPop] =
                            Equilibrium<Descriptor>::compute(
                                iPop,
                                analyticDonor.rho,
                                analyticDonor.ux,
                                analyticDonor.uy,
                                analyticDonor.uz);
                    }
                    const boundary::GuoDonorState recovered =
                        boundary::GuoNonEquilibriumExtrapolationScheme<
                            Descriptor>::donor_state(donorPopulations);
                    const boundary::initialization::
                        ValidBoundaryMacroscopicState donor{
                            recovered.rho,
                            recovered.ux,
                            recovered.uy,
                            recovered.uz};

                    ValidBoundaryInitializationExpectedFace effectiveFace;
                    bool effectiveLowSide = true;
                    if (onYLo || onYHi || onZLo || onZHi) {
                        effectiveFace.physical =
                            boundary::BoundaryPhysical::NoSlipWall;
                    } else {
                        effectiveFace = onXLo ? xloFace : xhiFace;
                        effectiveLowSide = onXLo;
                    }
                    const auto state =
                        complete_valid_boundary_initialization_state(
                            effectiveFace,
                            donor,
                            effectiveLowSide,
                            i,
                            j,
                            k,
                            domain);

                    amrex::GpuArray<amrex::Real, Descriptor::q>
                        formerInitialization{};
                    if (effectiveFace.physical ==
                        boundary::BoundaryPhysical::FreeOutflow) {
                        for (int iPop = 0;
                             iPop < Descriptor::q; ++iPop) {
                            formerInitialization[iPop] =
                                donorPopulations[iPop];
                        }
                    } else {
                        boundary::GuoNonEquilibriumExtrapolationScheme<
                            Descriptor>::reconstruct(
                                donorPopulations,
                                formerInitialization,
                                state.rho,
                                amrex::GpuArray<amrex::Real, 3>{
                                    state.ux, state.uy, state.uz});
                    }

                    amrex::Real maxPopulationError = 0.0;
                    amrex::Real maxFormerPathDifference = 0.0;
                    amrex::Real sentinelLeft = 0.0;
                    for (int iPop = 0;
                         iPop < Descriptor::q; ++iPop) {
                        const amrex::Real expected =
                            Equilibrium<Descriptor>::compute(
                                iPop,
                                state.rho,
                                state.ux,
                                state.uy,
                                state.uz);
                        amrex::Real difference =
                            actual(i, j, k, iPop) - expected;
                        difference =
                            difference < 0.0 ? -difference : difference;
                        maxPopulationError =
                            difference > maxPopulationError ?
                                difference : maxPopulationError;

                        amrex::Real formerDifference =
                            actual(i, j, k, iPop) -
                            formerInitialization[iPop];
                        formerDifference =
                            formerDifference < 0.0 ?
                                -formerDifference : formerDifference;
                        maxFormerPathDifference =
                            formerDifference > maxFormerPathDifference ?
                                formerDifference :
                                maxFormerPathDifference;
                        if (actual(i, j, k, iPop) == sentinel) {
                            sentinelLeft = 1.0;
                        }
                    }
                    err(i, j, k, 0) = maxPopulationError;
                    err(i, j, k, 4) = maxFormerPathDifference;
                    err(i, j, k, 6) = sentinelLeft;

                    ConstPopulationCell boundaryCell{actual, i, j, k};
                    const amrex::Real rho =
                        Moments<Descriptor>::rho(boundaryCell);
                    amrex::Real ux = 0.0;
                    amrex::Real uy = 0.0;
                    amrex::Real uz = 0.0;
                    Moments<Descriptor>::velocity(
                        boundaryCell, rho, ux, uy, uz);
                    amrex::Real rhoError = rho - state.rho;
                    rhoError = rhoError < 0.0 ? -rhoError : rhoError;
                    const amrex::Real dux = ux - state.ux;
                    const amrex::Real duy = uy - state.uy;
                    const amrex::Real duz = uz - state.uz;
                    err(i, j, k, 1) = rhoError;
                    err(i, j, k, 2) =
                        std::sqrt(dux * dux + duy * duy + duz * duz);
                });
        }
        amrex::Gpu::streamSynchronize();
        expect(
            error.max(0) < tolerance,
            "valid-boundary initialization must write all-q equilibrium populations");
        expect(
            error.max(1) < tolerance && error.max(2) < tolerance,
            "valid-boundary initialization macroscopic state mismatch");
        expect(
            error.max(4) < tolerance,
            "equilibrium initialization differs excessively from former Guo initialization path");
        expect(
            error.max(5) == 0.0,
            "valid-boundary initialization must preserve strict interior populations");
        expect(
            error.max(6) == 0.0,
            "valid-boundary initialization left a boundary population unwritten");
    }

    // Directly exercise the Operator's current-time contract with a resolver
    // whose state changes with time; built-in boundary profiles are steady.
    LatticeField<Descriptor> timeProbe("boundary_initialization_time_probe", ba, dm);
    for (amrex::MFIter mfi(timeProbe.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const f = timeProbe.multifab().array(mfi);
        amrex::ParallelFor(
            bx,
            Descriptor::q,
            [=] AMREX_GPU_DEVICE(
                int i, int j, int k, int iPop) noexcept {
                const auto state =
                    valid_boundary_initialization_bulk_state<Descriptor>(
                        i, j, k);
                f(i, j, k, iPop) =
                    Equilibrium<Descriptor>::compute(
                        iPop,
                        state.rho,
                        state.ux,
                        state.uy,
                        state.uz);
            });
    }
    boundary::PostStreamBoundaryRegion probeRegion;
    probeRegion.box = domain;
    probeRegion.box.setBig(0, domain.smallEnd(0));
    probeRegion.box.growLo(1, -1);
    probeRegion.box.growHi(1, -1);
    if constexpr (Descriptor::d == 3) {
        probeRegion.box.growLo(2, -1);
        probeRegion.box.growHi(2, -1);
    }
    probeRegion.inward[0] = 1;
    boundary::LbmValidBoundaryInitializationOperator<Descriptor>
        initializationOperator;
    expect(
        initializationOperator.name() ==
            std::string_view("valid_boundary_initialization"),
        "valid-boundary initialization Operator name mismatch");
    expect(
        initializationOperator.scope() ==
            operators::OperatorScope::Level,
        "valid-boundary initialization Operator scope mismatch");
    initializationOperator.applyRegion(
        timeProbe,
        probeRegion,
        geom.data(),
        initializationTime,
        TimeDependentBoundaryInitializationProbe{
            Descriptor::d == 3 ? 5.0e-3 : 0.0});

    amrex::MultiFab timeError(ba, dm, 1, 0);
    timeError.setVal(0.0);
    const amrex::GeometryData geometry = geom.data();
    for (amrex::MFIter mfi(timeProbe.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const amrex::Box bx = mfi.tilebox() & probeRegion.box;
        if (!bx.ok()) {
            continue;
        }
        auto const actual = timeProbe.multifab().const_array(mfi);
        auto const error = timeError.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const auto analyticDonor =
                valid_boundary_initialization_bulk_state<Descriptor>(
                    i + 1, j, k);
            amrex::GpuArray<amrex::Real, Descriptor::q> donorPopulations{};
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                donorPopulations[iPop] =
                    Equilibrium<Descriptor>::compute(
                        iPop,
                        analyticDonor.rho,
                        analyticDonor.ux,
                        analyticDonor.uy,
                        analyticDonor.uz);
            }
            const boundary::GuoDonorState recovered =
                boundary::GuoNonEquilibriumExtrapolationScheme<
                    Descriptor>::donor_state(donorPopulations);
            const boundary::initialization::ValidBoundaryMacroscopicState
                donor{
                    recovered.rho,
                    recovered.ux,
                    recovered.uy,
                    recovered.uz};
            const auto expectedState =
                TimeDependentBoundaryInitializationProbe{
                    Descriptor::d == 3 ? 5.0e-3 : 0.0}(
                    donor,
                    boundary::boundary_cell_center_position(
                        geometry, i, j, k),
                    initializationTime);
            amrex::Real maxError = 0.0;
            for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                const amrex::Real expected =
                    Equilibrium<Descriptor>::compute(
                        iPop,
                        expectedState.rho,
                        expectedState.ux,
                        expectedState.uy,
                        expectedState.uz);
                amrex::Real difference =
                    actual(i, j, k, iPop) - expected;
                difference = difference < 0.0 ? -difference : difference;
                maxError = difference > maxError ? difference : maxError;
            }
            error(i, j, k, 0) = maxError;
        });
    }
    amrex::Gpu::streamSynchronize();
    expect(
        timeError.max(0) < tolerance,
        "valid-boundary initialization Operator must pass current t_0 to the state resolver");

    amrex::Print() << Descriptor::name()
                   << " valid-boundary initialization integration tests passed.\n";
}

} // namespace lbm::tests::integration

#endif
