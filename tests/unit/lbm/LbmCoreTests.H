#ifndef LBM_TESTS_UNIT_LBM_LBMCORETESTS_H
#define LBM_TESTS_UNIT_LBM_LBMCORETESTS_H

#include "unit/TestAssertions.H"

#include "lbm/config/LbmConfig.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamics.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/dynamics/DynamicsModel.H"
#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"

#include <AMReX_Array.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <type_traits>

namespace lbm::tests {

template <class Descriptor>
void run_lbm_core_tests()
{
    using OtherDescriptor = std::conditional_t<
        Descriptor::d == 2,
        D3Q27Descriptor,
        D2Q9Descriptor>;
    static_assert(
        is_dynamics_model_for_descriptor<
            BGKDynamics<Descriptor>,
            Descriptor>::value);
    static_assert(
        !is_dynamics_model_for_descriptor<
            BGKDynamics<Descriptor>,
            OtherDescriptor>::value);

    amrex::Real weightSum = 0.0;
    require(
        Descriptor::cx(0) == 0 && Descriptor::cy(0) == 0 &&
            Descriptor::cz(0) == 0,
        "Descriptor rest population must have zero lattice velocity");
    const int half = (Descriptor::q - 1) / 2;
    for (int population = 0; population < Descriptor::q; ++population) {
        weightSum += Descriptor::w(population);
        const int opposite = Descriptor::opposite(population);
        require(
            Descriptor::opposite(opposite) == population &&
                Descriptor::cx(population) + Descriptor::cx(opposite) == 0 &&
                Descriptor::cy(population) + Descriptor::cy(opposite) == 0 &&
                Descriptor::cz(population) + Descriptor::cz(opposite) == 0,
            "Descriptor opposite mapping must be involutive and antisymmetric");
    }
    for (int population = 1; population <= half; ++population) {
        require(
            Descriptor::opposite(population) == population + half &&
                Descriptor::opposite(population + half) == population,
            "Descriptor opposite mapping must preserve Palabos ordering");
    }
    require(
        std::abs(weightSum - 1.0) < 1.0e-14 &&
            Descriptor::rho0_lb() > 0.0,
        "Descriptor weights or reference density are invalid");

    constexpr amrex::Real rho0 = 1.0;
    constexpr amrex::Real ux0 = 0.01;
    constexpr amrex::Real uy0 = -0.02;
    const amrex::Real uz0 = Descriptor::d == 3 ? 0.012 : 0.0;
    amrex::GpuArray<amrex::Real, Descriptor::q> equilibrium{};
    for (int population = 0; population < Descriptor::q; ++population) {
        equilibrium[population] = Equilibrium<Descriptor>::compute(
            population, rho0, ux0, uy0, uz0);
    }
    const amrex::Real rho = Moments<Descriptor>::rho(equilibrium);
    amrex::Real ux = 0.0;
    amrex::Real uy = 0.0;
    amrex::Real uz = 0.0;
    Moments<Descriptor>::velocity(equilibrium, rho, ux, uy, uz);
    require(
        std::abs(rho - rho0) < 1.0e-14 &&
            std::abs(ux - ux0) < 1.0e-14 &&
            std::abs(uy - uy0) < 1.0e-14 &&
            std::abs(uz - uz0) < 1.0e-14,
        "Equilibrium moments must recover the prescribed state");

    config::ResolvedLbmConfig config;
    config.omega = 0.7;
    auto prepared = BGKDynamicsSelection{}.prepare<Descriptor>(config);
    require(
        prepared.ok() &&
            std::abs(prepared.dynamics->omega() - config.omega) < 1.0e-15,
        "BGK selection must preserve the resolved relaxation value");

    auto cell = equilibrium;
    prepared.dynamics->collide(cell);
    for (int population = 0; population < Descriptor::q; ++population) {
        require(
            std::abs(cell[population] - equilibrium[population]) < 1.0e-14,
            "BGK collision must preserve an equilibrium state");
    }

    cell = equilibrium;
    cell[1] += 1.0e-3;
    const auto before = cell;
    const amrex::Real perturbedRho = Moments<Descriptor>::rho(before);
    Moments<Descriptor>::velocity(before, perturbedRho, ux, uy, uz);
    prepared.dynamics->collide(cell);
    for (int population = 0; population < Descriptor::q; ++population) {
        const amrex::Real expected = before[population] - config.omega *
            (before[population] - Equilibrium<Descriptor>::compute(
                population, perturbedRho, ux, uy, uz));
        require(
            std::abs(cell[population] - expected) < 1.0e-14,
            "BGK collision value does not match the independent relaxation expression");
    }

    amrex::Print() << Descriptor::name() << " LBM core unit tests passed.\n";
}

} // namespace lbm::tests

#endif
