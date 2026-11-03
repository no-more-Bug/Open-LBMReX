#ifndef LBM_TESTS_UNIT_APP_APPLICATIONVALUETESTS_H
#define LBM_TESTS_UNIT_APP_APPLICATIONVALUETESTS_H

#include "unit/TestAssertions.H"

#include "lbm/config/LbmConfig.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/InitializationConfigResolver.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "runtime/SingleLevelRunControl.H"

#include <AMReX_Array.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace lbm::app::tests {

inline config::ResolvedLbmConfig value_test_lbm(
    config::LbmParameterizationKind kind =
        config::LbmParameterizationKind::FromReynolds)
{
    config::ResolvedLbmConfig lbm;
    lbm.parameterizationKind = kind;
    lbm.rhoRefLb = 1.125;
    lbm.uRefLb = 0.075;
    lbm.uRefPhys = 1.0;
    lbm.lengthRefLb = 10.0;
    lbm.lengthRefPhys = 1.0;
    lbm.nuLb = 0.02;
    lbm.reynolds = 20.0;
    lbm.tau = 0.56;
    lbm.omega = 1.0 / lbm.tau;
    return lbm;
}

inline runtime::SingleLevelRunIntent direct_run_intent()
{
    runtime::SingleLevelRunIntent run;
    run.maxStep = 17;
    run.finalTimePhys = 0.2;
    run.kind = "focused";
    run.name = "unit-run";
    run.variant = "unit";
    return run;
}

inline void test_run_control_resolution()
{
    config::ResolvedLbmConfig lbm = value_test_lbm();
    lbm.uRefLb = 0.1;
    lbm.nuLb = 0.05;
    lbm.tau = 0.65;
    lbm.omega = 1.0 / lbm.tau;
    auto run = direct_run_intent();
    auto result = runtime::resolve_single_level_run_control(run, lbm);
    lbm::tests::require(
        result.ok() && result.resolved.maxStep == 17 &&
            result.resolved.kind == "focused",
        "Direct run control must preserve explicit step and identity values");

    run.convergenceEnabled = 1;
    lbm.parameterizationKind = config::LbmParameterizationKind::FromTau;
    result = runtime::resolve_single_level_run_control(run, lbm);
    lbm::tests::require(
        result.ok() && result.resolved.maxStep == 20 &&
            result.resolved.kind == "convergence" &&
            result.resolved.convergenceFixedTau == 1,
        "From-tau convergence run control closure changed");
    lbm.parameterizationKind = config::LbmParameterizationKind::FromReynolds;
    result = runtime::resolve_single_level_run_control(run, lbm);
    lbm::tests::require(
        result.ok() && result.resolved.maxStep == 20 &&
            result.resolved.convergenceFixedTau == 0,
        "From-Reynolds convergence run control closure changed");
    lbm.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    lbm::tests::require(
        !runtime::resolve_single_level_run_control(run, lbm).ok(),
        "From-nu convergence run control must remain unsupported");
    run.convergenceEnabled = 2;
    lbm::tests::require(
        !runtime::resolve_single_level_run_control(run, value_test_lbm()).ok(),
        "Non-boolean convergence flag must fail resolution");
}

template <class Descriptor>
void test_initialization_resolution_and_profiles()
{
    using initialization::InitializationConfigIntent;
    using initialization::InitializationKind;
    const config::ResolvedLbmConfig lbm = value_test_lbm();
    const amrex::Array<int, AMREX_SPACEDIM> cells{
        AMREX_D_DECL(4, 5, Descriptor::d == 3 ? 3 : 1)};

    InitializationConfigIntent missingKind;
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             missingKind, lbm, cells).ok(),
        "Initialization kind must remain explicit");

    InitializationConfigIntent rest;
    rest.kind = InitializationKind::Rest;
    rest.rhoLb = 1.2;
    auto restResult = initialization::resolve_initialization_config<Descriptor>(
        rest, lbm, cells);
    lbm::tests::require(
        restResult.ok() && restResult.resolved.constantState.rho == 1.2 &&
            restResult.resolved.constantState.ux == 0.0,
        "Rest initialization resolution changed");
    InitializationConfigIntent defaultRest;
    defaultRest.kind = InitializationKind::Rest;
    const auto defaultRestResult =
        initialization::resolve_initialization_config<Descriptor>(
            defaultRest, lbm, cells);
    lbm::tests::require(
        defaultRestResult.ok() &&
            defaultRestResult.resolved.constantState.rho == lbm.rhoRefLb &&
            defaultRestResult.resolved.constantState.ux == 0.0,
        "Rest initialization defaults changed");
    rest.uniformULb = amrex::Array<amrex::Real, 3>{0.1, 0.0, 0.0};
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             rest, lbm, cells).ok(),
        "Rest initialization must reject uniform velocity intent");

    InitializationConfigIntent uniform;
    uniform.kind = InitializationKind::Uniform;
    uniform.rhoLb = 1.1;
    uniform.uniformULb =
        amrex::Array<amrex::Real, 3>{0.03, -0.01, 0.0};
    auto uniformResult =
        initialization::resolve_initialization_config<Descriptor>(
            uniform, lbm, cells);
    lbm::tests::require(
        uniformResult.ok() &&
            uniformResult.resolved.constantState.rho == 1.1 &&
            uniformResult.resolved.constantState.ux == 0.03 &&
            uniformResult.resolved.constantState.uy == -0.01,
        "Uniform initialization resolution changed");
    InitializationConfigIntent defaultUniform;
    defaultUniform.kind = InitializationKind::Uniform;
    auto defaultUniformResult =
        initialization::resolve_initialization_config<Descriptor>(
            defaultUniform, lbm, cells);
    lbm::tests::require(
        defaultUniformResult.ok() &&
            defaultUniformResult.resolved.constantState.rho == lbm.rhoRefLb &&
            defaultUniformResult.resolved.constantState.ux == lbm.uRefLb,
        "Uniform initialization defaults changed");
    uniform.taylorGreenVelocityAmplitudeLb = 0.02;
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             uniform, lbm, cells).ok(),
        "Uniform initialization must reject Taylor-Green intent");
    uniform.taylorGreenVelocityAmplitudeLb.reset();
    uniform.uniformULb =
        amrex::Array<amrex::Real, 3>{0.03, -0.01, 0.02};
    auto dimensionalUniform =
        initialization::resolve_initialization_config<Descriptor>(
            uniform, lbm, cells);
    lbm::tests::require(
        Descriptor::d == 2 ? !dimensionalUniform.ok() :
                             dimensionalUniform.ok(),
        "Uniform initialization dimensional velocity contract changed");
    if constexpr (Descriptor::d == 3) {
        lbm::tests::require(
            dimensionalUniform.resolved.constantState.uz == 0.02,
            "D3 uniform initialization must preserve explicit uz");
    }

    InitializationConfigIntent tgv;
    tgv.kind = InitializationKind::TaylorGreen;
    tgv.rhoLb = 1.05;
    tgv.taylorGreenVelocityAmplitudeLb = 0.04;
    auto tgvResult = initialization::resolve_initialization_config<Descriptor>(
        tgv, lbm, cells);
    lbm::tests::require(
        tgvResult.ok() && tgvResult.resolved.taylorGreenParams.rho0 == 1.05 &&
            tgvResult.resolved.taylorGreenParams.u0 == 0.04,
        "Taylor-Green initialization resolution changed");
    InitializationConfigIntent defaultTgv;
    defaultTgv.kind = InitializationKind::TaylorGreen;
    const auto defaultTgvResult =
        initialization::resolve_initialization_config<Descriptor>(
            defaultTgv, lbm, cells);
    lbm::tests::require(
        defaultTgvResult.ok() &&
            defaultTgvResult.resolved.taylorGreenParams.rho0 == lbm.rhoRefLb &&
            defaultTgvResult.resolved.taylorGreenParams.u0 == lbm.uRefLb,
        "Taylor-Green initialization defaults changed");
    tgv.uniformULb = amrex::Array<amrex::Real, 3>{0.01, 0.0, 0.0};
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             tgv, lbm, cells).ok(),
        "Taylor-Green initialization must reject uniform intent");

    InitializationConfigIntent poiseuille;
    poiseuille.kind = InitializationKind::Poiseuille;
    poiseuille.rhoLb = 1.0;
    poiseuille.poiseuilleUMaxLb = 0.08;
    poiseuille.poiseuilleRhoXLoLb = 1.04;
    poiseuille.poiseuilleRhoXHiLb = 0.96;
    auto poiseuilleResult =
        initialization::resolve_initialization_config<Descriptor>(
            poiseuille, lbm, cells);
    lbm::tests::require(
        poiseuilleResult.ok() &&
            poiseuilleResult.resolved.poiseuilleParams.linearDensity == 1,
        "Poiseuille initialization resolution changed");
    InitializationConfigIntent defaultPoiseuille;
    defaultPoiseuille.kind = InitializationKind::Poiseuille;
    const auto defaultPoiseuilleResult =
        initialization::resolve_initialization_config<Descriptor>(
            defaultPoiseuille, lbm, cells);
    lbm::tests::require(
        defaultPoiseuilleResult.ok() &&
            defaultPoiseuilleResult.resolved.poiseuilleParams.rhoLb ==
                lbm.rhoRefLb &&
            defaultPoiseuilleResult.resolved.poiseuilleParams.uMaxLb ==
                lbm.uRefLb &&
            defaultPoiseuilleResult.resolved.poiseuilleParams.linearDensity == 0,
        "Poiseuille initialization defaults changed");
    InitializationConfigIntent partialPoiseuille;
    partialPoiseuille.kind = InitializationKind::Poiseuille;
    partialPoiseuille.poiseuilleRhoXLoLb = 1.04;
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             partialPoiseuille, lbm, cells).ok(),
        "Poiseuille endpoint densities must remain paired");
    poiseuille.taylorGreenVelocityAmplitudeLb = 0.02;
    lbm::tests::require(
        !initialization::resolve_initialization_config<Descriptor>(
             poiseuille, lbm, cells).ok(),
        "Poiseuille initialization must reject Taylor-Green intent");

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
        AMREX_D_DECL(2.0, 0.5, 0.25)};
    const auto constant = initialization::ConstantInitializationProfile{
        uniformResult.resolved.constantState}(position, 7.0);
    lbm::tests::require(
        constant.rho == 1.1 && constant.ux == 0.03 &&
            constant.uy == -0.01,
        "Constant initialization profile must preserve the resolved state");
    const auto profile = initialization::PoiseuilleInitializationProfile{
        poiseuilleResult.resolved.poiseuilleParams};
    const auto center = profile(position, 4.0);
    const auto low = profile(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(0.0, 0.0, 0.0)},
        4.0);
    const auto high = profile(
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>{
            AMREX_D_DECL(4.0, 1.0, 0.0)},
        4.0);
    lbm::tests::require(
        std::abs(center.rho - 1.0) < 1.0e-14 &&
            std::abs(center.ux - 0.08) < 1.0e-14 &&
            std::abs(low.rho - 1.04) < 1.0e-14 && low.ux == 0.0 &&
            std::abs(high.rho - 0.96) < 1.0e-14 && high.ux == 0.0,
        "Poiseuille initialization profile analytic values changed");
}

template <class Descriptor>
void run_application_value_tests()
{
    test_run_control_resolution();
    test_initialization_resolution_and_profiles<Descriptor>();
    amrex::Print() << Descriptor::name()
                   << " application value unit tests passed.\n";
}

} // namespace lbm::app::tests

#endif
