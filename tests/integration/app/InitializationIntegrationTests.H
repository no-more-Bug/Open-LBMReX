#ifndef LBM_TESTS_INTEGRATION_APP_INITIALIZATIONINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_APP_INITIALIZATIONINTEGRATIONTESTS_H

#include "lbm/dynamics/Equilibrium.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "lbm/initialization/SingleLevelInitializer.H"
#include "lbm/initialization/TaylorGreenVortexProfile.H"
#include "lbm/operators/level/EquilibriumPopulationInitializationOperator.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <cmath>

namespace lbm::app {

namespace initialization_integration_test_detail {

inline amrex::Geometry make_geometry(
    int nx,
    int ny,
    int nz,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probLo,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probHi)
{
    amrex::IntVect const lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect const hi(
        AMREX_D_DECL(nx - 1, ny - 1, nz - 1));
    amrex::Box const domain(lo, hi);
    amrex::RealBox const realBox(probLo.data(), probHi.data());
    amrex::Array<int, AMREX_SPACEDIM> const periodic{
        AMREX_D_DECL(1, 1, 1)};
    return amrex::Geometry(
        domain,
        &realBox,
        amrex::CoordSys::cartesian,
        periodic.data());
}

struct PositionTimeProbeProfile {
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
void expect_operator_position_and_time()
{
    constexpr amrex::Real initializationTime = 0.375;
    const int nz = Descriptor::d == 3 ? 3 : 1;
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo{
        AMREX_D_DECL(-1.25, 2.5, -0.75)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probHi{
        AMREX_D_DECL(0.75, 7.5, Descriptor::d == 3 ? 0.75 : -0.25)};
    const amrex::Geometry geom =
        make_geometry(4, 5, nz, probLo, probHi);
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(2);
    amrex::DistributionMapping dm(ba);
    LatticeField<Descriptor> populations(
        "initialization_position_time_probe", ba, dm);

    PositionTimeProbeProfile const profile;
    operators::EquilibriumPopulationInitializationOperator<
        Descriptor,
        PositionTimeProbeProfile> initialization(profile, geom);
    initialization.apply(populations, initializationTime);

    amrex::MultiFab error(ba, dm, 1, 0);
    const amrex::GeometryData geometry = geom.data();
    for (amrex::MFIter mfi(
             populations.multifab(),
             amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::GpuArray<
                    amrex::Real,
                    AMREX_SPACEDIM> position{
                    AMREX_D_DECL(
                        probLo[0] +
                            (static_cast<amrex::Real>(i) + 0.5) *
                                geometry.CellSize(0),
                        probLo[1] +
                            (static_cast<amrex::Real>(j) + 0.5) *
                                geometry.CellSize(1),
                        probLo[2] +
                            (static_cast<amrex::Real>(k) + 0.5) *
                                geometry.CellSize(2))};
                const initialization::MacroscopicState state =
                    profile(position, initializationTime);
                amrex::Real maxError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    amrex::Real difference =
                        f(i, j, k, iPop) -
                        Equilibrium<Descriptor>::compute(
                            iPop,
                            state.rho,
                            state.ux,
                            state.uy,
                            state.uz);
                    difference =
                        difference < 0.0 ? -difference : difference;
                    maxError =
                        difference > maxError ? difference : maxError;
                }
                err(i, j, k, 0) = maxError;
            });
    }
    amrex::Gpu::streamSynchronize();
    AMREX_ALWAYS_ASSERT(error.max(0) < 1.0e-14);
}

template <class Descriptor, class Profile>
void expect_initializer_matches_profile(
    initialization::ResolvedInitializationConfig const& setup,
    Profile expectedProfile,
    amrex::Geometry const& geom)
{
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(2);
    amrex::DistributionMapping dm(ba);
    LatticeField<Descriptor> populations(
        "initialization_self_check", ba, dm);

    initialization::SingleLevelInitializer<Descriptor> initializer;
    initializer.initialize(populations, geom, setup);

    amrex::MultiFab error(ba, dm, 1, 0);
    const amrex::GeometryData geometry = geom.data();
    for (amrex::MFIter mfi(
             populations.multifab(),
             amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        const auto profile = expectedProfile;
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::GpuArray<
                    amrex::Real,
                    AMREX_SPACEDIM> position{
                    AMREX_D_DECL(
                        geometry.ProbLo(0) +
                            (static_cast<amrex::Real>(i) + 0.5) *
                                geometry.CellSize(0),
                        geometry.ProbLo(1) +
                            (static_cast<amrex::Real>(j) + 0.5) *
                                geometry.CellSize(1),
                        geometry.ProbLo(2) +
                            (static_cast<amrex::Real>(k) + 0.5) *
                                geometry.CellSize(2))};
                const initialization::MacroscopicState state =
                    profile(position, 0.0);
                amrex::Real maxError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    amrex::Real difference =
                        f(i, j, k, iPop) -
                        Equilibrium<Descriptor>::compute(
                            iPop,
                            state.rho,
                            state.ux,
                            state.uy,
                            state.uz);
                    difference =
                        difference < 0.0 ? -difference : difference;
                    maxError =
                        difference > maxError ? difference : maxError;
                }
                err(i, j, k, 0) = maxError;
            });
    }
    amrex::Gpu::streamSynchronize();

    // MultiFab::max(local=false) is an MPI all-reduce. Every rank executes
    // this helper in the same four-kind order.
    AMREX_ALWAYS_ASSERT(error.max(0) < 1.0e-14);
}

template <class Descriptor>
void expect_poiseuille_step_zero_equivalence(
    initialization::ResolvedInitializationConfig const& setup,
    bool onLattice)
{
    const int nx = onLattice ? 5 : 4;
    const int ny = onLattice ? 6 : 5;
    const int nz = Descriptor::d == 3 ? 3 : 1;
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo{
        AMREX_D_DECL(
            onLattice ? -0.5 : 0.0,
            onLattice ? -0.1 : 0.0,
            0.0)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probHi{
        AMREX_D_DECL(
            onLattice ? 4.5 : 4.0,
            onLattice ? 1.1 : 1.0,
            static_cast<amrex::Real>(nz))};
    const amrex::Geometry geom =
        make_geometry(nx, ny, nz, probLo, probHi);
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(2);
    amrex::DistributionMapping dm(ba);
    LatticeField<Descriptor> populations(
        onLattice ? "poiseuille_guo_step_zero"
                  : "poiseuille_hbb_step_zero",
        ba,
        dm);
    initialization::SingleLevelInitializer<Descriptor> initializer;
    initializer.initialize(populations, geom, setup);

    const initialization::PoiseuilleInitializationParams p =
        setup.poiseuilleParams;
    amrex::MultiFab error(ba, dm, 1, 0);
    for (amrex::MFIter mfi(
             populations.multifab(),
             amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const f = populations.multifab().const_array(mfi);
        auto const err = error.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const amrex::Real xi = onLattice ?
                    static_cast<amrex::Real>(i) /
                        static_cast<amrex::Real>(nx - 1) :
                    (static_cast<amrex::Real>(i) + 0.5) /
                        static_cast<amrex::Real>(nx);
                const amrex::Real eta = onLattice ?
                    static_cast<amrex::Real>(j) /
                        static_cast<amrex::Real>(ny - 1) :
                    (static_cast<amrex::Real>(j) + 0.5) /
                        static_cast<amrex::Real>(ny);
                const amrex::Real rho = p.linearDensity != 0 ?
                    p.rhoXLoLb + xi * (p.rhoXHiLb - p.rhoXLoLb) :
                    p.rhoLb;
                const amrex::Real ux =
                    4.0 * p.uMaxLb * eta * (1.0 - eta);
                amrex::Real maxError = 0.0;
                for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
                    amrex::Real difference =
                        f(i, j, k, iPop) -
                        Equilibrium<Descriptor>::compute(
                            iPop, rho, ux, 0.0, 0.0);
                    difference =
                        difference < 0.0 ? -difference : difference;
                    maxError =
                        difference > maxError ? difference : maxError;
                }
                err(i, j, k, 0) = maxError;
            });
    }
    amrex::Gpu::streamSynchronize();
    AMREX_ALWAYS_ASSERT(error.max(0) < 1.0e-14);
}

} // namespace initialization_integration_test_detail

/**
 * @brief Check initialization field geometry and Operator dispatch.
 *
 * All ranks execute the same cases and collective error reductions. The tests
 * launch only small equilibrium initialization kernels; they do not exercise
 * time advancement or boundary reconstruction.
 */
template <class Descriptor>
void run_initialization_integration_tests()
{
    using namespace initialization_integration_test_detail;
    const int nz = Descriptor::d == 3 ? 3 : 1;
    amrex::Array<int, AMREX_SPACEDIM> const nCell{
        AMREX_D_DECL(4, 5, nz)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo{
        AMREX_D_DECL(0.0, 0.0, 0.0)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probHi{
        AMREX_D_DECL(
            1.0,
            1.0,
            static_cast<amrex::Real>(nz))};
    amrex::Geometry const geom =
        make_geometry(nCell[0], nCell[1], nCell[2], probLo, probHi);

    expect_operator_position_and_time<Descriptor>();

    initialization::ResolvedInitializationConfig rest;
    rest.kind = initialization::InitializationKind::Rest;
    rest.constantState = {1.2, 0.0, 0.0, 0.0};
    initialization::ResolvedInitializationConfig uniform;
    uniform.kind = initialization::InitializationKind::Uniform;
    uniform.constantState = {1.1, 0.03, -0.01, 0.0};
    initialization::ResolvedInitializationConfig taylorGreen;
    taylorGreen.kind = initialization::InitializationKind::TaylorGreen;
    taylorGreen.taylorGreenParams = initialization::make_tgv_params(
        nCell[0], nCell[1], 1.05, 0.04, 0.02, Descriptor::cs2(), 0.0);
    initialization::ResolvedInitializationConfig poiseuille;
    poiseuille.kind = initialization::InitializationKind::Poiseuille;
    poiseuille.poiseuilleParams = {1.0, 0.08, 1, 1.04, 0.96};

    expect_initializer_matches_profile<Descriptor>(
        rest,
        initialization::ConstantInitializationProfile{
            rest.constantState},
        geom);
    expect_initializer_matches_profile<Descriptor>(
        uniform,
        initialization::ConstantInitializationProfile{
            uniform.constantState},
        geom);
    expect_initializer_matches_profile<Descriptor>(
        taylorGreen,
        initialization::TaylorGreenVortexProfile{
            taylorGreen.taylorGreenParams},
        geom);
    initialization::PoiseuilleInitializationParams const& p =
        poiseuille.poiseuilleParams;
    expect_initializer_matches_profile<Descriptor>(
        poiseuille,
        initialization::PoiseuilleInitializationProfile{p},
        geom);
    expect_poiseuille_step_zero_equivalence<Descriptor>(
        poiseuille, false);
    expect_poiseuille_step_zero_equivalence<Descriptor>(
        poiseuille, true);

    amrex::Print() << Descriptor::name()
                   << " initialization integration tests passed.\n";
}

} // namespace lbm::app

#endif
