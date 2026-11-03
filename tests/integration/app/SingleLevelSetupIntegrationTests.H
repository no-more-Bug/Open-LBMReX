#ifndef LBM_TESTS_INTEGRATION_APP_SINGLELEVELSETUPINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_APP_SINGLELEVELSETUPINTEGRATIONTESTS_H

#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/boundary/domain/OnLatticeReferenceGeometryResolver.H"
#include "lbm/config/LbmConfig.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include <utility>
#include <vector>

namespace lbm::app {

namespace setup_integration_test_detail {

inline config::ResolvedLbmConfig resolved_lbm(
    config::LbmParameterizationKind kind)
{
    config::ResolvedLbmConfig lbm;
    lbm.parameterizationKind = kind;
    lbm.rhoRefLb = 1.0;
    lbm.uRefLb = 0.1;
    lbm.uRefPhys = 1.0;
    lbm.lengthRefLb = 10.0;
    lbm.lengthRefPhys = 1.0;
    lbm.nuLb = 0.05;
    lbm.reynolds = 20.0;
    lbm.tau = 0.65;
    lbm.omega = 1.0 / lbm.tau;
    return lbm;
}

struct CavityFixture {
    amrex::Geometry geometry;
    boundary::ResolvedBoundarySet resolved;
    boundary::PostStreamingBoundaryPlan plan;
};

template <class Descriptor>
CavityFixture make_cavity_fixture(
    boundary::BoundaryScheme scheme)
{
    const bool onLattice =
        scheme == boundary::BoundaryScheme::GuoExtrapolation;
    const int nxy = onLattice ? 6 : 5;
    const amrex::Array<int, AMREX_SPACEDIM> nCell{
        AMREX_D_DECL(nxy, nxy, 4)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo{
        AMREX_D_DECL(
            onLattice ? -0.1 : 0.0,
            onLattice ? -0.1 : 0.0,
            0.0)};
    const amrex::Array<amrex::Real, AMREX_SPACEDIM> probHi{
        AMREX_D_DECL(
            onLattice ? 1.1 : 1.0,
            onLattice ? 1.1 : 1.0,
            0.8)};
    const amrex::Array<int, AMREX_SPACEDIM> periodic{
        AMREX_D_DECL(0, 0, 1)};
    const amrex::IntVect lo(AMREX_D_DECL(0, 0, 0));
    const amrex::IntVect hi(AMREX_D_DECL(
        nCell[0] - 1, nCell[1] - 1, nCell[2] - 1));
    const amrex::Box domain(lo, hi);
    const amrex::RealBox realBox(probLo.data(), probHi.data());

    CavityFixture fixture;
    fixture.geometry = amrex::Geometry(
        domain,
        &realBox,
        amrex::CoordSys::cartesian,
        periodic.data());

    std::vector<boundary::DomainBoundarySpec> specs(4);
    for (int side = 0; side < 4; ++side) {
        specs[side].name = "cavity_" + std::to_string(side);
        specs[side].location = boundary::AllDomainSides[side];
        specs[side].physical =
            boundary::BoundaryPhysical::NoSlipWall;
        specs[side].scheme = scheme;
    }
    specs[3].physical = boundary::BoundaryPhysical::MovingWall;
    specs[3].profile = boundary::BoundaryProfileKind::Uniform;
    specs[3].hasPrescribedULb = true;
    specs[3].prescribedULb = {0.1, 0.0, 0.0};

    boundary::BoundaryResolveResult boundaryResult =
        boundary::resolve_domain_boundary_specs<Descriptor>(
            specs, fixture.geometry);
    AMREX_ALWAYS_ASSERT(boundaryResult.ok());
    fixture.resolved = std::move(boundaryResult.resolved);
    boundary::PostStreamingBoundaryPlanResult planResult =
        boundary::compile_post_streaming_boundary_plan<Descriptor>(
            fixture.geometry, fixture.resolved.domain);
    AMREX_ALWAYS_ASSERT(planResult.ok());
    fixture.plan = std::move(planResult.plan);
    return fixture;
}

inline config::ResolvedLbmConfig cavity_lbm()
{
    config::ResolvedLbmConfig lbm =
        resolved_lbm(config::LbmParameterizationKind::FromReynolds);
    lbm.lengthRefLb = 5.0;
    lbm.lengthRefPhys = 1.0;
    lbm.nuLb = lbm.uRefLb * lbm.lengthRefLb / lbm.reynolds;
    lbm.tau = 0.5 + lbm.nuLb / (1.0 / 3.0);
    lbm.omega = 1.0 / lbm.tau;
    return lbm;
}

} // namespace setup_integration_test_detail

/**
 * @brief Check production boundary resolution and reference geometry.
 *
 * The tests-owned fixtures verify HBB/Guo stage selection, valid-node
 * geometry and case-agnostic on-lattice reference matching. They launch no
 * kernels and enter no MPI collectives.
 */
template <class Descriptor>
void run_single_level_setup_integration_tests()
{
    using namespace setup_integration_test_detail;

    config::ResolvedLbmConfig const canonicalLbm = cavity_lbm();
    CavityFixture hbb = make_cavity_fixture<Descriptor>(
        boundary::BoundaryScheme::HalfwayBounceBack);
    AMREX_ALWAYS_ASSERT(
        hbb.resolved.domain.size() == 4 &&
        hbb.plan.hasPreStreamingFaces &&
        !hbb.plan.hasPostStreamingFaces &&
        hbb.plan.regions.empty());
    for (auto const& face : hbb.resolved.domain) {
        AMREX_ALWAYS_ASSERT(
            face.scheme == boundary::BoundaryScheme::HalfwayBounceBack &&
            face.stage ==
                boundary::BoundaryExecutionStage::PreStreamingPhysicalGhost);
    }

    CavityFixture guo = make_cavity_fixture<Descriptor>(
        boundary::BoundaryScheme::GuoExtrapolation);
    AMREX_ALWAYS_ASSERT(
        guo.resolved.domain.size() == 4 && !guo.plan.hasPreStreamingFaces &&
        guo.plan.hasPostStreamingFaces && guo.plan.axisGeometry[0].active &&
        guo.plan.axisGeometry[0].validNodes == 6 &&
        guo.plan.axisGeometry[0].latticeIntervals == 5 &&
        guo.plan.axisGeometry[1].active &&
        guo.plan.axisGeometry[1].validNodes == 6 &&
        guo.plan.axisGeometry[1].latticeIntervals == 5);
    for (auto const& face : guo.resolved.domain) {
        AMREX_ALWAYS_ASSERT(
            face.scheme == boundary::BoundaryScheme::GuoExtrapolation &&
            face.stage ==
                boundary::BoundaryExecutionStage::PostStreamingValidNode);
    }

    boundary::OnLatticeReferenceGeometryResolveResult hbbReference =
        boundary::resolve_lbm_reference_geometry(hbb.plan, canonicalLbm);
    AMREX_ALWAYS_ASSERT(hbbReference.ok() && hbbReference.axis == -1);
    boundary::OnLatticeReferenceGeometryResolveResult guoReference =
        boundary::resolve_lbm_reference_geometry(guo.plan, canonicalLbm);
    AMREX_ALWAYS_ASSERT(guoReference.ok() && guoReference.axis == 0);
    config::ResolvedLbmConfig mismatchedReference = canonicalLbm;
    mismatchedReference.lengthRefLb = 4.0;
    AMREX_ALWAYS_ASSERT(
        !boundary::resolve_lbm_reference_geometry(
             guo.plan, mismatchedReference).ok());

    amrex::Print() << "Single-level setup integration tests passed\n";
}

} // namespace lbm::app

#endif
