#ifndef LBM_TESTS_UNIT_LBM_LBMCONFIGURATIONTESTS_H
#define LBM_TESTS_UNIT_LBM_LBMCONFIGURATIONTESTS_H

#include "lbm/config/LbmConfigResolver.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>
#include <utility>

namespace lbm::config::tests {

namespace resolver_test_detail {

template <class T>
void provide(
    input::ParsedInputValue<T>& field,
    T value)
{
    field.value = std::move(value);
    field.source = input::InputValueSource::Canonical;
}

inline RawLbmConfig base_raw(LbmParameterizationKind kind)
{
    RawLbmConfig raw;
    provide(
        raw.parameterizationKind,
        kind);
    raw.rhoRefLb.value = 1.0;
    raw.uRefLb.value = 0.01;
    raw.uRefPhys.value = 1.0;
    raw.lengthRefLb.value = -1.0;
    raw.lengthRefPhys.value = -1.0;
    raw.nuLb.value = 0.01;
    raw.reynolds.value = 10.0;
    raw.tau.value = 0.75;
    raw.streamingBackend.value = "baseline";
    return raw;
}

inline amrex::Array<int, AMREX_SPACEDIM> mesh_cells()
{
    return {AMREX_D_DECL(10, 10, 4)};
}

inline amrex::Array<amrex::Real, AMREX_SPACEDIM> mesh_lo()
{
    return {AMREX_D_DECL(0.0, 0.0, 0.0)};
}

inline amrex::Array<amrex::Real, AMREX_SPACEDIM> mesh_hi()
{
    return {AMREX_D_DECL(1.0, 1.0, 0.4)};
}

inline void provide_lengths(RawLbmConfig& raw)
{
    provide(raw.lengthRefLb, amrex::Real(10.0));
    provide(
        raw.lengthRefPhys,
        amrex::Real(1.0));
}

} // namespace resolver_test_detail

/**
 * @brief Exercise the explicit LBM strategy matrix without boundary input.
 *
 * These checks cover every strategy's required and forbidden authorities,
 * reference closure in both directions, redundant consistency, length/mesh
 * semantics, and provenance of derived values.
 * They perform no MPI communication and allocate no AMReX fields.
 */
template <class Descriptor>
void run_lbm_configuration_tests()
{
    using namespace resolver_test_detail;
    const auto cells = mesh_cells();
    const auto lo = mesh_lo();
    const auto hi = mesh_hi();

    AMREX_ALWAYS_ASSERT(detail::nearly_equal(
        Descriptor::cs2(), amrex::Real(1.0 / 3.0)));

    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        auto result = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(result.ok());
        const amrex::Real expectedNu = 0.05;
        const amrex::Real expectedTau =
            0.5 + expectedNu / Descriptor::cs2();
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.reynolds, 20.0));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.nuLb, expectedNu));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.tau, expectedTau));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(
                result.resolved.omega, 1.0 / expectedTau));
        AMREX_ALWAYS_ASSERT(
            result.resolved.sources.reynoldsDerived);
        AMREX_ALWAYS_ASSERT(
            result.resolved.sources.tauDerived);
        AMREX_ALWAYS_ASSERT(
            !result.resolved.sources.lengthReferenceExplicit);
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.reynolds, amrex::Real(20.0));
        auto result = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(result.ok());
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.uRefLb, 0.1));
        AMREX_ALWAYS_ASSERT(result.resolved.sources.uRefLbDerived);
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        provide(raw.reynolds, amrex::Real(20.0));
        AMREX_ALWAYS_ASSERT(
            resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        raw.reynolds.value = 21.0;
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromTau);
        provide(raw.tau, amrex::Real(0.8));
        provide(raw.reynolds, amrex::Real(12.0));
        provide_lengths(raw);
        auto result = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(result.ok());
        const amrex::Real expectedNu =
            Descriptor::cs2() * amrex::Real(0.3);
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.nuLb, expectedNu));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.tau, 0.8));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.omega, 1.0 / 0.8));
        AMREX_ALWAYS_ASSERT(detail::nearly_equal(
            result.resolved.uRefLb,
            amrex::Real(12.0) * expectedNu / amrex::Real(10.0)));
        AMREX_ALWAYS_ASSERT(result.resolved.sources.nuLbDerived);
        AMREX_ALWAYS_ASSERT(result.resolved.sources.uRefLbDerived);
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromTau);
        provide(raw.tau, amrex::Real(0.8));
        provide(raw.uRefLb, amrex::Real(0.12));
        auto result = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(result.ok());
        AMREX_ALWAYS_ASSERT(detail::nearly_equal(
            result.resolved.reynolds,
            result.resolved.uRefLb * result.resolved.lengthRefLb /
                result.resolved.nuLb));
        provide(raw.reynolds, amrex::Real(12.0));
        AMREX_ALWAYS_ASSERT(
            resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        raw.reynolds.value = 13.0;
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromReynolds);
        provide(raw.uRefLb, amrex::Real(0.1));
        provide(raw.reynolds, amrex::Real(20.0));
        provide_lengths(raw);
        auto result = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(result.ok());
        const amrex::Real expectedNu = 0.05;
        const amrex::Real expectedTau =
            0.5 + expectedNu / Descriptor::cs2();
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.nuLb, expectedNu));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(result.resolved.tau, expectedTau));
        AMREX_ALWAYS_ASSERT(
            detail::nearly_equal(
                result.resolved.omega, 1.0 / expectedTau));
        AMREX_ALWAYS_ASSERT(result.resolved.sources.nuLbDerived);
        AMREX_ALWAYS_ASSERT(result.resolved.sources.tauDerived);

        // No boundary object or value participates in either resolution.
        auto repeated = resolve_lbm_config<Descriptor>(raw, cells, lo, hi);
        AMREX_ALWAYS_ASSERT(repeated.ok());
        AMREX_ALWAYS_ASSERT(
            repeated.resolved.uRefLb == result.resolved.uRefLb &&
            repeated.resolved.reynolds == result.resolved.reynolds &&
            repeated.resolved.nuLb == result.resolved.nuLb &&
            repeated.resolved.tau == result.resolved.tau &&
            repeated.resolved.omega == result.resolved.omega);
    }

    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        raw.parameterizationKind.source =
            input::InputValueSource::Default;
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.uRefLb, amrex::Real(0.1));
        // The positive struct default is not an explicit nu authority.
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.tau, amrex::Real(0.65));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        raw.nuLb.value = -0.05;
        provide(raw.uRefLb, amrex::Real(0.1));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromTau);
        provide(raw.reynolds, amrex::Real(12.0));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        provide(raw.tau, amrex::Real(0.5));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        raw.tau.value = 0.8;
        provide(raw.nuLb, amrex::Real(0.1));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromTau);
        provide(raw.tau, amrex::Real(0.8));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromReynolds);
        provide(raw.uRefLb, amrex::Real(0.1));
        provide(raw.reynolds, amrex::Real(20.0));
        // from_reynolds cannot use the otherwise valid x-envelope default.
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        provide_lengths(raw);
        provide(raw.nuLb, amrex::Real(0.05));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        raw.nuLb = {};
        raw.nuLb.value = 0.01;
        provide(raw.tau, amrex::Real(0.65));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        RawLbmConfig missingU =
            base_raw(LbmParameterizationKind::FromReynolds);
        provide(missingU.reynolds, amrex::Real(20.0));
        provide_lengths(missingU);
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(
                missingU, cells, lo, hi).ok());

        RawLbmConfig missingRe =
            base_raw(LbmParameterizationKind::FromReynolds);
        provide(missingRe.uRefLb, amrex::Real(0.1));
        provide_lengths(missingRe);
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(
                missingRe, cells, lo, hi).ok());
    }
    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        provide(raw.lengthRefLb, amrex::Real(10.0));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
        provide(
            raw.lengthRefPhys,
            amrex::Real(2.0));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, cells, lo, hi).ok());
    }
    {
        auto badHi = hi;
        badHi[1] *= 2.0;
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(
                raw, cells, lo, badHi).ok());
    }

    {
        RawLbmConfig raw = base_raw(LbmParameterizationKind::FromNuLb);
        provide(raw.nuLb, amrex::Real(0.05));
        provide(raw.uRefLb, amrex::Real(0.1));
        provide_lengths(raw);
        auto onePlane = cells;
        onePlane[2] = 1;
        for (amrex::Real thickness : {amrex::Real(0.1), amrex::Real(0.2)}) {
            auto bounds = hi;
            bounds[2] = thickness;
            auto result = resolve_lbm_config<Descriptor>(raw, onePlane, lo, bounds);
            AMREX_ALWAYS_ASSERT(result.ok() ==
                (Descriptor::d == 2 || thickness == amrex::Real(0.1)));
            if (result.ok()) {
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.lengthRefLb, 10.0));
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.lengthRefPhys, 1.0));
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.nuLb, 0.05));
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.tau, 0.65));
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.uRefLb, 0.1));
                AMREX_ALWAYS_ASSERT(detail::nearly_equal(result.resolved.reynolds, 20.0));
            }
        }
        for (amrex::Real thickness : {amrex::Real(0.0), amrex::Real(-0.1),
                 std::numeric_limits<amrex::Real>::infinity(),
                 std::numeric_limits<amrex::Real>::quiet_NaN()}) {
            auto bounds = hi;
            bounds[2] = thickness;
            AMREX_ALWAYS_ASSERT(
                !resolve_lbm_config<Descriptor>(raw, onePlane, lo, bounds).ok());
        }
        auto overflowLo = lo;
        auto overflowHi = hi;
        overflowLo[2] = -std::numeric_limits<amrex::Real>::max();
        overflowHi[2] = std::numeric_limits<amrex::Real>::max();
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, onePlane, overflowLo, overflowHi).ok());

        auto thickHi = hi;
        thickHi[2] = 0.2;
        raw.lengthRefPhys.value = 2.0; // Matching dz must not replace the xy reference.
        AMREX_ALWAYS_ASSERT(
            !resolve_lbm_config<Descriptor>(raw, onePlane, lo, thickHi).ok());
    }

    amrex::Print() << "LBM configuration unit tests passed\n";
}

} // namespace lbm::config::tests

#endif
