#ifndef LBM_CONFIG_LBMCONFIGRESOLVER_H
#define LBM_CONFIG_LBMCONFIGRESOLVER_H

#include "lbm/config/LbmConfig.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lbm::config {

namespace intent_detail {

template <class T>
input::ParsedInputValue<T> as_parsed_value(
    std::optional<T> const& value,
    T defaultValue)
{
    input::ParsedInputValue<T> result;
    result.value = value.has_value() ? *value : std::move(defaultValue);
    result.source = value.has_value() ? input::InputValueSource::Canonical :
                                        input::InputValueSource::Default;
    return result;
}

inline RawLbmConfig as_raw_config(LbmConfigIntent const& intent)
{
    RawLbmConfig raw;
    raw.parameterizationKind = as_parsed_value(
        intent.parameterizationKind,
        LbmParameterizationKind::FromNuLb);
    raw.rhoRefLb = as_parsed_value(intent.rhoRefLb, amrex::Real(1.0));
    raw.uRefLb = as_parsed_value(intent.uRefLb, amrex::Real(0.01));
    raw.uRefPhys = as_parsed_value(intent.uRefPhys, amrex::Real(1.0));
    raw.lengthRefLb = as_parsed_value(intent.lengthRefLb, amrex::Real(-1.0));
    raw.lengthRefPhys =
        as_parsed_value(intent.lengthRefPhys, amrex::Real(-1.0));
    raw.nuLb = as_parsed_value(intent.nuLb, amrex::Real(0.01));
    raw.reynolds = as_parsed_value(intent.reynolds, amrex::Real(10.0));
    raw.tau = as_parsed_value(intent.tau, amrex::Real(0.75));
    raw.streamingBackend =
        as_parsed_value(intent.streamingBackend, std::string("baseline"));
    return raw;
}

} // namespace intent_detail

struct LbmConfigResolveResult {
    ResolvedLbmConfig resolved;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

namespace detail {

inline bool nearly_equal(amrex::Real lhs, amrex::Real rhs) noexcept
{
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const amrex::Real scale = std::max(
        std::max(std::abs(lhs), std::abs(rhs)), amrex::Real(1.0));
    return std::abs(lhs - rhs) <=
           64.0 * std::numeric_limits<amrex::Real>::epsilon() * scale;
}

inline bool finite_positive(amrex::Real value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

inline void require_source(
    bool provided,
    char const* key,
    LbmParameterizationKind kind,
    std::vector<std::string>& errors)
{
    if (!provided) {
        errors.push_back(
            std::string("lbm.parameterization.kind=") + to_string(kind) +
            " requires explicit " + key);
    }
}

inline void forbid_source(
    bool provided,
    char const* key,
    LbmParameterizationKind kind,
    std::vector<std::string>& errors)
{
    if (provided) {
        errors.push_back(
            std::string("lbm.parameterization.kind=") + to_string(kind) +
            " forbids explicit " + key);
    }
}

} // namespace detail

/**
 * @brief Close one single-level LBM numerical configuration by explicit strategy.
 *
 * The resolver consumes syntax-level LBM values and the mesh envelope needed
 * for the lattice/physical length conversion. It never reads application
 * names, benchmark data, boundary actual values, or a compiled boundary plan.
 * Explicit-presence metadata controls every required/forbidden decision;
 * defaults cannot masquerade as authoritative inputs.
 *
 * @tparam Descriptor Lattice model supplying the compile-time cs2 constant.
 * @pre Arrays describe the same cell-centered Cartesian mesh that will be used
 *      to construct AMReX Geometry.
 * @post On success rho/u/Re/nu/tau/omega and both reference lengths are finite,
 *       positive, mutually consistent, and immutable to downstream consumers.
 */
template <class Descriptor>
LbmConfigResolveResult resolve_lbm_config(
    RawLbmConfig const& raw,
    amrex::Array<int, AMREX_SPACEDIM> const& nCell,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probLo,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probHi)
{
    LbmConfigResolveResult result;
    ResolvedLbmConfig& resolved = result.resolved;
    std::vector<std::string>& errors = result.errors;

    resolved.parameterizationKind = raw.parameterizationKind.value;
    resolved.rhoRefLb = raw.rhoRefLb.value;
    resolved.uRefLb = raw.uRefLb.value;
    resolved.uRefPhys = raw.uRefPhys.value;
    resolved.nuLb = raw.nuLb.value;
    resolved.reynolds = raw.reynolds.value;
    resolved.tau = raw.tau.value;
    resolved.streamingBackend = raw.streamingBackend.value;

    const amrex::Real cs2 = Descriptor::cs2();
    const bool modelConstantValid =
        std::isfinite(cs2) && cs2 > 0.0;
    if (!modelConstantValid) {
        errors.push_back("Descriptor::cs2() must be finite and positive");
    }

    if (!raw.parameterizationKind.provided()) {
        errors.push_back(
            "lbm.parameterization.kind must be explicitly specified");
    }

    bool meshValid = true;
    amrex::Real physicalDx = 0.0;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (nCell[d] <= 0 || !std::isfinite(probLo[d]) ||
            !std::isfinite(probHi[d]) || !(probHi[d] > probLo[d])) {
            errors.push_back(
                "mesh requires positive amr.n_cell and finite increasing "
                "geometry.prob_lo/prob_hi in every direction");
            meshValid = false;
            break;
        }
        const amrex::Real dx =
            (probHi[d] - probLo[d]) / static_cast<amrex::Real>(nCell[d]);
        if (!detail::finite_positive(dx)) {
            errors.push_back("mesh requires finite positive physical cell sizes");
            meshValid = false;
            break;
        }
        if (d == 0) {
            physicalDx = dx;
        } else if (d < Descriptor::d &&
                   !detail::nearly_equal(dx, physicalDx)) {
            errors.push_back(
                "LBM requires uniform Cartesian physical cell sizes: "
                "(prob_hi[d]-prob_lo[d])/n_cell[d] must match in every active direction");
            meshValid = false;
            break;
        }
    }

    const bool hasLengthRefLb = raw.lengthRefLb.provided();
    const bool hasLengthRefPhys = raw.lengthRefPhys.provided();
    if (hasLengthRefLb != hasLengthRefPhys) {
        errors.push_back(
            "lbm.length_ref_lb and lbm.length_ref_phys must be specified together");
    }
    resolved.sources.lengthReferenceExplicit =
        hasLengthRefLb && hasLengthRefPhys;
    if (resolved.sources.lengthReferenceExplicit) {
        resolved.lengthRefLb = raw.lengthRefLb.value;
        resolved.lengthRefPhys = raw.lengthRefPhys.value;
    } else if (meshValid) {
        resolved.lengthRefLb = static_cast<amrex::Real>(nCell[0]);
        resolved.lengthRefPhys = probHi[0] - probLo[0];
    }

    if (!detail::finite_positive(resolved.lengthRefLb) ||
        !detail::finite_positive(resolved.lengthRefPhys)) {
        errors.push_back(
            "resolved LBM reference lengths must be finite and positive");
    } else if (meshValid &&
               !detail::nearly_equal(
                   resolved.lengthRefPhys / resolved.lengthRefLb,
                   physicalDx)) {
        errors.push_back(
            "lbm.length_ref_phys/lbm.length_ref_lb must match the AMReX "
            "physical cell size");
    }

    if (!detail::finite_positive(resolved.rhoRefLb)) {
        errors.push_back("lbm.rho_ref_lb must be finite and positive");
    }
    if (!detail::finite_positive(resolved.uRefPhys)) {
        errors.push_back("lbm.u_ref_phys must be finite and positive");
    }
    if (resolved.streamingBackend.empty()) {
        errors.push_back("lbm.streaming_backend must not be empty");
    }

    const LbmParameterizationKind kind =
        resolved.parameterizationKind;
    const bool hasURef = raw.uRefLb.provided();
    const bool hasRe = raw.reynolds.provided();
    const bool hasNu = raw.nuLb.provided();
    const bool hasTau = raw.tau.provided();
    auto require_reference_closure = [&] {
        if (!hasURef && !hasRe) {
            errors.push_back(
                std::string("lbm.parameterization.kind=") + to_string(kind) +
                " requires at least one explicit lbm.u_ref_lb or lbm.re");
        }
    };
    auto validate_explicit_reference_values = [&] {
        if (hasURef && !detail::finite_positive(raw.uRefLb.value)) {
            errors.push_back("lbm.u_ref_lb must be finite and positive");
        }
        if (hasRe && !detail::finite_positive(raw.reynolds.value)) {
            errors.push_back("lbm.re must be finite and positive");
        }
    };
    auto close_reference_relation = [&] {
        if (!detail::finite_positive(resolved.nuLb) ||
            !detail::finite_positive(resolved.lengthRefLb)) {
            return;
        }
        if (hasURef && hasRe) {
            const amrex::Real expectedRe =
                raw.uRefLb.value * resolved.lengthRefLb / resolved.nuLb;
            if (detail::finite_positive(raw.uRefLb.value) &&
                detail::finite_positive(raw.reynolds.value) &&
                !detail::nearly_equal(raw.reynolds.value, expectedRe)) {
                errors.push_back(
                    "explicit lbm.u_ref_lb and lbm.re are inconsistent with "
                    "the selected viscosity and lbm.length_ref_lb");
            }
        } else if (hasURef &&
                   detail::finite_positive(raw.uRefLb.value)) {
            resolved.uRefLb = raw.uRefLb.value;
            resolved.reynolds =
                resolved.uRefLb * resolved.lengthRefLb / resolved.nuLb;
            resolved.sources.reynoldsDerived = true;
        } else if (hasRe &&
                   detail::finite_positive(raw.reynolds.value)) {
            resolved.reynolds = raw.reynolds.value;
            resolved.uRefLb =
                resolved.reynolds * resolved.nuLb /
                resolved.lengthRefLb;
            resolved.sources.uRefLbDerived = true;
        }
    };

    switch (kind) {
    case LbmParameterizationKind::FromNuLb:
        detail::require_source(hasNu, "lbm.nu_lb", kind, errors);
        detail::forbid_source(hasTau, "lbm.tau", kind, errors);
        require_reference_closure();
        validate_explicit_reference_values();
        if (hasNu) {
            resolved.nuLb = raw.nuLb.value;
            if (!detail::finite_positive(resolved.nuLb)) {
                errors.push_back("lbm.nu_lb must be finite and positive");
            }
        }
        close_reference_relation();
        if (detail::finite_positive(resolved.nuLb) &&
            modelConstantValid) {
            resolved.tau =
                0.5 + resolved.nuLb / cs2;
            resolved.sources.tauDerived = true;
        }
        break;

    case LbmParameterizationKind::FromTau:
        detail::require_source(hasTau, "lbm.tau", kind, errors);
        detail::forbid_source(hasNu, "lbm.nu_lb",
                              kind, errors);
        require_reference_closure();
        validate_explicit_reference_values();
        if (hasTau) {
            resolved.tau = raw.tau.value;
            if (!std::isfinite(resolved.tau) ||
                !(resolved.tau > 0.5)) {
                errors.push_back(
                    "lbm.tau must be finite and greater than 0.5");
            } else {
                resolved.nuLb = cs2 * (resolved.tau - 0.5);
                resolved.sources.nuLbDerived = true;
            }
        }
        close_reference_relation();
        break;

    case LbmParameterizationKind::FromReynolds:
        detail::require_source(hasRe, "lbm.re", kind, errors);
        detail::require_source(hasURef, "lbm.u_ref_lb", kind, errors);
        detail::require_source(
            resolved.sources.lengthReferenceExplicit,
            "lbm.length_ref_lb and lbm.length_ref_phys",
            kind,
            errors);
        detail::forbid_source(hasNu, "lbm.nu_lb",
                              kind, errors);
        detail::forbid_source(hasTau, "lbm.tau", kind, errors);
        validate_explicit_reference_values();
        if (hasURef && hasRe && modelConstantValid &&
            detail::finite_positive(raw.uRefLb.value) &&
            detail::finite_positive(raw.reynolds.value) &&
            detail::finite_positive(resolved.lengthRefLb)) {
            resolved.uRefLb = raw.uRefLb.value;
            resolved.reynolds = raw.reynolds.value;
            resolved.nuLb =
                resolved.uRefLb * resolved.lengthRefLb /
                resolved.reynolds;
            resolved.sources.nuLbDerived = true;
            resolved.tau =
                0.5 + resolved.nuLb / cs2;
            resolved.sources.tauDerived = true;
        }
        break;
    }

    if (!detail::finite_positive(resolved.uRefLb)) {
        errors.push_back("resolved lbm.u_ref_lb must be finite and positive");
    }
    if (!detail::finite_positive(resolved.reynolds)) {
        errors.push_back("resolved lbm.re must be finite and positive");
    }
    if (!detail::finite_positive(resolved.nuLb)) {
        errors.push_back("resolved lbm.nu_lb must be finite and positive");
    }
    if (!std::isfinite(resolved.tau) || !(resolved.tau > 0.5)) {
        errors.push_back(
            "resolved lbm.tau must be finite and greater than 0.5");
    } else {
        resolved.omega = 1.0 / resolved.tau;
        if (!std::isfinite(resolved.omega) ||
            !(resolved.omega > 0.0 && resolved.omega < 2.0)) {
            errors.push_back("resolved BGK omega must be in (0, 2)");
        }
    }

    return result;
}

/** Resolve parser-free numerical intent through the same closure implementation. */
template <class Descriptor>
LbmConfigResolveResult resolve_lbm_config(
    LbmConfigIntent const& intent,
    amrex::Array<int, AMREX_SPACEDIM> const& nCell,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probLo,
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const& probHi)
{
    return resolve_lbm_config<Descriptor>(
        intent_detail::as_raw_config(intent), nCell, probLo, probHi);
}

} // namespace lbm::config

#endif
