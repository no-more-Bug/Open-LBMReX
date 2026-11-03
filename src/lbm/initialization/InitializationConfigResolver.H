#ifndef LBM_INITIALIZATION_INITIALIZATIONCONFIGRESOLVER_H
#define LBM_INITIALIZATION_INITIALIZATIONCONFIGRESOLVER_H

#include "lbm/config/LbmConfig.H"
#include "lbm/initialization/InitializationConfigIntent.H"
#include "lbm/initialization/ResolvedInitializationConfig.H"
#include "lbm/initialization/TaylorGreenVortexMath.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>
#include <vector>

namespace lbm::initialization {

struct InitializationConfigResolveResult {
    ResolvedInitializationConfig resolved;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

namespace resolver_detail {

inline void forbid(
    bool provided,
    char const* subject,
    InitializationKind kind,
    std::vector<std::string>& errors)
{
    if (provided) {
        errors.push_back(
            std::string("initialization kind '") + to_string(kind) +
            "' forbids explicit " + subject);
    }
}

inline bool finite_positive(amrex::Real value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

} // namespace resolver_detail

template <class Descriptor>
InitializationConfigResolveResult resolve_initialization_config(
    InitializationConfigIntent const& intent,
    config::ResolvedLbmConfig const& lbm,
    amrex::Array<int, AMREX_SPACEDIM> const& nCell)
{
    using namespace resolver_detail;

    InitializationConfigResolveResult result;
    ResolvedInitializationConfig& resolved = result.resolved;
    std::vector<std::string>& errors = result.errors;
    resolved.kind = intent.kind.value_or(InitializationKind::TaylorGreen);

    if (!intent.kind.has_value()) {
        errors.push_back(
            "initialization kind must be explicitly specified");
    }

    const amrex::Real rhoLb = intent.rhoLb.value_or(lbm.rhoRefLb);
    if (!finite_positive(rhoLb)) {
        errors.push_back(
            "resolved initialization rho_lb must be finite and positive");
    }

    const InitializationKind kind = resolved.kind;
    switch (kind) {
    case InitializationKind::Rest:
        forbid(
            intent.uniformULb.has_value(),
            "uniform velocity",
            kind,
            errors);
        forbid(
            intent.taylorGreenVelocityAmplitudeLb.has_value(),
            "Taylor-Green velocity amplitude",
            kind,
            errors);
        forbid(
            intent.poiseuilleUMaxLb.has_value(),
            "Poiseuille maximum velocity",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXLoLb.has_value(),
            "Poiseuille low-side density",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXHiLb.has_value(),
            "Poiseuille high-side density",
            kind,
            errors);
        resolved.constantState = MacroscopicState{rhoLb, 0.0, 0.0, 0.0};
        break;

    case InitializationKind::Uniform: {
        forbid(
            intent.taylorGreenVelocityAmplitudeLb.has_value(),
            "Taylor-Green velocity amplitude",
            kind,
            errors);
        forbid(
            intent.poiseuilleUMaxLb.has_value(),
            "Poiseuille maximum velocity",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXLoLb.has_value(),
            "Poiseuille low-side density",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXHiLb.has_value(),
            "Poiseuille high-side density",
            kind,
            errors);
        amrex::Array<amrex::Real, 3> velocity = intent.uniformULb.value_or(
            amrex::Array<amrex::Real, 3>{lbm.uRefLb, 0.0, 0.0});
        for (amrex::Real component : velocity) {
            if (!std::isfinite(component)) {
                errors.push_back(
                    "uniform velocity components "
                    "must be finite");
                break;
            }
        }
        for (int axis = Descriptor::d; axis < 3; ++axis) {
            if (std::abs(velocity[axis]) > 0.0) {
                errors.push_back(
                    std::string(Descriptor::name()) +
                    " does not support uniform u_lb component on axis " +
                    std::to_string(axis));
            }
        }
        resolved.constantState =
            MacroscopicState{rhoLb, velocity[0], velocity[1], velocity[2]};
        break;
    }

    case InitializationKind::TaylorGreen: {
        forbid(
            intent.uniformULb.has_value(),
            "uniform velocity",
            kind,
            errors);
        forbid(
            intent.poiseuilleUMaxLb.has_value(),
            "Poiseuille maximum velocity",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXLoLb.has_value(),
            "Poiseuille low-side density",
            kind,
            errors);
        forbid(
            intent.poiseuilleRhoXHiLb.has_value(),
            "Poiseuille high-side density",
            kind,
            errors);
        const amrex::Real amplitude =
            intent.taylorGreenVelocityAmplitudeLb.value_or(lbm.uRefLb);
        if (!std::isfinite(amplitude) || amplitude < 0.0) {
            errors.push_back(
                "Taylor-Green velocity amplitude must be finite and "
                "non-negative");
        }
        if (nCell[0] <= 0 || nCell[1] <= 0) {
            errors.push_back(
                "Taylor-Green initialization requires positive x/y cell counts");
        } else {
            resolved.taylorGreenParams = make_tgv_params(
                nCell[0],
                nCell[1],
                rhoLb,
                amplitude,
                lbm.nuLb,
                Descriptor::cs2(),
                0.0);
        }
        break;
    }

    case InitializationKind::Poiseuille: {
        forbid(
            intent.uniformULb.has_value(),
            "uniform velocity",
            kind,
            errors);
        forbid(
            intent.taylorGreenVelocityAmplitudeLb.has_value(),
            "Taylor-Green velocity amplitude",
            kind,
            errors);
        PoiseuilleInitializationParams& params = resolved.poiseuilleParams;
        params.rhoLb = rhoLb;
        params.uMaxLb = intent.poiseuilleUMaxLb.value_or(lbm.uRefLb);
        if (!finite_positive(params.uMaxLb)) {
            errors.push_back(
                "Poiseuille maximum velocity must be finite and positive");
        }
        const bool hasXLo = intent.poiseuilleRhoXLoLb.has_value();
        const bool hasXHi = intent.poiseuilleRhoXHiLb.has_value();
        if (hasXLo != hasXHi) {
            errors.push_back(
                "Poiseuille low- and high-side densities must be specified "
                "together");
        }
        if (hasXLo && hasXHi) {
            params.linearDensity = 1;
            params.rhoXLoLb = *intent.poiseuilleRhoXLoLb;
            params.rhoXHiLb = *intent.poiseuilleRhoXHiLb;
            if (!finite_positive(params.rhoXLoLb) ||
                !finite_positive(params.rhoXHiLb)) {
                errors.push_back(
                    "Poiseuille endpoint densities must be finite and positive");
            }
        } else {
            params.linearDensity = 0;
            params.rhoXLoLb = rhoLb;
            params.rhoXHiLb = rhoLb;
        }
        break;
    }
    }

    return result;
}

} // namespace lbm::initialization

#endif
