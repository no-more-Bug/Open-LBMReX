#ifndef LBM_CONFIG_LBMCONFIGINTENT_H
#define LBM_CONFIG_LBMCONFIGINTENT_H

#include <AMReX_REAL.H>

#include <optional>
#include <string>

namespace lbm::config {

enum class LbmParameterizationKind {
    FromNuLb,
    FromTau,
    FromReynolds
};

inline char const* to_string(LbmParameterizationKind kind) noexcept
{
    switch (kind) {
    case LbmParameterizationKind::FromNuLb:
        return "from_nu_lb";
    case LbmParameterizationKind::FromTau:
        return "from_tau";
    case LbmParameterizationKind::FromReynolds:
        return "from_reynolds";
    }
    return "unknown";
}

inline std::optional<LbmParameterizationKind>
parse_lbm_parameterization_kind(std::string const& value)
{
    if (value == "from_nu_lb") return LbmParameterizationKind::FromNuLb;
    if (value == "from_tau") return LbmParameterizationKind::FromTau;
    if (value == "from_reynolds") return LbmParameterizationKind::FromReynolds;
    return std::nullopt;
}

/**
 * @brief Parser-free numerical intent consumed by single-level preparation.
 *
 * Empty optionals mean that the established general default is requested.
 * Presence remains semantic for parameterization closure, but canonical/legacy
 * keyword provenance does not cross the runtime assembly seam.
 */
struct LbmConfigIntent {
    std::optional<LbmParameterizationKind> parameterizationKind;
    std::optional<amrex::Real> rhoRefLb;
    std::optional<amrex::Real> uRefLb;
    std::optional<amrex::Real> uRefPhys;
    std::optional<amrex::Real> lengthRefLb;
    std::optional<amrex::Real> lengthRefPhys;
    std::optional<amrex::Real> nuLb;
    std::optional<amrex::Real> reynolds;
    std::optional<amrex::Real> tau;
    std::optional<std::string> streamingBackend;
};

} // namespace lbm::config

#endif
