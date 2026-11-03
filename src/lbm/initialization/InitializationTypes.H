#ifndef LBM_INITIALIZATION_INITIALIZATIONTYPES_H
#define LBM_INITIALIZATION_INITIALIZATIONTYPES_H

#include <optional>
#include <string>

namespace lbm::initialization {

enum class InitializationKind {
    Rest,
    Uniform,
    TaylorGreen,
    Poiseuille
};

inline char const* to_string(InitializationKind kind) noexcept
{
    switch (kind) {
    case InitializationKind::Rest:
        return "rest";
    case InitializationKind::Uniform:
        return "uniform";
    case InitializationKind::TaylorGreen:
        return "taylor_green";
    case InitializationKind::Poiseuille:
        return "poiseuille";
    }
    return "unknown";
}

inline std::optional<InitializationKind>
parse_initialization_kind(std::string const& value)
{
    if (value == "rest") {
        return InitializationKind::Rest;
    }
    if (value == "uniform") {
        return InitializationKind::Uniform;
    }
    if (value == "taylor_green") {
        return InitializationKind::TaylorGreen;
    }
    if (value == "poiseuille") {
        return InitializationKind::Poiseuille;
    }
    return std::nullopt;
}

} // namespace lbm::initialization

#endif
