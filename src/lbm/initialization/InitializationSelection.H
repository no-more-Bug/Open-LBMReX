#ifndef LBM_INITIALIZATION_INITIALIZATIONSELECTION_H
#define LBM_INITIALIZATION_INITIALIZATIONSELECTION_H

#include "lbm/config/LbmConfig.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/initialization/InitializationConfigIntent.H"
#include "lbm/initialization/InitializationConfigResolver.H"
#include "lbm/initialization/MacroscopicState.H"
#include "lbm/initialization/SingleLevelInitializer.H"
#include "lbm/operators/level/EquilibriumPopulationInitializationOperator.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace lbm::initialization {

template <class Initialization>
struct InitializationPreparationResult {
    std::optional<Initialization> initialization;
    std::vector<std::string> errors;

    bool ok() const noexcept
    {
        return initialization.has_value() && errors.empty();
    }
};

class ConfiguredInitialization {
public:
    ConfiguredInitialization(
        InitializationConfigIntent intent,
        ResolvedInitializationConfig resolved)
        : intent_(std::move(intent)),
          resolved_(std::move(resolved))
    {}

    InitializationConfigIntent const& intent() const noexcept
    {
        return intent_;
    }

    ResolvedInitializationConfig const& resolved() const noexcept
    {
        return resolved_;
    }

private:
    InitializationConfigIntent intent_;
    ResolvedInitializationConfig resolved_;
};

/** Parser-free selection for one of the reusable built-in initialization kinds. */
struct ConfiguredInitializationSelection {
    InitializationConfigIntent intent;
};

template <class Profile>
class ProfileInitialization {
public:
    static_assert(
        std::is_trivially_copyable_v<Profile>,
        "Initialization profile must be trivially copyable for device capture");
    static_assert(
        std::is_nothrow_invocable_r_v<
            MacroscopicState,
            Profile const&,
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const&,
            amrex::Real>,
        "Initialization profile must be a noexcept device-safe macroscopic functor");

    explicit ProfileInitialization(Profile profile)
        : profile_(std::move(profile))
    {}

    Profile const& profile() const noexcept
    {
        return profile_;
    }

private:
    Profile profile_;
};

/** Preserve configured kind identity without weakening direct typed profiles. */
inline bool initialization_kind_matches(
    ConfiguredInitialization const& initialization,
    InitializationKind expected) noexcept
{
    return initialization.resolved().kind == expected;
}

template <class Profile>
constexpr bool initialization_kind_matches(
    ProfileInitialization<Profile> const&,
    InitializationKind) noexcept
{
    // A direct profile carries its scientific identity in its C++ type.
    return true;
}

/**
 * Read-only initialization observation shared by direct and configured paths.
 *
 * These overloads expose already-owned values; they do not resolve or copy
 * case policy and are valid only while the prepared initialization lives.
 */
inline MacroscopicState const& constant_initialization_state(
    ConfiguredInitialization const& initialization) noexcept
{
    return initialization.resolved().constantState;
}

template <class Profile>
auto const& constant_initialization_state(
    ProfileInitialization<Profile> const& initialization) noexcept
{
    return initialization.profile().state;
}

inline PoiseuilleInitializationParams const&
poiseuille_initialization_params(
    ConfiguredInitialization const& initialization) noexcept
{
    return initialization.resolved().poiseuilleParams;
}

template <class Profile>
auto const& poiseuille_initialization_params(
    ProfileInitialization<Profile> const& initialization) noexcept
{
    return initialization.profile().params;
}

/** Direct C++ selection for an already constructed device-safe profile. */
template <class Profile>
struct ProfileInitializationSelection {
    Profile profile;
};

template <class Profile>
ProfileInitializationSelection<Profile> from_profile(Profile profile)
{
    return {std::move(profile)};
}

namespace selection_detail {

template <class Descriptor, class Selection>
struct PreparedInitialization;

template <class Descriptor>
struct PreparedInitialization<Descriptor, ConfiguredInitializationSelection> {
    using type = ConfiguredInitialization;
};

template <class Descriptor, class Profile>
struct PreparedInitialization<
    Descriptor,
    ProfileInitializationSelection<Profile>> {
    using type = ProfileInitialization<Profile>;
};

} // namespace selection_detail

template <class Descriptor, class Selection>
using PreparedInitialization =
    typename selection_detail::PreparedInitialization<
        Descriptor,
        Selection>::type;

template <class Descriptor>
InitializationPreparationResult<ConfiguredInitialization>
prepare_initialization(
    ConfiguredInitializationSelection const& selection,
    config::ResolvedLbmConfig const& lbm,
    amrex::Array<int, AMREX_SPACEDIM> const& nCell)
{
    InitializationPreparationResult<ConfiguredInitialization> result;
    InitializationConfigResolveResult resolved =
        resolve_initialization_config<Descriptor>(
            selection.intent, lbm, nCell);
    if (!resolved.ok()) {
        result.errors = std::move(resolved.errors);
        return result;
    }
    result.initialization.emplace(
        selection.intent, std::move(resolved.resolved));
    return result;
}

template <class Descriptor, class Profile>
InitializationPreparationResult<ProfileInitialization<Profile>>
prepare_initialization(
    ProfileInitializationSelection<Profile> const& selection,
    config::ResolvedLbmConfig const&,
    amrex::Array<int, AMREX_SPACEDIM> const&)
{
    InitializationPreparationResult<ProfileInitialization<Profile>> result;
    result.initialization.emplace(selection.profile);
    return result;
}

template <class Descriptor>
void initialize_populations(
    ConfiguredInitialization const& initialization,
    LatticeField<Descriptor>& populations,
    amrex::Geometry const& geometry)
{
    SingleLevelInitializer<Descriptor>{}.initialize(
        populations, geometry, initialization.resolved());
}

template <class Descriptor, class Profile>
void initialize_populations(
    ProfileInitialization<Profile> const& initialization,
    LatticeField<Descriptor>& populations,
    amrex::Geometry const& geometry)
{
    operators::EquilibriumPopulationInitializationOperator<
        Descriptor,
        Profile> operation(initialization.profile(), geometry);
    operation.apply(populations, 0.0);
}

} // namespace lbm::initialization

#endif
