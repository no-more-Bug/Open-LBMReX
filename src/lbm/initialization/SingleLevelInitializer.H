#ifndef LBM_INITIALIZATION_SINGLELEVELINITIALIZER_H
#define LBM_INITIALIZATION_SINGLELEVELINITIALIZER_H

#include "lbm/fields/LatticeField.H"
#include "lbm/initialization/ConstantInitializationProfile.H"
#include "lbm/initialization/PoiseuilleInitializationProfile.H"
#include "lbm/initialization/ResolvedInitializationConfig.H"
#include "lbm/initialization/TaylorGreenVortexProfile.H"
#include "lbm/operators/level/EquilibriumPopulationInitializationOperator.H"

#include <AMReX_BLassert.H>
#include <AMReX_Geometry.H>

namespace lbm::initialization {

/**
 * @brief Host-side dispatcher for built-in single-level bulk initialization.
 *
 * The dispatcher selects one device-safe macroscopic profile from an already
 * resolved setup. Every branch delegates valid-cell equilibrium reconstruction
 * and FillBoundary to EquilibriumPopulationInitializationOperator; this façade
 * launches no field kernel and writes no MultiFab directly.
 *
 * @pre setup was produced by InitializationConfigResolver for geom's domain.
 * @post all valid populations contain f_eq(rho^0,u^0), and inter-FAB/periodic
 *       ghosts have been synchronized. Non-periodic physical ghosts carry no
 *       boundary-condition meaning at this point.
 */
template <class Descriptor>
class SingleLevelInitializer {
public:
    void initialize(
        LatticeField<Descriptor>& populations,
        amrex::Geometry const& geom,
        ResolvedInitializationConfig const& setup) const
    {
        switch (setup.kind) {
        case InitializationKind::Rest:
        case InitializationKind::Uniform:
            apply_profile(
                populations,
                geom,
                ConstantInitializationProfile{setup.constantState});
            return;
        case InitializationKind::TaylorGreen:
            apply_profile(
                populations,
                geom,
                TaylorGreenVortexProfile{setup.taylorGreenParams});
            return;
        case InitializationKind::Poiseuille:
            apply_profile(
                populations,
                geom,
                PoiseuilleInitializationProfile{setup.poiseuilleParams});
            return;
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            false, "Unsupported resolved initialization kind");
    }

private:
    template <class Profile>
    static void apply_profile(
        LatticeField<Descriptor>& populations,
        amrex::Geometry const& geom,
        Profile profile)
    {
        operators::EquilibriumPopulationInitializationOperator<
            Descriptor,
            Profile> initialization(profile, geom);
        initialization.apply(populations, 0.0);
    }
};

} // namespace lbm::initialization

#endif
