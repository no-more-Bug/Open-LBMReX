#ifndef LBM_TESTS_INTEGRATION_LBM_LBMDIAGNOSTICSINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_LBMDIAGNOSTICSINTEGRATIONTESTS_H

#include "lbm/diagnostics/SteadyVelocityResidual.H"
#include "lbm/fields/MacroField.H"

#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace lbm::tests::integration {

/**
 * @brief Exercise case-agnostic diagnostics primitives on every MPI rank.
 *
 * MultiFab reductions inside compute_relative_l2_delta_u are collective, so
 * callers must enter this check symmetrically.
 */
template <class Descriptor>
void run_diagnostics_primitive_integration_tests()
{
    amrex::Box const domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(3, 2, 0)));
    amrex::BoxArray const boxArray(domain);
    amrex::DistributionMapping const distribution(boxArray);
    MacroField previous("diagnostics_previous", boxArray, distribution);
    MacroField current("diagnostics_current", boxArray, distribution);
    MacroField copied("diagnostics_copied", boxArray, distribution);

    previous.multifab().setVal(0.0);
    current.multifab().setVal(0.0);
    copied.multifab().setVal(0.0);
    previous.multifab().setVal(1.0, MacroField::Ux, 1, 0);
    previous.multifab().setVal(2.0, MacroField::Uy, 1, 0);
    previous.multifab().setVal(0.5, MacroField::Uz, 1, 0);
    current.multifab().setVal(1.1, MacroField::Ux, 1, 0);
    current.multifab().setVal(1.8, MacroField::Uy, 1, 0);
    current.multifab().setVal(0.55, MacroField::Uz, 1, 0);

    diagnostics::SteadyVelocityResidual const residual =
        diagnostics::compute_relative_l2_delta_u(
            current, previous, boxArray, distribution);
    const amrex::Real expected =
        std::sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.05 * 0.05) /
        std::sqrt(1.1 * 1.1 + 1.8 * 1.8 + 0.55 * 0.55);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::abs(residual.relativeL2DeltaU - expected) < 1.0e-14,
        "case-agnostic relative velocity residual mismatch");

    diagnostics::copy_macro_velocity(current, copied);
    diagnostics::SteadyVelocityResidual const copyResidual =
        diagnostics::compute_relative_l2_delta_u(
            current, copied, boxArray, distribution);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        copyResidual.relativeL2DeltaU == 0.0,
        "case-agnostic velocity copy must produce zero residual");

    amrex::Print() << Descriptor::name()
                   << " diagnostics primitive integration tests passed.\n";
}

} // namespace lbm::tests::integration

#endif
