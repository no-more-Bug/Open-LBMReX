#ifndef LBM_DIAGNOSTICS_STEADYVELOCITYRESIDUAL_H
#define LBM_DIAGNOSTICS_STEADYVELOCITYRESIDUAL_H

#include "lbm/fields/MacroField.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace lbm::diagnostics {

/**
 * @brief Case-agnostic relative change in the macroscopic velocity field.
 *
 * The value is computed collectively through MultiFab reductions; every rank
 * must enter the calculation in the same order. It contains no stopping
 * threshold, benchmark identity, or output policy.
 */
struct SteadyVelocityResidual {
    amrex::Real relativeL2DeltaU = -1.0;
};

inline void copy_macro_velocity(MacroField const& src, MacroField& dst)
{
    amrex::MultiFab::Copy(
        dst.multifab(), src.multifab(), MacroField::Ux, MacroField::Ux, 3, 0);
}

inline SteadyVelocityResidual compute_relative_l2_delta_u(
    MacroField const& current,
    MacroField const& previous,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm)
{
    amrex::MultiFab scratch(ba, dm, 2, 0);
    amrex::MultiFab const& currentMf = current.multifab();
    amrex::MultiFab const& previousMf = previous.multifab();

    for (amrex::MFIter mfi(currentMf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const c = currentMf.const_array(mfi);
        auto const p = previousMf.const_array(mfi);
        auto const s = scratch.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real dux = c(i, j, k, MacroField::Ux) - p(i, j, k, MacroField::Ux);
            const amrex::Real duy = c(i, j, k, MacroField::Uy) - p(i, j, k, MacroField::Uy);
            const amrex::Real duz = c(i, j, k, MacroField::Uz) - p(i, j, k, MacroField::Uz);
            const amrex::Real ux = c(i, j, k, MacroField::Ux);
            const amrex::Real uy = c(i, j, k, MacroField::Uy);
            const amrex::Real uz = c(i, j, k, MacroField::Uz);
            s(i, j, k, 0) = dux * dux + duy * duy + duz * duz;
            s(i, j, k, 1) = ux * ux + uy * uy + uz * uz;
        });
    }
    amrex::Gpu::streamSynchronize();

    const amrex::Real deltaSqr = scratch.sum(0);
    const amrex::Real normSqr = scratch.sum(1);
    SteadyVelocityResidual residual;
    residual.relativeL2DeltaU =
        normSqr > 0.0 ? std::sqrt(deltaSqr) / std::sqrt(normSqr) : std::sqrt(deltaSqr);
    return residual;
}

} // namespace lbm::diagnostics

#endif
