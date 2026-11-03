#ifndef LBM_RUNTIME_SINGLELEVELDIAGNOSTICS_H
#define LBM_RUNTIME_SINGLELEVELDIAGNOSTICS_H

#include "lbm/config/LbmConfig.H"
#include "lbm/diagnostics/SteadyVelocityResidual.H"
#include "lbm/fields/MacroField.H"
#include "runtime/SingleLevelDiagnosticRow.H"
#include "runtime/SingleLevelDiagnosticsConfig.H"

#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

namespace lbm::runtime {

namespace detail {

template <class ExactReference>
void fill_diagnostic_scratch(
    amrex::Box const& bx,
    amrex::Array4<amrex::Real const> m,
    amrex::Array4<amrex::Real> d,
    amrex::GeometryData geometry,
    ExactReference deviceExactReference)
{
    amrex::ParallelFor(
        bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const int index[AMREX_SPACEDIM] = {
                AMREX_D_DECL(i, j, k)};
            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
                cellCenter{};
            for (int axis = 0; axis < AMREX_SPACEDIM; ++axis) {
                cellCenter[axis] =
                    geometry.ProbLo(axis) +
                    (static_cast<amrex::Real>(index[axis]) + 0.5) *
                        geometry.CellSize(axis);
            }
            const auto exact =
                deviceExactReference(cellCenter, i, j, k);

            const amrex::Real uxErr =
                m(i, j, k, MacroField::Ux) - exact.ux;
            const amrex::Real uyErr =
                m(i, j, k, MacroField::Uy) - exact.uy;
            const amrex::Real uzErr =
                m(i, j, k, MacroField::Uz) - exact.uz;
            const amrex::Real rhoErr =
                m(i, j, k, MacroField::Rho) - exact.rho;
            const amrex::Real uSqr =
                m(i, j, k, MacroField::Ux) *
                    m(i, j, k, MacroField::Ux) +
                m(i, j, k, MacroField::Uy) *
                    m(i, j, k, MacroField::Uy) +
                m(i, j, k, MacroField::Uz) *
                    m(i, j, k, MacroField::Uz);
            const amrex::Real uErrSqr =
                uxErr * uxErr + uyErr * uyErr + uzErr * uzErr;
            const amrex::Real uExactSqr =
                exact.ux * exact.ux +
                exact.uy * exact.uy +
                exact.uz * exact.uz;

            d(i, j, k, 0) = m(i, j, k, MacroField::Rho);
            d(i, j, k, 1) = uxErr * uxErr;
            d(i, j, k, 2) = uyErr * uyErr;
            d(i, j, k, 3) = uzErr * uzErr;
            d(i, j, k, 4) = uErrSqr;
            d(i, j, k, 5) = uSqr;
            d(i, j, k, 6) =
                rhoErr < 0.0 ? -rhoErr : rhoErr;
            d(i, j, k, 7) = uExactSqr;
        });
}

} // namespace detail

/**
 * @brief Own generic diagnostic-row, plot and steady observation state.
 *
 * The fixed runtime supplies an immutable macro field only at resolved
 * observation moments. This class owns generic diagnostic row/plot/steady
 * state and passes completed records to its output implementation; concrete case
 * reductions, acceptance and artifact schemas remain outside this module.
 *
 * `initialize()` observes f^0, initializes the optional steady reference from
 * that same layer, and performs the step-zero plot request. `onStep()` must be
 * called after swap, so step n always observes the completed f^n layer.
 * `ensureFinalObservation()` supplies any missing final diagnostic/plot
 * observation before case metrics and the final summary. These methods must be entered by every
 * rank in identical order because sampling, case diagnostics, and plotfiles
 * contain MPI collectives.
 *
 * The referenced resolved configuration, Geometry/layout, and
 * output writer must outlive this object and remain immutable during the run.
 * This class neither advances populations nor applies boundary conditions.
 */
template <class Descriptor, class Output>
class SingleLevelDiagnostics {
public:
    SingleLevelDiagnostics(
        SingleLevelDiagnosticsConfig diagnosticsConfig,
        config::ResolvedLbmConfig const& lbmConfig,
        amrex::Geometry const& geometry,
        amrex::BoxArray const& boxArray,
        amrex::DistributionMapping const& distributionMapping,
        SingleLevelSteadyObservationPolicy steadyPolicy,
        Output& output)
        : cfg_(diagnosticsConfig),
          lbm_(lbmConfig),
          geom_(geometry),
          ba_(boxArray),
          dm_(distributionMapping),
          output_(output),
          steadyPolicy_(steadyPolicy)
    {}

    SingleLevelDiagnostics(SingleLevelDiagnostics const&) = delete;
    SingleLevelDiagnostics& operator=(SingleLevelDiagnostics const&) = delete;

    /**
     * @brief Observe initialized f^0 and establish observer state.
     *
     * The supplied macro was reconstructed from the fully initialized f^0
     * after any on-lattice valid-boundary overwrite.
     */
    template <class ExactReference>
    void initialize(
        MacroField const& macro,
        ExactReference const& exactReference)
    {
        last_ = compute_and_write_diagnostics(
            macro, exactReference, 0, true);
        hasLast_ = true;
        initialize_steady_monitor(macro);
        if (cfg_.plotInterval > 0) {
            write_plotfile(macro, 0);
        }
    }

    /**
     * @brief Run interval observers for a completed post-swap population layer.
     *
     * The call preserves the historical order: diagnostics row, plotfile, then
     * steady monitor. It returns true only when the steady criterion is reached
     * and stop-on-steady is enabled; the caller remains responsible for ending
     * the timestep loop after profiler completion.
     */
    template <class ExactReference>
    bool onStep(
        MacroField const& macro,
        ExactReference const& exactReference,
        int step)
    {
        if (cfg_.interval > 0 &&
            (step % cfg_.interval == 0 || step == cfg_.maxStep)) {
            last_ = compute_and_write_diagnostics(
                macro, exactReference, step, false);
            hasLast_ = true;
        }
        if (cfg_.plotInterval > 0 &&
            (step % cfg_.plotInterval == 0 || step == cfg_.maxStep)) {
            write_plotfile(macro, step);
        }
        return advance_steady_monitor(macro, step) &&
               steadyPolicy_.stopOnSteady != 0;
    }

    /**
     * @brief Complete final observations and case-level acceptance output.
     *
     * `finalStep` names the current population layer. If early steady stopping
     * occurred between diagnostic/plot intervals, this method samples that
     * exact layer before writing case metrics, convergence data, and the final
     * summary. The order and collective participation match the pre-extraction
     * solver lifecycle.
     */
    template <class ExactReference>
    void ensureFinalObservation(
        MacroField const& macro,
        ExactReference const& exactReference,
        int finalStep)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            hasLast_,
            "SingleLevelDiagnostics::initialize must precede finalize");
        if (last_.step != finalStep) {
            last_ = compute_and_write_diagnostics(
                macro, exactReference, finalStep, false);
        }
        if (cfg_.writeFinalPlotfile != 0 &&
            (cfg_.plotInterval <= 0 || finalStep % cfg_.plotInterval != 0)) {
            write_plotfile(macro, finalStep);
        }
    }

    void writeFinalSummary()
    {
        bool convergenceRowWritten = false;
        if (cfg_.convergenceEnabled != 0) {
            output_.writeConvergenceRow(last_);
            convergenceRowWritten = true;
        }
        output_.writeFinalSummary(
            last_,
            convergenceRowWritten,
            finalSummaryView());
    }

    SingleLevelFinalSummaryView finalSummaryView() const noexcept
    {
        return {
            steadyPolicy_.label,
            steadyPolicy_.enabled,
            steadyReached_,
            steadyReachedStep_,
            lastSteadyResidual_};
    }

    bool steadyReached() const noexcept { return steadyReached_; }
    amrex::Real lastSteadyResidual() const noexcept
    {
        return lastSteadyResidual_;
    }

private:
    amrex::Real reference_dx_phys() const noexcept
    {
        return lbm_.lengthRefPhys / lbm_.lengthRefLb;
    }

    static amrex::Real lb_time_from_step(int step) noexcept
    {
        return static_cast<amrex::Real>(step);
    }

    amrex::Real physical_time_from_lb_time(
        amrex::Real timeLb) const noexcept
    {
        return timeLb * lbm_.uRefLb * reference_dx_phys() / lbm_.uRefPhys;
    }

    amrex::Real physical_time_from_step(int step) const noexcept
    {
        return physical_time_from_lb_time(lb_time_from_step(step));
    }

    void initialize_steady_monitor(MacroField const& macro)
    {
        if (!steadyPolicy_.enabled) {
            return;
        }
        steadyPreviousMacro_ =
            std::make_unique<MacroField>("steady_previous_macro", ba_, dm_);
        lbm::diagnostics::copy_macro_velocity(
            macro, *steadyPreviousMacro_);
        steadyReferenceInitialized_ = true;
        lastSteadyResidual_ = -1.0;
        steadyReached_ = false;
        steadyReachedStep_ = -1;
    }

    bool advance_steady_monitor(
        MacroField const& macro,
        int step)
    {
        if (!steadyPolicy_.enabled) {
            return false;
        }
        if (!steadyReferenceInitialized_) {
            if (!steadyPreviousMacro_) {
                steadyPreviousMacro_ = std::make_unique<MacroField>(
                    "steady_previous_macro", ba_, dm_);
            }
            lbm::diagnostics::copy_macro_velocity(
                macro, *steadyPreviousMacro_);
            steadyReferenceInitialized_ = true;
            return false;
        }

        bool reached = false;
        if (step % steadyPolicy_.checkInterval == 0) {
            lbm::diagnostics::SteadyVelocityResidual const residual =
                lbm::diagnostics::compute_relative_l2_delta_u(
                    macro, *steadyPreviousMacro_, ba_, dm_);
            lastSteadyResidual_ = residual.relativeL2DeltaU;

            std::ostringstream os;
            os << std::setprecision(12)
               << steadyPolicy_.label << " step=" << step
               << " relative_l2_delta_u=" << lastSteadyResidual_
               << " steady_l2_tol=" << steadyPolicy_.l2Tolerance
               << " delta_step=1\n";
            output_.printAndLog(os.str());

            if (lastSteadyResidual_ < steadyPolicy_.l2Tolerance) {
                steadyReached_ = true;
                steadyReachedStep_ = step;
                std::ostringstream reachedMessage;
                reachedMessage
                    << steadyPolicy_.label << " reached step=" << step
                    << " relative_l2_delta_u=" << lastSteadyResidual_
                    << " delta_step=1\n";
                output_.printAndLog(reachedMessage.str());
                reached = true;
            }
        }
        lbm::diagnostics::copy_macro_velocity(
            macro, *steadyPreviousMacro_);
        return reached;
    }

    /**
     * @brief Reduce generic errors for one completed population time layer.
     *
     * Exact-reference policies receive AMReX physical cell-center positions
     * computed from device-safe GeometryData. Any benchmark-specific
     * integer-node interpretation remains encapsulated by the selected policy.
     * All ranks must enter this method because the MultiFab reductions are
     * collective.
     */
    template <class ExactReference>
    DiagnosticRow compute_diagnostics(
        MacroField const& macro,
        ExactReference const& exactReference,
        int step)
    {
        // Scratch components:
        // 0 rho, 1 ux error^2, 2 uy error^2, 3 uz error^2,
        // 4 |u-u_exact|^2, 5 |u|^2, 6 |rho-rho_exact|,
        // 7 |u_exact|^2.
        const auto deviceExactReference = exactReference;
        const amrex::GeometryData geometry = geom_.data();

        amrex::MultiFab const& macroMf = macro.multifab();
        amrex::MultiFab scratch(ba_, dm_, 8, 0);
        for (amrex::MFIter mfi(
                 macroMf, amrex::TilingIfNotGPU());
             mfi.isValid();
             ++mfi) {
            amrex::Box const& bx = mfi.tilebox();
            auto const m = macroMf.const_array(mfi);
            auto const d = scratch.array(mfi);
            detail::fill_diagnostic_scratch(
                bx, m, d, geometry, deviceExactReference);
        }
        amrex::Gpu::streamSynchronize();

        const auto nPts =
            static_cast<amrex::Real>(geom_.Domain().numPts());
        DiagnosticRow row;
        row.step = step;
        row.timeLb = lb_time_from_step(step);
        row.timePhys = physical_time_from_lb_time(row.timeLb);
        row.mass = scratch.sum(0);
        const amrex::Real referenceMass =
            exactReference.massReferenceDensity * nPts;
        row.relativeMassError =
            std::abs(row.mass - referenceMass) / referenceMass;
        const amrex::Real uxErrSum = scratch.sum(1);
        const amrex::Real uyErrSum = scratch.sum(2);
        const amrex::Real uzErrSum = scratch.sum(3);
        const amrex::Real uErrSum = scratch.sum(4);
        const amrex::Real uExactSum = scratch.sum(7);
        row.l2Ux = std::sqrt(uxErrSum / nPts);
        row.l2Uy = std::sqrt(uyErrSum / nPts);
        row.l2Uz = std::sqrt(uzErrSum / nPts);
        row.l2U = std::sqrt(uErrSum / nPts);
        row.l2UExact = std::sqrt(uExactSum / nPts);
        row.relativeL2U = uExactSum > 0.0 ?
            std::sqrt(uErrSum) / std::sqrt(uExactSum) : 0.0;
        row.linfU = std::sqrt(scratch.max(4));
        row.linfUz = std::sqrt(scratch.max(3));
        row.maxU = std::sqrt(scratch.max(5));
        row.maxRhoError = scratch.max(6);
        return row;
    }

    template <class ExactReference>
    DiagnosticRow compute_and_write_diagnostics(
        MacroField const& macro,
        ExactReference const& exactReference,
        int step,
        bool resetFile)
    {
        DiagnosticRow row = compute_diagnostics(
            macro, exactReference, step);
        if (cfg_.convergenceEnabled == 0) {
            output_.writeDiagnosticsRow(row, resetFile);
        }
        output_.writeDiagnosticObservation(row);
        return row;
    }

    void write_plotfile(
        MacroField const& macro,
        int step)
    {
        output_.writePlotfile(
            macro,
            geom_,
            ba_,
            dm_,
            step,
            physical_time_from_step(step));
    }

    SingleLevelDiagnosticsConfig cfg_;
    config::ResolvedLbmConfig const& lbm_;
    amrex::Geometry const& geom_;
    amrex::BoxArray const& ba_;
    amrex::DistributionMapping const& dm_;
    Output& output_;
    SingleLevelSteadyObservationPolicy steadyPolicy_;
    std::unique_ptr<MacroField> steadyPreviousMacro_;
    DiagnosticRow last_;
    bool hasLast_ = false;
    bool steadyReferenceInitialized_ = false;
    bool steadyReached_ = false;
    int steadyReachedStep_ = -1;
    amrex::Real lastSteadyResidual_ = -1.0;
};

} // namespace lbm::runtime

#endif
