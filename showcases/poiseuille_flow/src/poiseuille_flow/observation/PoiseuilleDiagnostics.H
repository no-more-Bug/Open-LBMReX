#ifndef LBM_CASES_POISEUILLEFLOW_DIAGNOSTICS_H
#define LBM_CASES_POISEUILLEFLOW_DIAGNOSTICS_H

#include "lbm/fields/MacroField.H"

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace lbm::cases::poiseuille_flow {

// Describes how profile samples map to the physical channel.  HBB uses fluid
// cell centers strictly inside domain faces, whereas valid-node Guo places the
// two physical walls on the outermost valid centers.
struct PoiseuilleDiagnosticGeometry {
    int streamwiseIndexLo = 0;
    int streamwiseIndexHi = 0;
    int streamwiseValidNodes = 0;
    int streamwiseLatticeIntervals = 0;
    int streamwiseOnLatticeBoundaryNodes = 0;
    int profileAxis = 1;
    int profileIndexLo = 0;
    int profileIndexHi = 0;
    int profileValidNodes = 0;
    int profileLatticeIntervals = 0;
    int onLatticeBoundaryNodes = 0;
    amrex::Real streamwisePhysicalLo = 0.0;
    amrex::Real streamwisePhysicalHi = 1.0;
    amrex::Real profilePhysicalLo = 0.0;
    amrex::Real profilePhysicalHi = 1.0;
    amrex::GpuArray<int, AMREX_SPACEDIM> trapezoidalAxes{};
};

// Host/device-safe metadata for density-target Poiseuille diagnostics.  The
// target density difference is computed from physical lattice intervals, not
// from the N+1-node AMReX envelope.  `enabled` controls the shared density
// diagnostics; the drive-kind flags keep pressure and prescribed-state output
// semantics distinct.
struct PoiseuillePressureDiagnosticConfig {
    int enabled = 0;
    int pressureDriven = 0;
    int prescribedState = 0;
    int scalingCode = 0;
    amrex::Real rhoInLb = 0.0;
    amrex::Real rhoOutLb = 0.0;
    amrex::Real densityDeltaTargetLb = 0.0;
    amrex::Real densityDeltaExpectedLb = 0.0;
    amrex::Real densityDeltaOverRhoRef = 0.0;
    amrex::Real mach = 0.0;
    amrex::Real reEffective = 0.0;
    amrex::Real streamwiseLengthLb = 0.0;
    amrex::Real channelHeightLb = 0.0;
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real poiseuille_streamwise_xi(
    PoiseuilleDiagnosticGeometry const& geometry,
    int streamwiseIndex) noexcept
{
    const amrex::Real offset =
        geometry.streamwiseOnLatticeBoundaryNodes != 0 ? 0.0 : 0.5;
    const int denominator = geometry.streamwiseOnLatticeBoundaryNodes != 0 ?
        geometry.streamwiseLatticeIntervals : geometry.streamwiseValidNodes;
    return denominator > 0 ?
        (static_cast<amrex::Real>(
             streamwiseIndex - geometry.streamwiseIndexLo) + offset) /
            static_cast<amrex::Real>(denominator) :
        0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real poiseuille_eta(
    PoiseuilleDiagnosticGeometry const& geometry,
    int profileIndex) noexcept
{
    const amrex::Real offset =
        geometry.onLatticeBoundaryNodes != 0 ? 0.0 : 0.5;
    const int denominator = geometry.onLatticeBoundaryNodes != 0 ?
        geometry.profileLatticeIntervals : geometry.profileValidNodes;
    return denominator > 0 ?
        (static_cast<amrex::Real>(profileIndex - geometry.profileIndexLo) + offset) /
            static_cast<amrex::Real>(denominator) :
        0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real poiseuille_ux_exact(
    PoiseuilleDiagnosticGeometry const& geometry,
    int profileIndex,
    amrex::Real uMaxLb) noexcept
{
    const amrex::Real eta = poiseuille_eta(geometry, profileIndex);
    return 4.0 * uMaxLb * eta * (1.0 - eta);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real poiseuille_density_delta(
    amrex::Real lengthLb,
    amrex::Real heightLb,
    amrex::Real rho0,
    amrex::Real uMaxLb,
    amrex::Real nuLb,
    amrex::Real cs2) noexcept
{
    return 8.0 * rho0 * nuLb * uMaxLb * lengthLb /
           (cs2 * heightLb * heightLb);
}

struct PoiseuilleProfileOutput {
    int step = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nIntervalsY = 0;
    int onLatticeBoundaryNodes = 0;
    amrex::Real uMaxLb = 0.0;
    amrex::Real e2Profile = 0.0;
    amrex::Real eInfProfile = 0.0;
    amrex::Real e2ProfileFull = 0.0;
    amrex::Real eInfProfileFull = 0.0;
    amrex::Real maxAbsUyOverUmax = 0.0;
    amrex::Real maxAbsUzOverUmax = 0.0;
    amrex::Real relativeMassError = 0.0;
    amrex::Real rawLatticeMass = 0.0;
    amrex::Real rawLatticeReferenceMass = 0.0;
    amrex::Real physicalWeightedMass = 0.0;
    amrex::Real physicalWeightedReferenceMass = 0.0;
    amrex::Real physicalWeightedRelativeMassError = 0.0;
    amrex::Real rhoMeanMinusRho0 = 0.0;
    amrex::Real rhoStdOverRho0 = 0.0;
    amrex::Real rhoMin = 0.0;
    amrex::Real rhoMax = 0.0;
    amrex::Real rhoStreamwiseXLo = 0.0;
    amrex::Real rhoStreamwiseXHi = 0.0;
    amrex::Real pressureStreamwiseXLo = 0.0;
    amrex::Real pressureStreamwiseXHi = 0.0;
    amrex::Real rhoStreamwiseSlope = 0.0;
    amrex::Real pressureStreamwiseSlope = 0.0;
    amrex::Real rhoStreamwiseLinearR2 = 0.0;
    int pressureDriven = 0;
    int prescribedState = 0;
    int pressureScalingCode = 0;
    amrex::Real rhoInTargetLb = 0.0;
    amrex::Real rhoOutTargetLb = 0.0;
    amrex::Real densityDeltaTargetLb = 0.0;
    amrex::Real densityDeltaExpectedLb = 0.0;
    amrex::Real densityDeltaOverRhoRef = 0.0;
    amrex::Real mach = 0.0;
    amrex::Real reEffective = 0.0;
    amrex::Real maxAbsPressureBoundaryRhoError = 0.0;
    amrex::Real rhoStreamwiseDeltaMeasured = 0.0;
    amrex::Real rhoStreamwiseDeltaRelativeError = 0.0;
    amrex::Real maxAbsProfileErrorAllXOverUmax = 0.0;
    amrex::Real maxStreamwiseUxSpreadOverUmax = 0.0;
};

/**
 * @brief Collective Poiseuille metrics and rows ready for serialization.
 *
 * All MultiFab reductions are complete when this value is returned, and every
 * rank owns the same host vectors. The application output layer is solely
 * responsible for paths and IO-rank file writes; it must not alter the values
 * or repeat the reductions.
 */
struct PoiseuilleProfileData {
    PoiseuilleProfileOutput output;
    int domainLoX = 0;
    int domainLoY = 0;
    int centerI = 0;
    std::vector<amrex::Real> uxMean;
    std::vector<amrex::Real> uyAbsMax;
    std::vector<amrex::Real> uzAbsMax;
    std::vector<amrex::Real> profilePhysicalCoordinate;
    std::vector<amrex::Real> profileEta;
    std::vector<amrex::Real> profileExactVelocity;
    std::vector<int> profileBoundaryNode;
    std::vector<amrex::Real> streamwisePhysicalCoordinate;
    std::vector<amrex::Real> streamwiseXi;
    std::vector<amrex::Real> rhoCrossSectionMean;
    std::vector<amrex::Real> pressureCrossSectionMean;
};

inline PoiseuilleProfileData compute_poiseuille_profile_data(
    MacroField const& macro,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int step,
    amrex::Real rho0,
    amrex::Real uMaxLb,
    amrex::Real nuLb,
    amrex::Real cs2Lb,
    PoiseuilleDiagnosticGeometry const& profileGeometry,
    PoiseuillePressureDiagnosticConfig const& pressureConfig)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        profileGeometry.profileAxis == 1,
        "Poiseuille diagnostics currently require profile_axis=y");

    amrex::Box const domain = geom.Domain();
    const int nx = domain.length(0);
    const int ny = domain.length(1);
    const int nz = domain.length(2);
    const int centerI = domain.smallEnd(0) + nx / 2;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        profileGeometry.streamwiseValidNodes == nx,
        "Poiseuille streamwise diagnostics require geometry valid_nodes == nx");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nuLb > 0.0 && cs2Lb > 0.0,
        "Poiseuille diagnostics require positive nu_lb and cs2_lb");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        pressureConfig.enabled ==
            pressureConfig.pressureDriven + pressureConfig.prescribedState &&
            pressureConfig.pressureDriven >= 0 &&
            pressureConfig.pressureDriven <= 1 &&
            pressureConfig.prescribedState >= 0 &&
            pressureConfig.prescribedState <= 1,
        "Poiseuille density diagnostics require one consistent drive kind");
    // 0..14 retain the established scalar metrics; 15 is the strict-face
    // prescribed-density error and 16 is the all-x profile error.  Per-y
    // rows then hold the center profile plus all-x max/min velocity data.
    const int rowBase = 17;
    amrex::MultiFab scratch(ba, dm, rowBase + 5 * ny, 0);
    scratch.setVal(0.0);

    const int domainLoX = domain.smallEnd(0);
    const int domainHiX = domain.bigEnd(0);
    const int domainLoY = domain.smallEnd(1);
    const int domainHiY = domain.bigEnd(1);
    const int domainLoZ = domain.smallEnd(2);
    const int domainHiZ = domain.bigEnd(2);
    const int onLatticeBoundaryNodes = profileGeometry.onLatticeBoundaryNodes;
    const auto trapezoidalAxes = profileGeometry.trapezoidalAxes;
    const amrex::Real uxMinShift = 1.0 + std::abs(uMaxLb);

    amrex::MultiFab const& macroMf = macro.multifab();
    for (amrex::MFIter mfi(macroMf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const m = macroMf.const_array(mfi);
        auto const s = scratch.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const int jy = j - domainLoY;
            const amrex::Real rho = m(i, j, k, MacroField::Rho);
            const amrex::Real ux = m(i, j, k, MacroField::Ux);
            const amrex::Real uy = m(i, j, k, MacroField::Uy);
            const amrex::Real uz = m(i, j, k, MacroField::Uz);
            const amrex::Real exact =
                poiseuille_ux_exact(profileGeometry, j, uMaxLb);
            const amrex::Real err = ux - exact;
            const amrex::Real absErr = err < 0.0 ? -err : err;
            const amrex::Real absUy = uy < 0.0 ? -uy : uy;
            const amrex::Real absUz = uz < 0.0 ? -uz : uz;
            const amrex::Real rhoErr = rho - rho0;
            const bool strictInteriorProfileNode =
                onLatticeBoundaryNodes == 0 ||
                (j != domainLoY && j != domainHiY);

            amrex::Real physicalWeight = 1.0;
            if (trapezoidalAxes[0] != 0 &&
                (i == domainLoX || i == domainHiX)) {
                physicalWeight *= 0.5;
            }
            if (trapezoidalAxes[1] != 0 &&
                (j == domainLoY || j == domainHiY)) {
                physicalWeight *= 0.5;
            }
            if (trapezoidalAxes[2] != 0 &&
                (k == domainLoZ || k == domainHiZ)) {
                physicalWeight *= 0.5;
            }

            s(i, j, k, 0) = 0.0;
            s(i, j, k, 1) = 0.0;
            s(i, j, k, 2) = 0.0;
            s(i, j, k, 3) = absUy;
            s(i, j, k, 4) = absUz;
            s(i, j, k, 5) = rho;
            s(i, j, k, 6) = rhoErr;
            s(i, j, k, 7) = 0.0;
            s(i, j, k, 8) = 0.0;
            s(i, j, k, 9) = 0.0;
            s(i, j, k, 10) = physicalWeight * rho;
            s(i, j, k, 11) = physicalWeight;
            s(i, j, k, 12) = rhoErr * rhoErr;
            s(i, j, k, 13) = rho;
            s(i, j, k, 14) = -rho;
            s(i, j, k, 15) = 0.0;
            s(i, j, k, 16) = strictInteriorProfileNode ? absErr : 0.0;

            if (pressureConfig.enabled != 0 && strictInteriorProfileNode &&
                (i == domainLoX || i == domainHiX)) {
                const amrex::Real rhoTarget =
                    i == domainLoX ? pressureConfig.rhoInLb :
                                     pressureConfig.rhoOutLb;
                const amrex::Real rhoBoundaryError = rho - rhoTarget;
                s(i, j, k, 15) = rhoBoundaryError < 0.0 ?
                    -rhoBoundaryError : rhoBoundaryError;
            }

            s(i, j, k, rowBase + 3 * ny + jy) = ux;
            // Other y-row components remain zero.  A positive shift keeps the
            // selected row above those zeros while still allowing min(ux) to
            // be recovered through a max reduction.
            s(i, j, k, rowBase + 4 * ny + jy) = uxMinShift - ux;

            if (i == centerI) {
                if (strictInteriorProfileNode) {
                    s(i, j, k, 0) = err * err;
                    s(i, j, k, 1) = exact * exact;
                    s(i, j, k, 2) = absErr;
                }
                s(i, j, k, 7) = err * err;
                s(i, j, k, 8) = exact * exact;
                s(i, j, k, 9) = absErr;
                s(i, j, k, rowBase + jy) = ux;
                s(i, j, k, rowBase + ny + jy) = absUy;
                s(i, j, k, rowBase + 2 * ny + jy) = absUz;
            }
        });
    }
    amrex::Gpu::streamSynchronize();

    const amrex::Real nPts = static_cast<amrex::Real>(domain.numPts());
    const amrex::Real invUmax = uMaxLb > 0.0 ? 1.0 / uMaxLb : 0.0;
    const amrex::Real invRho0 = rho0 > 0.0 ? 1.0 / rho0 : 0.0;
    const amrex::Real errSqrSum = scratch.sum(0);
    const amrex::Real exactSqrSum = scratch.sum(1);
    const amrex::Real fullErrSqrSum = scratch.sum(7);
    const amrex::Real fullExactSqrSum = scratch.sum(8);
    const amrex::Real rhoSum = scratch.sum(5);
    const amrex::Real rhoErrSqrSum = scratch.sum(12);
    const amrex::Real physicalWeightedMass = scratch.sum(10);
    const amrex::Real physicalWeightSum = scratch.sum(11);

    PoiseuilleProfileData data;
    PoiseuilleProfileOutput& output = data.output;
    output.step = step;
    output.nx = nx;
    output.ny = ny;
    output.nz = nz;
    output.nIntervalsY = profileGeometry.profileLatticeIntervals;
    output.onLatticeBoundaryNodes = onLatticeBoundaryNodes;
    output.uMaxLb = uMaxLb;
    output.e2Profile = exactSqrSum > 0.0 ?
        std::sqrt(errSqrSum / exactSqrSum) : std::sqrt(errSqrSum / nPts);
    output.eInfProfile = scratch.max(2) * invUmax;
    output.e2ProfileFull = fullExactSqrSum > 0.0 ?
        std::sqrt(fullErrSqrSum / fullExactSqrSum) :
        std::sqrt(fullErrSqrSum / nPts);
    output.eInfProfileFull = scratch.max(9) * invUmax;
    output.maxAbsUyOverUmax = scratch.max(3) * invUmax;
    output.maxAbsUzOverUmax = scratch.max(4) * invUmax;
    output.rawLatticeMass = rhoSum;
    output.rawLatticeReferenceMass = rho0 * nPts;
    output.relativeMassError =
        std::abs(output.rawLatticeMass - output.rawLatticeReferenceMass) /
        output.rawLatticeReferenceMass;
    output.physicalWeightedMass = physicalWeightedMass;
    output.physicalWeightedReferenceMass = rho0 * physicalWeightSum;
    output.physicalWeightedRelativeMassError = std::abs(
        output.physicalWeightedMass - output.physicalWeightedReferenceMass) /
        output.physicalWeightedReferenceMass;
    output.rhoMeanMinusRho0 = rhoSum / nPts - rho0;
    const amrex::Real rhoVariance =
        rhoErrSqrSum / nPts -
        output.rhoMeanMinusRho0 * output.rhoMeanMinusRho0;
    output.rhoStdOverRho0 =
        std::sqrt(rhoVariance > 0.0 ? rhoVariance : 0.0) * invRho0;
    output.rhoMax = scratch.max(13);
    output.rhoMin = -scratch.max(14);
    output.pressureDriven = pressureConfig.pressureDriven;
    output.prescribedState = pressureConfig.prescribedState;
    output.pressureScalingCode = pressureConfig.scalingCode;
    output.rhoInTargetLb = pressureConfig.rhoInLb;
    output.rhoOutTargetLb = pressureConfig.rhoOutLb;
    output.densityDeltaTargetLb = pressureConfig.densityDeltaTargetLb;
    output.densityDeltaExpectedLb = pressureConfig.densityDeltaExpectedLb;
    output.densityDeltaOverRhoRef = pressureConfig.densityDeltaOverRhoRef;
    output.mach = pressureConfig.mach;
    output.reEffective =
        uMaxLb * static_cast<amrex::Real>(profileGeometry.profileLatticeIntervals) /
        nuLb;
    output.maxAbsPressureBoundaryRhoError = scratch.max(15);
    output.maxAbsProfileErrorAllXOverUmax = scratch.max(16) * invUmax;
    const amrex::Real invRowCount =
        1.0 / static_cast<amrex::Real>(nz);
    std::vector<amrex::Real> uxMean(ny);
    std::vector<amrex::Real> uyAbsMax(ny);
    std::vector<amrex::Real> uzAbsMax(ny);
    std::vector<amrex::Real> profilePhysicalCoordinate(ny);
    std::vector<amrex::Real> profileEta(ny);
    std::vector<amrex::Real> profileExactVelocity(ny);
    std::vector<int> profileBoundaryNode(ny);
    amrex::Real maxStreamwiseUxSpread = 0.0;
    // MultiFab reductions are collective, so every rank must execute them.
    for (int jy = 0; jy < ny; ++jy) {
        uxMean[jy] = scratch.sum(rowBase + jy) * invRowCount;
        uyAbsMax[jy] = scratch.max(rowBase + ny + jy);
        uzAbsMax[jy] = scratch.max(rowBase + 2 * ny + jy);
        const int j = domainLoY + jy;
        profileEta[jy] = poiseuille_eta(profileGeometry, j);
        profilePhysicalCoordinate[jy] =
            profileGeometry.profilePhysicalLo + profileEta[jy] *
                (profileGeometry.profilePhysicalHi -
                 profileGeometry.profilePhysicalLo);
        profileExactVelocity[jy] =
            poiseuille_ux_exact(profileGeometry, j, uMaxLb);
        profileBoundaryNode[jy] =
            onLatticeBoundaryNodes != 0 &&
            (j == domainLoY || j == domainHiY) ? 1 : 0;
        const amrex::Real uxMax = scratch.max(rowBase + 3 * ny + jy);
        const amrex::Real uxMin =
            uxMinShift - scratch.max(rowBase + 4 * ny + jy);
        const amrex::Real spread = uxMax - uxMin;
        maxStreamwiseUxSpread =
            spread > maxStreamwiseUxSpread ? spread : maxStreamwiseUxSpread;
    }
    output.maxStreamwiseUxSpreadOverUmax =
        maxStreamwiseUxSpread * invUmax;

    // Build a streamwise density/pressure distribution from a physical-domain
    // trapezoidal cross-section average.  Components 10/11 both contain the
    // same x-endpoint factor, so it cancels in their ratio while y endpoint
    // weighting remains active.  MultiFab::sum(Box, ...) is a global
    // collective; every rank executes these slices in the same x order.
    std::vector<amrex::Real> rhoCrossSectionMean(nx);
    std::vector<amrex::Real> pressureCrossSectionMean(nx);
    std::vector<amrex::Real> streamwisePhysicalCoordinate(nx);
    std::vector<amrex::Real> streamwiseXi(nx);
    for (int ix = 0; ix < nx; ++ix) {
        const int i = domainLoX + ix;
        amrex::Box slice = domain;
        slice.setSmall(0, i);
        slice.setBig(0, i);
        const amrex::Real weightedRho = scratch.sum(slice, 10);
        const amrex::Real weight = scratch.sum(slice, 11);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            weight > 0.0,
            "Poiseuille density/pressure cross-section weight must be positive");
        rhoCrossSectionMean[ix] = weightedRho / weight;
        pressureCrossSectionMean[ix] = cs2Lb * rhoCrossSectionMean[ix];
        streamwiseXi[ix] =
            poiseuille_streamwise_xi(profileGeometry, i);
        streamwisePhysicalCoordinate[ix] =
            profileGeometry.streamwisePhysicalLo + streamwiseXi[ix] *
                (profileGeometry.streamwisePhysicalHi -
                 profileGeometry.streamwisePhysicalLo);
    }

    amrex::Real xiMean = 0.0;
    amrex::Real rhoStreamwiseMean = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
        const int i = domainLoX + ix;
        xiMean += poiseuille_streamwise_xi(profileGeometry, i);
        rhoStreamwiseMean += rhoCrossSectionMean[ix];
    }
    xiMean /= static_cast<amrex::Real>(nx);
    rhoStreamwiseMean /= static_cast<amrex::Real>(nx);

    amrex::Real xiVarianceSum = 0.0;
    amrex::Real rhoVarianceSum = 0.0;
    amrex::Real xiRhoCovarianceSum = 0.0;
    for (int ix = 0; ix < nx; ++ix) {
        const int i = domainLoX + ix;
        const amrex::Real xi =
            poiseuille_streamwise_xi(profileGeometry, i);
        const amrex::Real xiDelta = xi - xiMean;
        const amrex::Real rhoDelta =
            rhoCrossSectionMean[ix] - rhoStreamwiseMean;
        xiVarianceSum += xiDelta * xiDelta;
        rhoVarianceSum += rhoDelta * rhoDelta;
        xiRhoCovarianceSum += xiDelta * rhoDelta;
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        xiVarianceSum > 0.0,
        "Poiseuille density/pressure trend requires at least two x positions");
    output.rhoStreamwiseXLo = rhoCrossSectionMean.front();
    output.rhoStreamwiseXHi = rhoCrossSectionMean.back();
    output.pressureStreamwiseXLo = pressureCrossSectionMean.front();
    output.pressureStreamwiseXHi = pressureCrossSectionMean.back();
    output.rhoStreamwiseSlope = xiRhoCovarianceSum / xiVarianceSum;
    output.pressureStreamwiseSlope = cs2Lb * output.rhoStreamwiseSlope;
    output.rhoStreamwiseLinearR2 = rhoVarianceSum > 0.0 ?
        (xiRhoCovarianceSum * xiRhoCovarianceSum) /
            (xiVarianceSum * rhoVarianceSum) :
        1.0;
    output.rhoStreamwiseDeltaMeasured =
        output.rhoStreamwiseXLo - output.rhoStreamwiseXHi;
    output.rhoStreamwiseDeltaRelativeError =
        pressureConfig.enabled != 0 && pressureConfig.densityDeltaTargetLb > 0.0 ?
            std::abs(
                output.rhoStreamwiseDeltaMeasured -
                pressureConfig.densityDeltaTargetLb) /
                pressureConfig.densityDeltaTargetLb :
            0.0;
    data.domainLoX = domainLoX;
    data.domainLoY = domainLoY;
    data.centerI = centerI;
    data.uxMean = std::move(uxMean);
    data.uyAbsMax = std::move(uyAbsMax);
    data.uzAbsMax = std::move(uzAbsMax);
    data.profilePhysicalCoordinate =
        std::move(profilePhysicalCoordinate);
    data.profileEta = std::move(profileEta);
    data.profileExactVelocity = std::move(profileExactVelocity);
    data.profileBoundaryNode = std::move(profileBoundaryNode);
    data.streamwisePhysicalCoordinate =
        std::move(streamwisePhysicalCoordinate);
    data.streamwiseXi = std::move(streamwiseXi);
    data.rhoCrossSectionMean = std::move(rhoCrossSectionMean);
    data.pressureCrossSectionMean = std::move(pressureCrossSectionMean);
    return data;
}

inline std::string poiseuille_profile_summary(PoiseuilleProfileOutput const& output)
{
    std::ostringstream os;
    os << std::setprecision(12)
       << "poiseuille_profile step=" << output.step
       << " nx=" << output.nx
       << " ny=" << output.ny
       << " nz=" << output.nz
       << " n_intervals_y=" << output.nIntervalsY
       << " on_lattice_boundary_nodes=" << output.onLatticeBoundaryNodes
       << " u_max_lb=" << output.uMaxLb
       << " E_2_profile(interior)=" << output.e2Profile
       << " E_inf_profile(interior)=" << output.eInfProfile
       << " E_2_profile_full=" << output.e2ProfileFull
       << " E_inf_profile_full=" << output.eInfProfileFull
       << " max_abs_uy/Umax=" << output.maxAbsUyOverUmax
       << " max_abs_uz/Umax=" << output.maxAbsUzOverUmax
       << " relative_mass_error(raw_lattice)=" << output.relativeMassError
       << " raw_lattice_mass=" << output.rawLatticeMass
       << " raw_lattice_reference_mass=" << output.rawLatticeReferenceMass
       << " physical_weighted_mass=" << output.physicalWeightedMass
       << " physical_weighted_reference_mass="
       << output.physicalWeightedReferenceMass
       << " physical_weighted_relative_mass_error="
       << output.physicalWeightedRelativeMassError
       << " rho_mean-rho0=" << output.rhoMeanMinusRho0
       << " rho_std/rho0=" << output.rhoStdOverRho0
       << " rho_min=" << output.rhoMin
       << " rho_max=" << output.rhoMax
       << " rho_streamwise_xlo=" << output.rhoStreamwiseXLo
       << " rho_streamwise_xhi=" << output.rhoStreamwiseXHi
       << " pressure_streamwise_xlo=" << output.pressureStreamwiseXLo
       << " pressure_streamwise_xhi=" << output.pressureStreamwiseXHi
       << " rho_streamwise_slope=" << output.rhoStreamwiseSlope
       << " pressure_streamwise_slope=" << output.pressureStreamwiseSlope
       << " rho_streamwise_linear_r2=" << output.rhoStreamwiseLinearR2
       << " pressure_driven=" << output.pressureDriven
       << " prescribed_state=" << output.prescribedState
       << " pressure_scaling_code=" << output.pressureScalingCode
       << " Re_effective=" << output.reEffective
       << " rho_in_target_lb=" << output.rhoInTargetLb
       << " rho_out_target_lb=" << output.rhoOutTargetLb
       << " density_delta_target_lb=" << output.densityDeltaTargetLb
       << " density_delta_expected_lb=" << output.densityDeltaExpectedLb
       << " density_delta/rho_ref=" << output.densityDeltaOverRhoRef
       << " Mach=" << output.mach
       << " max_abs_pressure_boundary_rho_error="
       << output.maxAbsPressureBoundaryRhoError
       << " rho_streamwise_delta_measured="
       << output.rhoStreamwiseDeltaMeasured
       << " rho_streamwise_delta_relative_error="
       << output.rhoStreamwiseDeltaRelativeError
       << " max_abs_profile_error_all_x/Umax="
       << output.maxAbsProfileErrorAllXOverUmax
       << " max_streamwise_ux_spread/Umax="
       << output.maxStreamwiseUxSpreadOverUmax;
    return os.str();
}

} // namespace lbm::cases::poiseuille_flow

#endif
