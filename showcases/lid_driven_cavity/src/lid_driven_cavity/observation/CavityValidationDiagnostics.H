#ifndef LBM_CASES_LIDDRIVENCAVITY_CAVITYVALIDATIONDIAGNOSTICS_H
#define LBM_CASES_LIDDRIVENCAVITY_CAVITYVALIDATIONDIAGNOSTICS_H

#include "lid_driven_cavity/observation/CavityCenterlineDiagnostics.H"
#include "lbm/fields/MacroField.H"

#include <AMReX_BLassert.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace lbm::cases::lid_driven_cavity {

struct CavityValidationMetrics {
    int step = 0;
    amrex::Real rawLatticeMass = 0.0;
    amrex::Real rawLatticeReferenceMass = 0.0;
    amrex::Real relativeMassError = 0.0;
    amrex::Real physicalWeightedMass = 0.0;
    amrex::Real physicalWeightedReferenceMass = 0.0;
    amrex::Real physicalWeightedRelativeMassError = 0.0;
    amrex::Real rhoMin = 0.0;
    amrex::Real rhoMax = 0.0;
    amrex::Real maxStaticWallVelocityErrorOverUlid = 0.0;
    amrex::Real maxMovingLidVelocityErrorOverUlid = 0.0;
    amrex::Real maxMovingLidTangentialErrorOverUlid = 0.0;
    amrex::Real maxMovingLidNormalVelocityOverUlid = 0.0;
    amrex::Real maxAbsUzOverUlid = 0.0;
};

inline CavityValidationMetrics compute_cavity_validation_metrics(
    MacroField const& macro,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int step,
    amrex::Real rho0,
    amrex::Real lidSpeedLb,
    CavityDiagnosticGeometry const& geometry)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        geometry.onLatticeBoundaryNodes != 0,
        "Cavity boundary metrics require on-lattice physical boundary nodes");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        rho0 > 0.0 && lidSpeedLb > 0.0,
        "Cavity boundary metrics require positive rho0 and lid speed");

    amrex::MultiFab scratch(ba, dm, 10, 0);
    amrex::MultiFab const& macroMf = macro.multifab();
    const amrex::Box& domain = geom.Domain();
    const int loX = domain.smallEnd(0);
    const int hiX = domain.bigEnd(0);
    const int loY = domain.smallEnd(1);
    const int hiY = domain.bigEnd(1);

    for (amrex::MFIter mfi(macroMf, amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const m = macroMf.const_array(mfi);
        auto const s = scratch.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real rho = m(i, j, k, MacroField::Rho);
            const amrex::Real ux = m(i, j, k, MacroField::Ux);
            const amrex::Real uy = m(i, j, k, MacroField::Uy);
            const amrex::Real uz = m(i, j, k, MacroField::Uz);
            const bool onSideWall = i == loX || i == hiX;
            const bool onBottomWall = j == loY;
            const bool onMovingLid = j == hiY && !onSideWall;
            const bool onStaticWall = onSideWall || onBottomWall;

            amrex::Real physicalWeight = 1.0;
            if (i == loX || i == hiX) {
                physicalWeight *= 0.5;
            }
            if (j == loY || j == hiY) {
                physicalWeight *= 0.5;
            }

            const amrex::Real staticVelocityErrorSqr = onStaticWall ?
                ux * ux + uy * uy + uz * uz : 0.0;
            const amrex::Real lidUxError = ux - lidSpeedLb;
            const amrex::Real movingLidVelocityErrorSqr = onMovingLid ?
                lidUxError * lidUxError + uy * uy + uz * uz : 0.0;
            const amrex::Real lidTangentialErrorSqr = onMovingLid ?
                lidUxError * lidUxError : 0.0;
            const amrex::Real lidNormalVelocitySqr = onMovingLid ? uy * uy : 0.0;

            s(i, j, k, 0) = rho;
            s(i, j, k, 1) = physicalWeight * rho;
            s(i, j, k, 2) = physicalWeight;
            s(i, j, k, 3) = rho;
            s(i, j, k, 4) = -rho;
            s(i, j, k, 5) = staticVelocityErrorSqr;
            s(i, j, k, 6) = movingLidVelocityErrorSqr;
            s(i, j, k, 7) = lidTangentialErrorSqr;
            s(i, j, k, 8) = lidNormalVelocitySqr;
            s(i, j, k, 9) = uz * uz;
        });
    }
    amrex::Gpu::streamSynchronize();

    const amrex::Real nPts = static_cast<amrex::Real>(domain.numPts());
    const amrex::Real invLidSpeed = 1.0 / lidSpeedLb;
    CavityValidationMetrics metrics;
    metrics.step = step;
    metrics.rawLatticeMass = scratch.sum(0);
    metrics.rawLatticeReferenceMass = rho0 * nPts;
    metrics.relativeMassError = std::abs(
        metrics.rawLatticeMass - metrics.rawLatticeReferenceMass) /
        metrics.rawLatticeReferenceMass;
    metrics.physicalWeightedMass = scratch.sum(1);
    const amrex::Real physicalWeightSum = scratch.sum(2);
    metrics.physicalWeightedReferenceMass = rho0 * physicalWeightSum;
    metrics.physicalWeightedRelativeMassError = std::abs(
        metrics.physicalWeightedMass - metrics.physicalWeightedReferenceMass) /
        metrics.physicalWeightedReferenceMass;
    metrics.rhoMax = scratch.max(3);
    metrics.rhoMin = -scratch.max(4);
    metrics.maxStaticWallVelocityErrorOverUlid =
        std::sqrt(scratch.max(5)) * invLidSpeed;
    metrics.maxMovingLidVelocityErrorOverUlid =
        std::sqrt(scratch.max(6)) * invLidSpeed;
    metrics.maxMovingLidTangentialErrorOverUlid =
        std::sqrt(scratch.max(7)) * invLidSpeed;
    metrics.maxMovingLidNormalVelocityOverUlid =
        std::sqrt(scratch.max(8)) * invLidSpeed;
    metrics.maxAbsUzOverUlid = std::sqrt(scratch.max(9)) * invLidSpeed;
    return metrics;
}

inline std::string cavity_validation_metrics_summary(
    CavityValidationMetrics const& metrics)
{
    std::ostringstream os;
    os << std::setprecision(12)
       << "cavity_validation step=" << metrics.step
       << " relative_mass_error(raw_lattice)=" << metrics.relativeMassError
       << " raw_lattice_mass=" << metrics.rawLatticeMass
       << " raw_lattice_reference_mass=" << metrics.rawLatticeReferenceMass
       << " physical_weighted_mass=" << metrics.physicalWeightedMass
       << " physical_weighted_reference_mass="
       << metrics.physicalWeightedReferenceMass
       << " physical_weighted_relative_mass_error="
       << metrics.physicalWeightedRelativeMassError
       << " rho_min=" << metrics.rhoMin
       << " rho_max=" << metrics.rhoMax
       << " max_static_wall_velocity_error/Ulid="
       << metrics.maxStaticWallVelocityErrorOverUlid
       << " max_moving_lid_velocity_error/Ulid="
       << metrics.maxMovingLidVelocityErrorOverUlid
       << " max_moving_lid_tangential_error/Ulid="
       << metrics.maxMovingLidTangentialErrorOverUlid
       << " max_moving_lid_normal_velocity/Ulid="
       << metrics.maxMovingLidNormalVelocityOverUlid
       << " max_abs_uz/Ulid=" << metrics.maxAbsUzOverUlid;
    return os.str();
}

} // namespace lbm::cases::lid_driven_cavity

#endif
