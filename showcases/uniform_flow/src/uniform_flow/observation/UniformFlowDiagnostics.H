#ifndef LBM_CASES_UNIFORMFLOW_DIAGNOSTICS_H
#define LBM_CASES_UNIFORMFLOW_DIAGNOSTICS_H

#include "lbm/fields/MacroField.H"

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace lbm::cases::uniform_flow {

struct UniformFlowIntegrationGeometry {
    amrex::Real flowLengthLb = 1.0;
    amrex::GpuArray<int, AMREX_SPACEDIM> trapezoidalAxes{};
};

struct UniformFlowMetrics {
    int step = 0;
    amrex::Real flowThroughs = 0.0;
    amrex::Real e2U = 0.0;
    amrex::Real eInfU = 0.0;
    amrex::Real meanUx = 0.0;
    amrex::Real stdUxOverU0 = 0.0;
    amrex::Real maxAbsUyOverU0 = 0.0;
    amrex::Real maxAbsUzOverU0 = 0.0;
    amrex::Real relativeMassError = 0.0;
    amrex::Real rawLatticeMass = 0.0;
    amrex::Real rawLatticeReferenceMass = 0.0;
    amrex::Real physicalWeightedMass = 0.0;
    amrex::Real physicalWeightedReferenceMass = 0.0;
    amrex::Real physicalWeightedRelativeMassError = 0.0;
    amrex::Real rhoMeanMinusRho0 = 0.0;
    amrex::Real rhoStdOverRho0 = 0.0;
};

inline UniformFlowMetrics compute_uniform_flow_metrics(
    MacroField const& macro,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    int step,
    amrex::Real rho0,
    amrex::Real u0,
    UniformFlowIntegrationGeometry const& integration)
{
    amrex::MultiFab scratch(ba, dm, 9, 0);
    amrex::MultiFab const& macroMf = macro.multifab();
    const amrex::Box& domain = geom.Domain();
    const int domainLoX = domain.smallEnd(0);
    const int domainHiX = domain.bigEnd(0);
    const int domainLoY = domain.smallEnd(1);
    const int domainHiY = domain.bigEnd(1);
    const int domainLoZ = domain.smallEnd(2);
    const int domainHiZ = domain.bigEnd(2);
    const auto trapezoidalAxes = integration.trapezoidalAxes;

    for (amrex::MFIter mfi(macroMf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const m = macroMf.const_array(mfi);
        auto const s = scratch.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const amrex::Real rho = m(i, j, k, MacroField::Rho);
            const amrex::Real ux = m(i, j, k, MacroField::Ux);
            const amrex::Real uy = m(i, j, k, MacroField::Uy);
            const amrex::Real uz = m(i, j, k, MacroField::Uz);
            const amrex::Real dux = ux - u0;
            const amrex::Real uErrSqr = dux * dux + uy * uy + uz * uz;
            const amrex::Real rhoErr = rho - rho0;
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

            s(i, j, k, 0) = rho;
            s(i, j, k, 1) = rhoErr * rhoErr;
            s(i, j, k, 2) = ux;
            s(i, j, k, 3) = dux * dux;
            s(i, j, k, 4) = uErrSqr;
            s(i, j, k, 5) = uy < 0.0 ? -uy : uy;
            s(i, j, k, 6) = uz < 0.0 ? -uz : uz;
            s(i, j, k, 7) = physicalWeight * rho;
            s(i, j, k, 8) = physicalWeight;
        });
    }
    amrex::Gpu::streamSynchronize();

    const amrex::Real nPts = static_cast<amrex::Real>(geom.Domain().numPts());
    const amrex::Real absU0 = std::abs(u0);
    const amrex::Real invAbsU0 = absU0 > 0.0 ? 1.0 / absU0 : 0.0;
    const amrex::Real invRho0 = rho0 > 0.0 ? 1.0 / rho0 : 0.0;

    const amrex::Real rhoSum = scratch.sum(0);
    const amrex::Real rhoErrSqrSum = scratch.sum(1);
    const amrex::Real uxSum = scratch.sum(2);
    const amrex::Real uxErrSqrSum = scratch.sum(3);
    const amrex::Real uErrSqrSum = scratch.sum(4);
    const amrex::Real physicalWeightedMass = scratch.sum(7);
    const amrex::Real physicalWeightSum = scratch.sum(8);

    UniformFlowMetrics metrics;
    metrics.step = step;
    metrics.flowThroughs =
        static_cast<amrex::Real>(step) * absU0 / integration.flowLengthLb;
    metrics.e2U =
        absU0 > 0.0 ? std::sqrt(uErrSqrSum) / std::sqrt(nPts * absU0 * absU0) :
                      std::sqrt(uErrSqrSum / nPts);
    metrics.eInfU = std::sqrt(scratch.max(4)) * invAbsU0;
    metrics.meanUx = uxSum / nPts;
    const amrex::Real uxMeanErr = metrics.meanUx - u0;
    const amrex::Real uxVar = uxErrSqrSum / nPts - uxMeanErr * uxMeanErr;
    metrics.stdUxOverU0 = std::sqrt(uxVar > 0.0 ? uxVar : 0.0) * invAbsU0;
    metrics.maxAbsUyOverU0 = scratch.max(5) * invAbsU0;
    metrics.maxAbsUzOverU0 = scratch.max(6) * invAbsU0;
    metrics.rawLatticeMass = rhoSum;
    metrics.rawLatticeReferenceMass = rho0 * nPts;
    metrics.relativeMassError =
        std::abs(metrics.rawLatticeMass - metrics.rawLatticeReferenceMass) /
        metrics.rawLatticeReferenceMass;
    metrics.physicalWeightedMass = physicalWeightedMass;
    metrics.physicalWeightedReferenceMass = rho0 * physicalWeightSum;
    metrics.physicalWeightedRelativeMassError = std::abs(
        metrics.physicalWeightedMass - metrics.physicalWeightedReferenceMass) /
        metrics.physicalWeightedReferenceMass;
    metrics.rhoMeanMinusRho0 = rhoSum / nPts - rho0;
    const amrex::Real rhoMeanErr = metrics.rhoMeanMinusRho0;
    const amrex::Real rhoVar = rhoErrSqrSum / nPts - rhoMeanErr * rhoMeanErr;
    metrics.rhoStdOverRho0 = std::sqrt(rhoVar > 0.0 ? rhoVar : 0.0) * invRho0;
    return metrics;
}

inline std::string uniform_flow_metrics_summary(UniformFlowMetrics const& metrics)
{
    std::ostringstream os;
    os << std::setprecision(12)
       << "uniform_flow step=" << metrics.step
       << " flow_throughs=" << metrics.flowThroughs
       << " E_2_u=" << metrics.e2U
       << " E_inf_u=" << metrics.eInfU
       << " mean_ux=" << metrics.meanUx
       << " std_ux/u0=" << metrics.stdUxOverU0
       << " max_abs_uy/u0=" << metrics.maxAbsUyOverU0
       << " max_abs_uz/u0=" << metrics.maxAbsUzOverU0
       << " relative_mass_error(raw_lattice)=" << metrics.relativeMassError
       << " raw_lattice_mass=" << metrics.rawLatticeMass
       << " raw_lattice_reference_mass=" << metrics.rawLatticeReferenceMass
       << " physical_weighted_mass=" << metrics.physicalWeightedMass
       << " physical_weighted_reference_mass="
       << metrics.physicalWeightedReferenceMass
       << " physical_weighted_relative_mass_error="
       << metrics.physicalWeightedRelativeMassError
       << " rho_mean-rho0=" << metrics.rhoMeanMinusRho0
       << " rho_std/rho0=" << metrics.rhoStdOverRho0;
    return os.str();
}

inline std::string write_uniform_flow_metrics_csv(
    std::string const& path,
    UniformFlowMetrics const& metrics)
{
    std::string error;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream os(path, std::ios::out);
        if (!os.is_open()) {
            error = "Failed to open uniform-flow metrics CSV: " + path;
        } else {
            os << std::setprecision(17);
            os << "step,flow_throughs,E_2_u,E_inf_u,mean_ux,std_ux_over_u0,"
                  "max_abs_uy_over_u0,max_abs_uz_over_u0,"
                  "relative_mass_error,rho_mean_minus_rho0,rho_std_over_rho0,"
                  "raw_lattice_mass,raw_lattice_reference_mass,"
                  "physical_weighted_mass,physical_weighted_reference_mass,"
                  "physical_weighted_relative_mass_error\n";
            os << metrics.step << ',' << metrics.flowThroughs << ','
               << metrics.e2U << ',' << metrics.eInfU << ','
               << metrics.meanUx << ',' << metrics.stdUxOverU0 << ','
               << metrics.maxAbsUyOverU0 << ','
               << metrics.maxAbsUzOverU0 << ','
               << metrics.relativeMassError << ','
               << metrics.rhoMeanMinusRho0 << ','
               << metrics.rhoStdOverRho0 << ','
               << metrics.rawLatticeMass << ','
               << metrics.rawLatticeReferenceMass << ','
               << metrics.physicalWeightedMass << ','
               << metrics.physicalWeightedReferenceMass << ','
               << metrics.physicalWeightedRelativeMassError << '\n';
            if (!os) {
                error = "Failed to write uniform-flow metrics CSV: " + path;
            }
        }
    }
    amrex::BroadcastString(
        error,
        amrex::ParallelDescriptor::MyProc(),
        amrex::ParallelDescriptor::IOProcessorNumber(),
        amrex::ParallelDescriptor::Communicator());
    return error;
}

} // namespace lbm::cases::uniform_flow

#endif
