#ifndef LBM_RUNTIME_SINGLELEVELPLOTFILE_H
#define LBM_RUNTIME_SINGLELEVELPLOTFILE_H

#include "lbm/fields/MacroField.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>

#include <filesystem>
#include <string>

namespace lbm::runtime {

inline void write_single_level_plotfile(
    std::string const& path,
    MacroField const& macro,
    amrex::Geometry const& geometry,
    amrex::BoxArray const& boxArray,
    amrex::DistributionMapping const& distributionMapping,
    amrex::Real rhoReference,
    amrex::Real velocityReference,
    int step,
    amrex::Real physicalTime)
{
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::filesystem::remove_all(path);
    }
    amrex::ParallelDescriptor::Barrier();

    amrex::MultiFab plot(
        boxArray,
        distributionMapping,
        MacroField::NumComp,
        0);
    amrex::MultiFab const& macroMf = macro.multifab();
    const amrex::Real invRho = 1.0 / rhoReference;
    const amrex::Real invVelocity = 1.0 / velocityReference;
    for (amrex::MFIter mfi(plot, amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        amrex::Box const& bx = mfi.tilebox();
        auto const m = macroMf.const_array(mfi);
        auto const p = plot.array(mfi);
        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                p(i, j, k, 0) = m(i, j, k, MacroField::Rho) * invRho;
                p(i, j, k, 1) = m(i, j, k, MacroField::Ux) * invVelocity;
                p(i, j, k, 2) = m(i, j, k, MacroField::Uy) * invVelocity;
                p(i, j, k, 3) = m(i, j, k, MacroField::Uz) * invVelocity;
            });
    }
    amrex::Gpu::streamSynchronize();

    amrex::Vector<std::string> const variableNames{
        "rho_norm", "ux_norm", "uy_norm", "uz_norm"};
    amrex::WriteSingleLevelPlotfile(
        path,
        plot,
        variableNames,
        geometry,
        physicalTime,
        step);
}

} // namespace lbm::runtime

#endif
