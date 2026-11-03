#ifndef LBM_RUNTIME_LBMAMRPLOTFILE_H
#define LBM_RUNTIME_LBMAMRPLOTFILE_H

#include "lbm/amr/LbmAmrSimulation.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_PlotFileUtil.H>

#include <cmath>
#include <string>

namespace lbm::runtime {

struct AmrPlotfileResult {
    std::string error;
    bool ok() const noexcept { return error.empty(); }
};

/** Write an observed hierarchy with common reference units, without resampling.
 * All ranks call this with the same view and controls. The caller owns the
 * output path; existing paths follow AMReX's native plotfile behavior.
 * Invalid references or missing macro observations are rejected before I/O.
 */
template<class Descriptor>
AmrPlotfileResult write_amr_plotfile(
    std::string const& path, amr::LbmAmrView<Descriptor> const& view,
    amrex::Real rhoReference, amrex::Real velocityReference)
{
    if (!std::isfinite(rhoReference) || rhoReference <= 0.0 ||
        !std::isfinite(velocityReference) || velocityReference <= 0.0 ||
        !std::isfinite(1.0/rhoReference) || !std::isfinite(1.0/velocityReference)) {
        return {"invalid_plotfile_reference: finite positive reference units with finite inverses required"};
    }
    if (view.levels.empty() || view.refRatios.size()+1 != view.levels.size()) {
        return {"invalid_plotfile_view: levels and refinement ratios are inconsistent"};
    }
    for (std::size_t lev = 0; lev < view.levels.size(); ++lev) {
        if (!view.levels[lev].macro)
            return {"missing_plotfile_macro: level=" + std::to_string(lev)};
    }

    const int nlevels = static_cast<int>(view.levels.size());
    amrex::Vector<amrex::MultiFab> storage;
    storage.reserve(nlevels);
    amrex::Vector<amrex::MultiFab const*> fields;
    amrex::Vector<amrex::Geometry> geometries;
    amrex::Vector<int> steps;
    const amrex::Real invRho = 1.0/rhoReference, invVelocity = 1.0/velocityReference;
    for (auto const& level : view.levels) {
        auto& plot = storage.emplace_back(level.boxes, level.mapping, MacroField::NumComp, 0);
        auto const& macro = level.macro->multifab();
        for (amrex::MFIter mfi(plot, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            auto const source = macro.const_array(mfi);
            auto const destination = plot.array(mfi);
            amrex::ParallelFor(mfi.tilebox(), MacroField::NumComp,
                [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    destination(i,j,k,n) = source(i,j,k,n) * (n == MacroField::Rho ? invRho : invVelocity);
                });
        }
        fields.push_back(&plot);
        geometries.push_back(level.geometry);
        steps.push_back(level.step);
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Vector<std::string> variables{"rho_norm","ux_norm","uy_norm","uz_norm"};
    const amrex::Vector<amrex::IntVect> ratios(view.refRatios.begin(), view.refRatios.end());
    amrex::WriteMultiLevelPlotfile(path, nlevels, fields, variables, geometries,
        view.physicalTime, steps, ratios);
    return {};
}

} // namespace lbm::runtime

#endif
