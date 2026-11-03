#ifndef LBM_AMR_REFINEMENTTAGGING_HH
#define LBM_AMR_REFINEMENTTAGGING_HH

#include "lbm/kernels/MacroscopicKernels.H"
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>

namespace lbm::amr {

template<class Descriptor, class Rule>
void tag_valid_cells(amrex::MultiFab const& populations, amrex::MultiFab const* velocity,
                     amrex::iMultiFab const& covered, amrex::Geometry const& geometry,
                     int level, amrex::Real time, Rule rule, amrex::TagBoxArray& tags,
                     RefinementConstraints const* constraints, amrex::TagBoxArray* optionalTags)
{
    amrex::Gpu::DeviceVector<amrex::Box> requiredBoxes, allowedBoxes;
    if (constraints) {
        auto upload = [](amrex::BoxArray const& cells, auto& device) {
            auto const boxes = cells.boxList();
            device.resize(boxes.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, boxes.begin(), boxes.end(), device.begin());
        };
        upload(constraints->requiredTags[level], requiredBoxes);
        auto const ratio = amrex::IntVect(2,2,Descriptor::d == 3 ? 2 : 1);
        upload(amrex::coarsen(constraints->allowedCells[level+1], ratio), allowedBoxes);
    }
    auto const* required = requiredBoxes.data();
    auto const* allowed = allowedBoxes.data();
    int const requiredCount = static_cast<int>(requiredBoxes.size());
    int const allowedCount = static_cast<int>(allowedBoxes.size());
    bool const constrained = constraints != nullptr;
    bool const collectOptional = optionalTags != nullptr;
    auto const lo = geometry.ProbLoArray();
    auto const dx = geometry.CellSizeArray();
    auto const inverseDx = geometry.InvCellSizeArray();
    for (amrex::MFIter mfi(populations,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto const f = populations.const_array(mfi);
        auto const mask = covered.const_array(mfi);
        auto const tag = tags.array(mfi);
        amrex::Array4<amrex::TagBox::TagType> optional;
        if (optionalTags) optional = optionalTags->array(mfi);
        amrex::Array4<amrex::Real const> u;
        if (velocity) u = velocity->const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            if (constrained) {
                bool permitted = false;
                for (int b = 0; b < allowedCount; ++b)
                    if (allowed[b].contains(i,j,k)) { permitted = true; break; }
                if (!permitted) {
                    tag(i,j,k) = amrex::TagBox::CLEAR;
                    return;
                }
                for (int b = 0; b < requiredCount; ++b) {
                    if (required[b].contains(i,j,k)) {
                        tag(i,j,k) = amrex::TagBox::SET;
                        return;
                    }
                }
            }
            ConstPopulationCell populationsCell{f,i,j,k};
            amrex::Real rho,ux,uy,uz;
            kernels::MacroscopicKernel<Descriptor>::compute(populationsCell,rho,ux,uy,uz);
            TaggingCell<Descriptor> cell{populationsCell,rho,{ux,uy,uz},
                {lo[0]+(i+0.5)*dx[0],lo[1]+(j+0.5)*dx[1],lo[2]+(k+0.5)*dx[2]},
                inverseDx,u,level,time,mask(i,j,k) != 0};
            auto const value = rule(cell) ? amrex::TagBox::SET : amrex::TagBox::CLEAR;
            tag(i,j,k) = value;
            if (collectOptional) optional(i,j,k) = value;
        });
    }
    // These boxes are borrowed by asynchronous kernels and owned by this call.
    if (constrained) amrex::Gpu::streamSynchronize();
}

} // namespace lbm::amr
#endif
