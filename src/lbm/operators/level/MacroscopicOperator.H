#ifndef LBM_OPERATORS_LEVEL_MACROSCOPICOPERATOR_H
#define LBM_OPERATORS_LEVEL_MACROSCOPICOPERATOR_H

#include "lbm/fields/LatticeField.H"
#include "lbm/fields/MacroField.H"
#include "lbm/kernels/MacroscopicKernels.H"
#include "lbm/operators/core/LevelOperator.H"

#include <AMReX_BLassert.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

#include <string_view>

namespace lbm::operators {

template <class Descriptor>
class MacroscopicOperator : public LevelOperator {
public:
    MacroscopicOperator() = default;

    std::string_view name() const noexcept override
    {
        return "macroscopic";
    }

    void apply(
        LatticeField<Descriptor> const& populations,
        MacroField& macro) const
    {
        AMREX_ALWAYS_ASSERT(macro.nComp() >= MacroField::NumComp);

        amrex::MultiFab const& f = populations.multifab();
        amrex::MultiFab& macroMf = macro.multifab();

        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.tilebox();
            auto const fArr = f.const_array(mfi);
            auto const macroArr = macroMf.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                ConstPopulationCell cell{fArr, i, j, k};
                amrex::Real rho = 0.0;
                amrex::Real ux = 0.0;
                amrex::Real uy = 0.0;
                amrex::Real uz = 0.0;
                kernels::MacroscopicKernel<Descriptor>::compute(cell, rho, ux, uy, uz);

                macroArr(i, j, k, MacroField::Rho) = rho;
                macroArr(i, j, k, MacroField::Ux) = ux;
                macroArr(i, j, k, MacroField::Uy) = uy;
                macroArr(i, j, k, MacroField::Uz) = uz;
            });
        }
    }
};

} // namespace lbm::operators

#endif
