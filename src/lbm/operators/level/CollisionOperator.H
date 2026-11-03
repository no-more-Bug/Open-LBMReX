#ifndef LBM_OPERATORS_LEVEL_COLLISIONOPERATOR_H
#define LBM_OPERATORS_LEVEL_COLLISIONOPERATOR_H

#include "lbm/dynamics/DynamicsModel.H"
#include "lbm/fields/LatticeField.H"
#include "lbm/kernels/CollisionKernels.H"
#include "lbm/operators/core/LevelOperator.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>

#include <string_view>

namespace lbm::operators {

template <class Descriptor, class Dynamics>
class CollisionOperator : public LevelOperator {
public:
    static_assert(
        is_dynamics_model_for_descriptor<Dynamics, Descriptor>::value,
        "Dynamics must derive from LocalDynamicsModel with the same descriptor");

    explicit CollisionOperator(Dynamics dynamics) noexcept
        : dynamics_(dynamics)
    {}

    std::string_view name() const noexcept override
    {
        return "collision";
    }

    void apply(LatticeField<Descriptor>& field) const
    {
        amrex::MultiFab& f = field.multifab();
        #ifdef AMREX_USE_OMP
        #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
        #endif
        for (amrex::MFIter mfi(f, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Box const& bx = mfi.tilebox();
            auto const fArr = f.array(mfi);
            auto const localDynamics = dynamics_;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                PopulationCell cell{fArr, i, j, k};
                kernels::CollisionKernel<Descriptor>::apply(cell, localDynamics);
            });
        }
    }

private:
    Dynamics dynamics_;
};

} // namespace lbm::operators

#endif
