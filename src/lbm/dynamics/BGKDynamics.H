#ifndef LBM_DYNAMICS_BGKDYNAMICS_H
#define LBM_DYNAMICS_BGKDYNAMICS_H

#include "lbm/dynamics/DynamicsModel.H"
#include "lbm/dynamics/Equilibrium.H"
#include "lbm/dynamics/Moments.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace lbm {

template <class Descriptor>
class BGKDynamics : public IsoThermalDynamicsModelBase<Descriptor> {
public:
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    explicit BGKDynamics(amrex::Real omega) noexcept
        : omega_(omega)
    {}

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real omega() const noexcept
    {
        return omega_;
    }

    template <class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void collide(Cell&& f) const noexcept
    {
        const amrex::Real rho = Moments<Descriptor>::rho(f);
        amrex::Real ux = 0.0;
        amrex::Real uy = 0.0;
        amrex::Real uz = 0.0;
        Moments<Descriptor>::velocity(f, rho, ux, uy, uz);

        for (int iPop = 0; iPop < Descriptor::q; ++iPop) {
            const amrex::Real feq =
                Equilibrium<Descriptor>::compute(iPop, rho, ux, uy, uz);
            f[iPop] -= omega_ * (f[iPop] - feq);
        }
    }

private:
    amrex::Real omega_;
};

} // namespace lbm

#endif
