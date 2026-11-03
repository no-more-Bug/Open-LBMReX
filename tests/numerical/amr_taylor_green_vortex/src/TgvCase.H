#ifndef LBM_TESTS_AMR_TGV_CASE_H
#define LBM_TESTS_AMR_TGV_CASE_H

#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"
#include <cmath>

namespace amr_tgv {
using Real = amrex::Real;
struct Initial {
    Real speed;
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& x, Real) const noexcept
    {
        Real const twoPi = 6.283185307179586476925286766559;
        Real const rho = 1-Real(0.75)*speed*speed*(std::cos(2*twoPi*x[0])+std::cos(2*twoPi*x[1]));
        return {rho, -speed*std::cos(twoPi*x[0])*std::sin(twoPi*x[1]),
                     speed*std::sin(twoPi*x[0])*std::cos(twoPi*x[1]), 0};
    }
};

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, int coarseCells, bool uniform, int maxBox)
{
    bool const d3 = dimensions == 3;
    int const n = coarseCells*(uniform ? 2 : 1);
    lbm::amr::StaticHierarchySpec spec;
    spec.nCell = {n, n, d3 ? n/5 : 1};
    spec.probHi = {1, 1, Real(1)/5};
    spec.refRatio = amrex::IntVect(2, 2, d3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(maxBox, maxBox, d3 ? maxBox : 1)};
    if (!uniform)
        spec.regions.push_back({1, {{Real(1)/4, Real(1)/4, 0}, {Real(3)/4, Real(3)/4, Real(1)/5}}});
    return spec;
}
} // namespace amr_tgv
#endif
