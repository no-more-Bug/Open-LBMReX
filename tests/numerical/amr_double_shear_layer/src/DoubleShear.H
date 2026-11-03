#ifndef LBM_TESTS_NUMERICAL_DOUBLESHEAR_H
#define LBM_TESTS_NUMERICAL_DOUBLESHEAR_H

#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"
#include <cmath>
#include <string>

namespace double_shear {
using Real=amrex::Real;
constexpr Real speed=0.1, reynolds=10000;

struct Initial {
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& x, Real) const noexcept
    {
        constexpr Real twoPi=6.283185307179586476925286766559;
        Real const y=x[1]<=Real(0.5) ? x[1]-Real(0.25) : Real(0.75)-x[1];
        return {1,speed*std::tanh(80*y),speed*Real(0.05)*std::sin(twoPi*(x[0]+Real(0.25))),0};
    }
};

inline lbm::amr::HierarchyGeometrySpec base_geometry(int dimensions, int n, int maxBox)
{
    bool const d3=dimensions==3;
    lbm::amr::HierarchyGeometrySpec spec;
    spec.nCell={n,n,d3 ? n/32 : 1};
    spec.probHi={1,1,d3 ? Real(1)/32 : Real(1)/16};
    spec.refRatio=amrex::IntVect(2,2,d3 ? 2 : 1);
    spec.maxGridSize={amrex::IntVect(maxBox,maxBox,d3 ? maxBox : 1)};
    return spec;
}

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, std::string const& layout,
                                             int uniformCells, int maxBox)
{
    bool const uniform=layout=="uniform";
    int const n=uniform ? uniformCells : 128;
    lbm::amr::StaticHierarchySpec spec;
    static_cast<lbm::amr::HierarchyGeometrySpec&>(spec)=base_geometry(dimensions,n,maxBox);
    auto band=[&](int lev, Real lo, Real hi) {
        auto low=spec.probLo, high=spec.probHi; low[1]=lo; high[1]=hi;
        spec.regions.push_back({lev,{low,high}});
    };
    if (uniform) return spec;
    if (layout=="2r") { band(1,Real(1)/8,Real(3)/8); band(1,Real(9)/16,Real(15)/16); }
    else if (layout=="baseline" || layout=="3l" || layout=="p1" || layout=="p2" || layout=="p3") {
        band(1,Real(0.5),1);
        if (layout=="p1" || layout=="p2" || layout=="p3") band(1,0,Real(3)/8);
    } else amrex::Abort("unknown double shear layout");
    if (layout!="baseline") {
        band(2,Real(5)/8,(layout=="p2" || layout=="p3") ? Real(1) : Real(7)/8);
        if (layout=="p3") band(2,0,Real(1)/8);
    }
    return spec;
}
} // namespace double_shear
#endif
