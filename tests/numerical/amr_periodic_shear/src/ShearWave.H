#ifndef LBM_TESTS_NUMERICAL_SHEARWAVE_H
#define LBM_TESTS_NUMERICAL_SHEARWAVE_H

#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"
#include <cmath>
#include <string>
#include <utility>

namespace periodic_shear {
using Real=amrex::Real;
constexpr Real viscosity=Real(1)/1280, background=0.05, amplitude=0.001;
constexpr Real twoPi=6.283185307179586476925286766559;

struct Wave {
    amrex::GpuArray<Real,3> direction{}, polarization{};
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& x, Real) const noexcept
    {
        Real phase=0, norm2=0;
        for (int d=0; d<3; ++d) { phase+=direction[d]*x[d]; norm2+=direction[d]*direction[d]; }
        Real const s=amplitude*std::sin(twoPi*phase), speed=background/std::sqrt(norm2);
        return {1,speed*direction[0]+s*polarization[0],
                  speed*direction[1]+s*polarization[1],speed*direction[2]+s*polarization[2]};
    }
};

inline Wave wave(int dimensions, std::string const& mode)
{
    if (mode=="axis") return dimensions==2 ? Wave{{1,0,0},{0,1,0}} : Wave{{0,0,1},{1,0,0}};
    if (mode!="diagonal") amrex::Abort("unknown shear mode");
    if (dimensions==2) return {{1,1,0},{-1/std::sqrt(Real(2)),1/std::sqrt(Real(2)),0}};
    return {{1,1,1},{1/std::sqrt(Real(6)),1/std::sqrt(Real(6)),-2/std::sqrt(Real(6))}};
}

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, int coarseCells,
                                             std::string const& layout, int maxBox)
{
    bool const d3=dimensions==3, uniform=layout=="uniform";
    int const n=coarseCells*(uniform ? 2 : 1);
    lbm::amr::StaticHierarchySpec spec;
    spec.nCell={n,n,d3 ? n : 1};
    spec.probHi={1,1,d3 ? Real(1) : Real(1)/16};
    spec.refRatio=amrex::IntVect(2,2,d3 ? 2 : 1);
    spec.maxGridSize={amrex::IntVect(maxBox,maxBox,d3 ? maxBox : 1)};
    auto add=[&](amrex::Array<Real,3> lo, amrex::Array<Real,3> hi) {
        if (!d3) { lo[2]=0; hi[2]=spec.probHi[2]; }
        spec.regions.push_back({1,{lo,hi}});
    };
    if (uniform) return spec;
    if (layout=="half") {
        auto lo=spec.probLo;
        lo[d3 ? 2 : 0]=0.5;
        add(lo,spec.probHi);
    } else if (layout=="separated") {
        std::pair<Real,Real> const intervals[]{{0,0.125},{0.375,0.625},{0.875,1}};
        for (auto const& interval : intervals) {
            auto lo=spec.probLo, hi=spec.probHi;
            lo[d3 ? 2 : 0]=interval.first; hi[d3 ? 2 : 0]=interval.second;
            add(lo,hi);
        }
    } else if (layout=="corner" || layout=="shell") {
        // Non-overlapping rectangular partition of an outer box minus a box.
        amrex::Array<Real,3> lo{0.125,0.125,0.125}, hi{0.875,0.875,0.875};
        Real const innerLo=layout=="corner" ? Real(0.5) : Real(5)/16;
        Real const innerHi=layout=="corner" ? Real(0.875) : Real(11)/16;
        for (int d=0; d<dimensions; ++d) {
            auto end=hi; end[d]=innerLo; add(lo,end);
            if (innerHi<hi[d]) { auto begin=lo; begin[d]=innerHi; add(begin,hi); }
            lo[d]=innerLo; hi[d]=innerHi;
        }
    } else amrex::Abort("unknown shear layout");
    return spec;
}

inline bool fine_sample(int step, int fineCells)
{
    for (int t : {1,2,4}) if (step<=t*fineCells && step>=t*fineCells-3) return true;
    return false;
}
} // namespace periodic_shear
#endif
