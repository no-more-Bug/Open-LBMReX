#ifndef LBM_TESTS_AMR_UNIFORM_FLOW_CASE_H
#define LBM_TESTS_AMR_UNIFORM_FLOW_CASE_H

#include "lbm/amr/AmrBoundarySystem.H"
#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"

namespace amr_uniform_flow {
using Real = amrex::Real;
constexpr Real speed = 0.1;

struct Initial {
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const&, Real) const noexcept
    { return {1, speed, 0, 0}; }
};

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, int maxBox)
{
    bool const d3 = dimensions == 3;
    Real const h = Real(1)/32, x0 = -h/2, depth = Real(1)/8;
    lbm::amr::StaticHierarchySpec spec;
    spec.nCell = {33, 32, d3 ? 4 : 1};
    spec.probLo = {x0, 0, 0}; spec.probHi = {1+h/2, 1, depth};
    spec.isPeriodic = {0, 1, 1};
    spec.refRatio = amrex::IntVect(2, 2, d3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(maxBox, maxBox, d3 ? maxBox : 1)};
    spec.regions = {{1, {{x0+8*h, Real(1)/4, 0}, {x0+24*h, Real(3)/4, depth}}},
                    {2, {{x0+12*h, Real(3)/8, 0}, {x0+20*h, Real(5)/8, depth}}}};
    return spec;
}

inline lbm::amr::AmrBoundarySpec boundaries()
{
    using namespace lbm::boundary;
    lbm::amr::AmrBoundarySpec spec;
    for (auto side : {DomainSide::XLo, DomainSide::XHi}) {
        DomainBoundarySpec boundary;
        boundary.name = side == DomainSide::XLo ? "inlet" : "outlet";
        boundary.location = side;
        boundary.physical = side == DomainSide::XLo ? BoundaryPhysical::VelocityInlet : BoundaryPhysical::VelocityOutlet;
        boundary.scheme = BoundaryScheme::GuoExtrapolation;
        boundary.profile = BoundaryProfileKind::Uniform;
        boundary.prescribedULb = {speed, 0, 0}; boundary.hasPrescribedULb = true;
        spec.domain.push_back(boundary);
    }
    return spec;
}
} // namespace amr_uniform_flow
#endif
