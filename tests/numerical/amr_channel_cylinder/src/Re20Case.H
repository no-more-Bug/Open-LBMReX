#ifndef LBM_TESTS_NUMERICAL_RE20CASE_H
#define LBM_TESTS_NUMERICAL_RE20CASE_H

#include "lbm/amr/AmrBoundarySystem.H"
#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"
#include <string>

namespace amr_re20 {
using Real = amrex::Real;
constexpr Real meanSpeed = 0.035;
constexpr Real comparisonDt = 0.000875;

struct Initial {
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const&, Real) const noexcept
    { return {1, meanSpeed, 0, 0}; }
};

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, std::string const& layout, int maxBox,
                                            int baseDiameterIntervals = 20)
{
    bool const d3 = dimensions == 3;
    int const diameterIntervals = layout == "fine" ? 80 : baseDiameterIntervals;
    Real const h = Real(0.1)/diameterIntervals;
    lbm::amr::StaticHierarchySpec spec;
    spec.nCell = {22*diameterIntervals+1, 41*diameterIntervals/10+1, d3 ? diameterIntervals/10 : 1};
    spec.probLo = {-h/2, -h/2, 0};
    spec.probHi = {Real(2.2)+h/2, Real(0.41)+h/2, Real(0.01)};
    spec.isPeriodic = {0, 0, 1};
    spec.refRatio = amrex::IntVect(2, 2, d3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(maxBox, maxBox, d3 ? maxBox : 1)};
    if (layout == "amr") {
        // Requests use exact L0/L1 cell edges, including the Guo half-cell envelope.
        spec.regions.push_back({1, {{spec.probLo[0]+10*h, spec.probLo[1]+10*h, 0},
                                   {spec.probLo[0]+241*h, spec.probLo[1]+71*h, Real(0.01)}}});
        spec.regions.push_back({2, {{spec.probLo[0]+41*h/2, spec.probLo[1]+41*h/2, 0},
                                   {spec.probLo[0]+401*h/2, spec.probLo[1]+121*h/2, Real(0.01)}}});
    }
    return spec;
}

inline lbm::amr::AmrBoundarySpec boundaries(std::string const& scheme)
{
    using namespace lbm::boundary;
    lbm::amr::AmrBoundarySpec spec;
    DomainBoundarySpec inlet;
    inlet.name = "inlet"; inlet.location = DomainSide::XLo;
    inlet.physical = BoundaryPhysical::VelocityInlet;
    inlet.scheme = BoundaryScheme::GuoExtrapolation;
    inlet.profile = BoundaryProfileKind::Poiseuille;
    inlet.profileAxis = 1; inlet.hasProfileAxis = true;
    inlet.uMeanLb = meanSpeed; inlet.hasUMeanLb = true;
    spec.domain.push_back(inlet);
    DomainBoundarySpec outlet;
    outlet.name = "outlet"; outlet.location = DomainSide::XHi;
    outlet.physical = BoundaryPhysical::PressureOutlet;
    outlet.scheme = BoundaryScheme::GuoExtrapolation;
    outlet.profile = BoundaryProfileKind::Constant;
    outlet.rhoLb = 1; outlet.hasRhoLb = true;
    spec.domain.push_back(outlet);
    for (auto side : {DomainSide::YLo, DomainSide::YHi}) {
        DomainBoundarySpec wall;
        wall.name = side == DomainSide::YLo ? "bottom_wall" : "top_wall";
        wall.location = side;
        wall.physical = BoundaryPhysical::NoSlipWall;
        wall.scheme = BoundaryScheme::GuoExtrapolation;
        spec.domain.push_back(wall);
    }
    InternalBoundarySpec cylinder;
    cylinder.name = "cylinder_wall"; cylinder.objectName = "cylinder";
    cylinder.geometry = "cylinder";
    cylinder.physical = BoundaryPhysical::NoSlipWall; cylinder.hasPhysical = true;
    cylinder.scheme = scheme == "yms" ? BoundaryScheme::Yms : BoundaryScheme::Bouzidi;
    cylinder.centerPhys = {0.2, 0.2, 0.005}; cylinder.hasCenterPhys = true;
    cylinder.radiusPhys = 0.05; cylinder.hasRadiusPhys = true;
    cylinder.axis = "z"; cylinder.hasAxis = true;
    spec.internal.push_back(cylinder);
    spec.collectForces = true;
    return spec;
}
} // namespace amr_re20
#endif
