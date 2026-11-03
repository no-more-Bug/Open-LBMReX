#ifndef LBM_TESTS_NUMERICAL_CAVITYCASE_H
#define LBM_TESTS_NUMERICAL_CAVITYCASE_H

#include "lbm/amr/AmrBoundarySystem.H"
#include "lbm/amr/StaticRefinement.H"
#include "lbm/initialization/MacroscopicState.H"

namespace amr_cavity {
using Real = amrex::Real;
constexpr Real lidSpeed = 0.1;

struct Initial {
    AMREX_GPU_HOST_DEVICE
    lbm::initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const&, Real) const noexcept
    { return {1, 0, 0, 0}; }
};

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, std::string const& layout, int maxBox)
{
    bool const d3 = dimensions == 3;
    int const n = layout == "fine" ? 256 : 128;
    lbm::amr::StaticHierarchySpec spec;
    spec.nCell = {n, n, d3 ? n/32 : 1};
    spec.probHi = {1, 1, d3 ? Real(1)/32 : Real(1)/16};
    spec.isPeriodic = {0, 0, 1};
    spec.refRatio = amrex::IntVect(2, 2, d3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(maxBox, maxBox, d3 ? maxBox : 1)};
    if (layout == "amr") spec.regions.push_back({1, {{0.25, 0.25, 0}, {0.75, 0.75, spec.probHi[2]}}});
    return spec;
}

inline lbm::amr::AmrBoundarySpec boundaries()
{
    using namespace lbm::boundary;
    lbm::amr::AmrBoundarySpec spec;
    for (auto side : {DomainSide::XLo, DomainSide::XHi, DomainSide::YLo, DomainSide::YHi}) {
        DomainBoundarySpec wall;
        wall.name = "wall_" + std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.scheme = BoundaryScheme::HalfwayBounceBack;
        if (side == DomainSide::YHi) {
            wall.physical = BoundaryPhysical::MovingWall;
            wall.profile = BoundaryProfileKind::Uniform;
            wall.prescribedULb = {lidSpeed, 0, 0};
            wall.hasPrescribedULb = true;
        }
        spec.domain.push_back(std::move(wall));
    }
    return spec;
}
} // namespace amr_cavity
#endif
