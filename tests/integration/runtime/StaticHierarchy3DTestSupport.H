#ifndef LBM_TESTS_STATICHIERARCHY3DTESTSUPPORT_H
#define LBM_TESTS_STATICHIERARCHY3DTESTSUPPORT_H

#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"

namespace lbm::tests::static_hierarchy::d3 {

using Descriptor = D3Q27Descriptor;
constexpr Real timestep = 0.00125;

struct Profile {
    bool uniform = false;
    Real density = 1.0;
    amrex::GpuArray<Real,3> origin{0,0,0}, inverseLength{1,1,1};
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const noexcept
    {
        if (uniform) return {density,0.01,0.007,0.005};
        constexpr Real twoPi = 6.283185307179586476925286766559;
        const Real x = twoPi*(p[0]-origin[0])*inverseLength[0];
        const Real y = twoPi*(p[1]-origin[1])*inverseLength[1];
        const Real z = twoPi*(p[2]-origin[2])*inverseLength[2];
        return {density+0.001*std::sin(x)*std::cos(y)*std::cos(z),
                0.01*std::cos(y),0.007*std::sin(z),0.005*std::cos(x)};
    }
};

using Simulation = amr::LbmAmrSimulation<Descriptor,Profile>;
using View = amr::LbmAmrView<Descriptor>;

inline amr::StaticHierarchySpec base_spec(int boxSize = 16)
{
    amr::StaticHierarchySpec spec;
    spec.nCell = {16,16,16};
    spec.probHi = {1,1,1};
    spec.refRatio = amrex::IntVect(2);
    spec.maxGridSize = {amrex::IntVect(boxSize)};
    spec.gapFillRadiusCells = 1;
    spec.coarseNestingMarginCells = 1;
    return spec;
}

inline Profile make_profile(amr::StaticHierarchySpec const& spec, bool uniform = false)
{
    Profile profile;
    profile.uniform = uniform;
    for (int d = 0; d < 3; ++d) {
        profile.origin[d] = spec.probLo[d];
        profile.inverseLength[d] = 1.0/(spec.probHi[d]-spec.probLo[d]);
    }
    return profile;
}

inline config::LbmConfigIntent intent()
{
    config::LbmConfigIntent result;
    result.parameterizationKind = config::LbmParameterizationKind::FromNuLb;
    result.rhoRefLb = 1.0; result.uRefLb = 0.02; result.uRefPhys = 1.0;
    result.nuLb = 0.05; result.lengthRefLb = 16.0; result.lengthRefPhys = 1.0;
    return result;
}

// Independent Palabos-order D3Q27 vectors; tensor-product one-axis weights.
constexpr std::array<std::array<int,3>,27> directions{{
    {0,0,0}, {-1,0,0}, {0,-1,0}, {0,0,-1}, {-1,-1,0}, {-1,1,0},
    {-1,0,-1}, {-1,0,1}, {0,-1,-1}, {0,-1,1}, {-1,-1,-1},
    {-1,-1,1}, {-1,1,-1}, {-1,1,1}, {1,0,0}, {0,1,0}, {0,0,1},
    {1,1,0}, {1,-1,0}, {1,0,1}, {1,0,-1}, {0,1,1}, {0,1,-1},
    {1,1,1}, {1,1,-1}, {1,-1,1}, {1,-1,-1}
}};

inline std::array<Real,4> expected_macros(amrex::Box const& domain, amrex::IntVect const& index,
                                         bool uniform = false, Real density = 1.0)
{
    const auto cell = index - domain.smallEnd();
    const Real pi = std::acos(Real(-1));
    const Real x = 2*pi*(cell[0]+0.5)/domain.length(0);
    const Real y = 2*pi*(cell[1]+0.5)/domain.length(1);
    const Real z = 2*pi*(cell[2]+0.5)/domain.length(2);
    return {uniform ? density : density+0.001*std::sin(x)*std::cos(y)*std::cos(z),
            uniform ? 0.01 : 0.01*std::cos(y),
            uniform ? 0.007 : 0.007*std::sin(z),
            uniform ? 0.005 : 0.005*std::cos(x)};
}

inline void check_fields(Frame const& frame, bool uniform = false, Real density = 1.0)
{
    auto const& f = frame.populations;
    auto const& m = frame.macro;
    require(f.components == 27 && m.components == 4 &&
        f.cellIndices == m.cellIndices && f.owners == m.owners,
        "D3 snapshot shape/coverage");
    for (std::size_t index = 0; index < f.owners.size(); ++index) {
        if (!f.owners[index]) continue;
        const auto initial = expected_macros(f.domain,f.cell_at(index),uniform,density);
        std::array<Real,4> moments{};
        for (int q = 0; q < 27; ++q) {
            const Real value = f.values[index*27+q];
            require(std::isfinite(value), "D3 nonfinite population");
            moments[0] += value;
            Real weight = 1.0, cu = 0.0, u2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                moments[d+1] += directions[q][d]*value;
                weight *= directions[q][d] == 0 ? 2.0/3.0 : 1.0/6.0;
                cu += directions[q][d]*initial[d+1];
                u2 += initial[d+1]*initial[d+1];
            }
            if (frame.step == 0 || uniform)
                close(value,weight*initial[0]*(1+3*cu+4.5*cu*cu-1.5*u2), "D3 independent equilibrium");
        }
        require(moments[0] > 0, "D3 nonpositive density");
        for (int n = 0; n < 4; ++n) {
            close(m.values[index*4+n], n ? moments[n]/moments[0] : moments[0], "D3 independent moment");
            if (frame.step == 0 || uniform) close(m.values[index*4+n],initial[n], "D3 initial/uniform macro");
        }
    }
}

inline std::vector<Frame> observe(Simulation& simulation, bool uniform = false, Real density = 1.0)
{
    std::vector<Frame> frames;
    simulation.observe(true,[&](View const& view) {
        for (auto const& level : view.levels) {
            require(level.populations.multifab().boxArray() == level.boxes &&
                level.populations.multifab().DistributionMap() == level.mapping, "D3 final BA/DM binding");
            auto frame = capture(level.step,level.physicalTime,level.geometry,
                level.populations.multifab(), &level.macro->multifab());
            frame.level = static_cast<int>(frames.size());
            check_fields(frame,uniform,density);
            frames.push_back(std::move(frame));
        }
    });
    return frames;
}

} // namespace lbm::tests::static_hierarchy::d3
#endif
