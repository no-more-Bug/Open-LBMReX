#ifndef LBM_TESTS_AMREVOLUTIONTESTCASE_H
#define LBM_TESTS_AMREVOLUTIONTESTCASE_H

#include "common/AmrHierarchyReference.H"
#include "integration/runtime/StaticHierarchyTestSupport.H"
#include "integration/runtime/StaticHierarchyTopologyFixtures.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"
#include <cmath>
#include <string_view>

namespace lbm::tests::amr_evolution {
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = D3Q27Descriptor;
#else
using Descriptor = D2Q9Descriptor;
#endif
using Real = amrex::Real;
using Fixture = static_hierarchy::topology::Fixture;
using Frame = static_hierarchy::Frame;
using Snapshot = static_hierarchy::ValidSnapshot;

struct EvolutionProfile {
    bool uniform = false;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const noexcept
    {
        if (uniform) return {1,0.01,0.007,Descriptor::d == 3 ? Real(0.005) : Real(0)};
        constexpr Real twoPi = 6.283185307179586476925286766559;
        Real const x=twoPi*p[0], y=twoPi*p[1], z=twoPi*p[2];
        return {1+0.001*std::sin(x)*std::cos(y)*(Descriptor::d == 3 ? std::cos(z) : 1),
            0.01*std::cos(y),0.007*std::sin(x),Descriptor::d == 3 ? 0.005*std::sin(z) : 0};
    }
};
using EvolutionSimulation = amr::LbmAmrSimulation<Descriptor,EvolutionProfile>;
using EvolutionView = amr::LbmAmrView<Descriptor>;

void require(bool condition, std::string_view message);
void close(Real actual, Real expected, std::string_view subject, Real scale = 1);
Fixture central_fixture(int levels, int box);
Fixture four_level_fixture(bool periodicShift = false);
std::unique_ptr<EvolutionSimulation> make_simulation(Fixture const& fixture, EvolutionProfile profile);
config::LbmConfigIntent evolution_intent(Real tau);
amr_reference::HierarchyReference initial_reference(EvolutionView const& view, Real finestTau,
                                                    EvolutionProfile profile);
std::vector<Frame> capture(EvolutionView const& view);
void check_populations(Snapshot const& actual, amr_reference::SparseVolumeGrid const& expected,
                       std::string const& subject);
void check_macros(Frame const& frame, amr_reference::SparseVolumeGrid const& reference);
void compare_frames(std::vector<Frame> const& actual, std::vector<Frame> const& expected,
                    std::string const& subject);

} // namespace lbm::tests::amr_evolution
#endif
