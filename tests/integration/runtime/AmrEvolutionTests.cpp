#include "integration/runtime/AmrEvolutionTests.H"
#include "common/HostFieldSnapshot.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/dynamics/DynamicsModel.H"
#include "lbm/initialization/MacroscopicState.H"
#include "runtime/LbmAmrPreparation.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace lbm::tests::amr_evolution {
namespace {

#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = D3Q27Descriptor;
#else
using Descriptor = D2Q9Descriptor;
#endif
using Real = amrex::Real;

void require(bool condition, std::string const& message)
{
    if (!condition) amrex::Abort(message);
}

void close(Real actual, Real expected, std::string const& subject, Real scale = 1)
{
    Real const tolerance = 4096 * std::numeric_limits<Real>::epsilon() *
        std::max({scale,std::abs(actual),std::abs(expected)});
    require(std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual-expected) <= tolerance, subject);
}

struct Profile {
    bool rest = false;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const noexcept
    {
        if (rest) return {1,0,0,0};
        constexpr Real twoPi = 6.283185307179586476925286766559;
        Real const x = twoPi*p[0], y = twoPi*p[1], z = twoPi*p[2];
        return {1 + 0.001*std::sin(x)*std::cos(y)*(Descriptor::d == 3 ? std::cos(z) : 1),
            0.01*std::cos(y),0.007*std::sin(x),Descriptor::d == 3 ? 0.005*std::sin(z) : 0};
    }
};
using Simulation = amr::LbmAmrSimulation<Descriptor,Profile>;
using View = amr::LbmAmrView<Descriptor>;

class UninitializedSimulation final : public Simulation {
public:
    explicit UninitializedSimulation(amr::StaticHierarchyLayout layout)
        : Simulation(std::move(layout),Profile{}) {}
};

// The C0 E1/E2 central two/three-level geometry, plus its single-level limit.
amr::StaticHierarchySpec specification(int levels, Real height = Real(1)/16)
{
    amr::StaticHierarchySpec spec;
    int const n = Descriptor::d == 2 ? 32 : 16;
    spec.nCell = {n,n,Descriptor::d == 3 ? n : 1};
    spec.probHi = {1,1,Descriptor::d == 3 ? 1 : height};
    spec.refRatio = amrex::IntVect(2,2,Descriptor::d == 3 ? 2 : 1);
    spec.maxGridSize = {amrex::IntVect(8,8,Descriptor::d == 3 ? 8 : 1)};
    spec.gapFillRadiusCells = spec.coarseNestingMarginCells = 1;
    for (int lev = 1; lev < levels; ++lev) {
        Real const lo = lev == 1 ? Real(1)/4 : Real(3)/8;
        Real const hi = 1-lo;
        spec.regions.push_back({lev,{{lo,lo,Descriptor::d == 3 ? lo : 0},
            {hi,hi,Descriptor::d == 3 ? hi : height}}});
    }
    return spec;
}

auto simulation(int levels = 1, bool rest = false, Real height = Real(1)/16)
{
    auto prepared = amr::prepare_static_hierarchy<Descriptor>(specification(levels,height),Profile{rest});
    for (auto const& error : prepared.errors) amrex::Print() << error << '\n';
    require(prepared.ok(),"C2 hierarchy preparation");
    auto result = prepared.take();
    require(result->numLevels() == levels,"C2 actual hierarchy level count");
    return result;
}

config::LbmConfigIntent intent()
{
    config::LbmConfigIntent result;
    result.parameterizationKind = config::LbmParameterizationKind::FromTau;
    result.tau = 0.8; result.rhoRefLb = 1; result.uRefLb = 0.02; result.uRefPhys = 1;
    return result;
}

auto prepare_bgk(Simulation const& owner, config::LbmConfigIntent const& values = intent())
{
    auto result = runtime::prepare_amr_evolution(owner,values,BGKDynamicsSelection{});
    for (auto const& error : result.errors) amrex::Print() << error << '\n';
    require(result.ok(),"C2 evolution preparation");
    return std::move(result).take();
}

struct State {
    std::vector<std::pair<int,Real>> clocks;
    std::vector<amrex::MultiFab> populations, macro;
};

State observe(Simulation& owner)
{
    State state;
    owner.observe(true,[&](View const& view) {
        require(view.step == view.levels.front().step &&
            view.physicalTime == view.levels.front().physicalTime,"C2 L0/public clock mismatch");
        for (auto const& level : view.levels) {
            state.clocks.emplace_back(level.step,level.physicalTime);
            state.populations.push_back(host_field_snapshot(level.populations.multifab()));
            state.macro.push_back(host_field_snapshot(level.macro->multifab()));
        }
    });
    return state;
}

void compare_fields(amrex::MultiFab const& a, amrex::MultiFab const& b,
                    bool exact, bool macro, std::string const& subject)
{
    require(a.boxArray() == b.boxArray() && a.DistributionMap() == b.DistributionMap() &&
        a.nComp() == b.nComp() && a.nGrowVect() == b.nGrowVect(),subject+": field layout");
    for (amrex::MFIter mfi(a,false); mfi.isValid(); ++mfi) {
        auto const left = a.const_array(mfi), right = b.const_array(mfi);
        for (amrex::BoxIterator it(a[mfi].box()); it.ok(); ++it)
            for (int q = 0; q < a.nComp(); ++q) {
                if (exact) require(left(it(),q) == right(it(),q),subject+": values changed");
                else close(left(it(),q),right(it(),q),subject,macro && q ? Real(0.02) : Real(1));
            }
    }
}

void compare(State const& a, State const& b, bool exact, std::string const& subject)
{
    require(a.clocks == b.clocks,subject+": clocks changed");
    for (std::size_t lev = 0; lev < a.clocks.size(); ++lev) {
        compare_fields(a.populations[lev],b.populations[lev],exact,false,subject);
        compare_fields(a.macro[lev],b.macro[lev],exact,true,subject);
    }
}

template<class Evolution>
void reject_without_changes(Simulation& owner, Evolution const& evolution,
                            int steps, std::string const& keyword)
{
    auto const before = observe(owner);
    int calls = 0;
    auto result = owner.evolve(evolution,steps,true,[&](View const&) { ++calls; });
    require(!result.ok() && result.error.find(keyword) != std::string::npos,
            "C2 missing rejection: "+keyword+"; actual="+result.error);
    require(calls == 0,"C2 rejected evolve called observer");
    compare(before,observe(owner),true,"C2 rejection "+keyword);
}

template<class Evolution>
void check_closed_levels(Simulation const& owner, Evolution const& evolution)
{
    int const finest = owner.numLevels()-1;
    int const n0 = Descriptor::d == 2 ? 32 : 16;
    Real const finestCells = std::ldexp(Real(n0),finest);
    Real const fineDt = Real(0.02)/finestCells;
    require(evolution.numLevels() == owner.numLevels(),"C2 prepared level count");
    for (int lev = 0; lev <= finest; ++lev) {
        int const scale = 1 << (finest-lev);
        auto const& level = evolution.level(lev);
        Real const tau = Real(0.5)+Real(0.3)/scale;
        require(level.dtLb == scale && level.dxLb() == scale,"C2 common lattice scale");
        close(level.dtPhys,fineDt*scale,"C2 physical timestep",fineDt);
        close(level.lbm.tau,tau,"C2 coarse relaxation");
        close(level.lbm.omega,Real(1)/tau,"C2 inverse relaxation");
        close(level.lbm.lengthRefLb,finestCells/scale,"C2 local reference length");
        close(level.lbm.lengthRefPhys,1,"C2 physical reference length");
        close(level.lbm.nuLb,Real(0.1)/scale,"C2 local lattice viscosity");
        close(level.lbm.uRefLb,0.02,"C2 common velocity reference",0.02);
        close(level.lbm.reynolds,Real(0.2)*finestCells,"C2 common Reynolds number");
        Real const h = owner.geometry(lev).CellSize(0);
        close((tau-Real(0.5))*h*h/(3*level.dtPhys),
            Real(0.1)/(finestCells*Real(0.02)),"C2 common physical viscosity");
    }
}

void closure_checks()
{
    for (int levels : {1,2,3}) {
        auto owner = simulation(levels);
        auto values = intent();
        for (auto kind : {config::LbmParameterizationKind::FromTau,
                          config::LbmParameterizationKind::FromNuLb,
                          config::LbmParameterizationKind::FromReynolds}) {
            values = intent(); values.parameterizationKind = kind;
            if (kind != config::LbmParameterizationKind::FromTau) values.tau.reset();
            if (kind == config::LbmParameterizationKind::FromNuLb) values.nuLb = 0.1;
            if (kind == config::LbmParameterizationKind::FromReynolds) {
                Real const n = owner->geometry(levels-1).Domain().length(0);
                values.lengthRefLb = n; values.lengthRefPhys = 1; values.reynolds = Real(0.2)*n;
            }
            auto evolution = prepare_bgk(*owner,values);
            check_closed_levels(*owner,evolution);
        }
    }
    if constexpr (Descriptor::d == 2) {
        auto thicker = simulation(3,false,Real(1)/4);
        auto evolution = prepare_bgk(*thicker);
        check_closed_levels(*thicker,evolution);
    }
    auto owner = simulation(3);
    auto const before = observe(*owner);
    for (int scenario = 0; scenario < 6; ++scenario) {
        auto values = intent();
        if (scenario == 0) values.nuLb = 0.1; // Incompatible with from_tau.
        if (scenario == 1) values.tau = std::numeric_limits<Real>::quiet_NaN();
        if (scenario == 2) { values.lengthRefLb = 16; values.lengthRefPhys = 1; }
        if (scenario == 3) values.streamingBackend = "unsupported";
        if (scenario == 4) values.tau = std::nextafter(Real(0.5),Real(1));
        if (scenario == 5) values.uRefPhys = std::numeric_limits<Real>::denorm_min();
        auto failed = runtime::prepare_amr_evolution(*owner,values,BGKDynamicsSelection{});
        require(!failed.ok() && !failed.errors.empty(),"C2 invalid preparation succeeded");
        compare(before,observe(*owner),true,"C2 failed preparation");
    }
    auto evolution = prepare_bgk(*owner);
    int calls = 0;
    require(owner->evolve(evolution,0,false,[&](View const&) { ++calls; }).ok() && calls == 0,
            "C2 multilevel zero-step no-op");
    reject_without_changes(*owner,evolution,-1,"negative_steps");
    reject_without_changes(*owner,evolution,std::numeric_limits<int>::max()/4+1,"step_range");
}

State bgk_run(bool split)
{
    auto owner = simulation();
    auto evolution = prepare_bgk(*owner);
    int calls = 0;
    auto callback = [&](View const& view) {
        require(view.step == ++calls,"C2 completed-step observer cadence");
        close(view.physicalTime,calls*evolution.level(0).dtPhys,"C2 observed time",
              evolution.level(0).dtPhys);
    };
    require(owner->evolve(evolution,split ? 1 : 4,!split,callback).ok(),"C2 initial BGK segment");
    if (split) require(owner->evolve(evolution,3,true,callback).ok(),"C2 continued BGK segment");
    require(calls == 4,"C2 completed-step observer count");
    return observe(*owner);
}

void identity_checks(int levels)
{
    auto owner = simulation(levels);
    auto first = prepare_bgk(*owner);
    auto second = prepare_bgk(*owner);
    auto layout = amr::prepare_static_layout(specification(levels),Descriptor::d);
    require(layout.ok(),"C2 uninitialized fixture layout");
    UninitializedSimulation unready(std::move(*layout.layout));
    auto unreadyPreparation = runtime::prepare_amr_evolution(unready,intent(),BGKDynamicsSelection{});
    require(!unreadyPreparation.ok(),"C2 uninitialized preparation accepted");
    int unreadyCalls = 0;
    auto unreadyResult = unready.evolve(first,0,false,[&](View const&) { ++unreadyCalls; });
    require(unreadyResult.error.find("invalid_run_state") != std::string::npos && unreadyCalls == 0,
            "C2 uninitialized zero step accepted");
    using Evolution = decltype(first);
    static_assert(!std::is_copy_constructible_v<Evolution> &&
                  !std::is_copy_assignable_v<Evolution> &&
                  std::is_move_constructible_v<Evolution> &&
                  !std::is_move_assignable_v<Evolution>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(first.level(0))>>);
    auto const initial = observe(*owner);
    int calls = 0;
    auto callback = [&](View const&) { ++calls; };
    require(owner->evolve(first,0,true,callback).ok() && calls == 0,"C2 zero step before binding");
    compare(initial,observe(*owner),true,"C2 zero step before binding");
    reject_without_changes(*owner,first,-1,"negative_steps");
    auto anotherOwner = simulation(levels);
    auto foreign = prepare_bgk(*anotherOwner);
    reject_without_changes(*owner,foreign,1,"hierarchy_mismatch");
    require(owner->evolve(second,1,false,callback).ok() && calls == 1,
            "C2 zero/failed prechecks unexpectedly bound first preparation");
    reject_without_changes(*owner,first,1,"evolution_identity");
    auto sameValues = prepare_bgk(*owner);
    reject_without_changes(*owner,sameValues,1,"evolution_identity");
    auto moved = std::move(second);
    reject_without_changes(*owner,second,1,"hierarchy_mismatch");
    require(owner->evolve(moved,1,true,callback).ok() && calls == 2,"C2 moved bound preparation");
    auto const beforeZero = observe(*owner);
    require(owner->evolve(first,0,true,callback).ok() &&
            owner->evolve(second,0,true,callback).ok() && calls == 2,
            "C2 zero step must ignore binding and a moved-from shell");
    compare(beforeZero,observe(*owner),true,"C2 zero step after binding");
    owner->observe(false,[&](View const&) {
        auto result = owner->evolve(moved,1,false,callback);
        require(result.error.find("invalid_run_state") != std::string::npos && calls == 2,
                "C2 observer reentry rejection");
    });
    compare(beforeZero,observe(*owner),true,"C2 observer reentry state");
    reject_without_changes(*owner,moved,std::numeric_limits<int>::max(),"step_range");
    if (levels != 1) return;
    auto timeOwner = simulation();
    auto values = intent(); values.uRefPhys = std::numeric_limits<Real>::min();
    auto hugeTime = prepare_bgk(*timeOwner,values);
    reject_without_changes(*timeOwner,hugeTime,std::numeric_limits<int>::max(),"time_range");
    auto normal = prepare_bgk(*timeOwner);
    require(timeOwner->evolve(normal,1,false,[](View const&) {}).ok(),
            "C2 time precheck failure bound its preparation");
}

template<class D>
struct IncrementDynamics : LocalDynamicsModel<D> {
    Real increment;
    AMREX_GPU_HOST_DEVICE explicit IncrementDynamics(Real value) noexcept : increment(value) {}
    template<class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void collide(Cell&& cell) const noexcept { cell[0] += increment; }
};

struct IncrementSelection {
    std::vector<config::ResolvedLbmConfig>* prepared;
    int failAt = -1;
    template<class D> using Dynamics = IncrementDynamics<D>;
    template<class D>
    DynamicsPreparationResult<Dynamics<D>> prepare(config::ResolvedLbmConfig const& facts) const
    {
        int const index = static_cast<int>(prepared->size());
        prepared->push_back(facts);
        DynamicsPreparationResult<Dynamics<D>> result;
        if (index == failAt) result.errors.push_back("tests_selected_preparation_failure");
        else result.dynamics.emplace(Real(1)/1024);
        return result;
    }
};

void injection_checks()
{
    auto layered = simulation(3,true);
    auto const before = observe(*layered);
    std::vector<config::ResolvedLbmConfig> selected;
    auto failed = runtime::prepare_amr_evolution(*layered,intent(),IncrementSelection{&selected,1});
    require(!failed.ok() && !failed.errors.empty(),"C2 selected dynamics failure was lost");
    compare(before,observe(*layered),true,"C2 selected preparation failure");
    selected.clear();
    auto preparedLevels = runtime::prepare_amr_evolution(*layered,intent(),IncrementSelection{&selected});
    require(preparedLevels.ok() && selected.size() == 3,"C2 selection did not prepare every level");
    auto levels = std::move(preparedLevels).take();
    for (int lev = 0; lev < 3; ++lev) {
        Real const scale = 1 << (2-lev);
        close(selected[lev].tau,Real(0.5)+Real(0.3)/scale,"C2 selected level tau");
        close(selected[lev].nuLb,Real(0.1)/scale,"C2 selected local viscosity");
    }
    check_closed_levels(*layered,levels);
    auto owner = simulation(1,true);
    auto initial = observe(*owner);
    selected.clear();
    auto prepared = runtime::prepare_amr_evolution(*owner,intent(),IncrementSelection{&selected});
    require(prepared.ok() && selected.size() == 1,"C2 single-level selection preparation");
    auto evolution = std::move(prepared).take();
    int calls = 0;
    auto callback = [&](View const& view) {
        ++calls;
        require(view.step == calls && selected.size() == 1,"C2 reconstructed selection during evolve");
        auto actual = host_field_snapshot(view.levels.front().populations.multifab());
        for (amrex::MFIter mfi(actual,false); mfi.isValid(); ++mfi) {
            auto const values = actual.const_array(mfi);
            auto const start = initial.populations.front().const_array(mfi);
            for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it)
                for (int q = 0; q < Descriptor::q; ++q)
                    close(values(it(),q),start(it(),q)+(q == 0 ? Real(calls)/1024 : 0),
                          "C2 selected collision result");
        }
    };
    require(owner->evolve(evolution,1,true,callback).ok(),"C2 injected first segment");
    require(owner->evolve(evolution,3,false,callback).ok(),"C2 injected continued segment");
    require(calls == 4 && selected.size() == 1,"C2 injected operation count");
}

void fine_callback_failure_checks()
{
    auto owner = simulation(2,true);
    auto evolution = prepare_bgk(*owner);
    int fineCalls = 0, synchronizedCalls = 0;
    bool caught = false;
    auto synchronized = [&](View const&) { ++synchronizedCalls; };
    try {
        owner->evolve(evolution,1,false,synchronized,
            [&](amr::FineSubstepView<Descriptor> const& view) {
                require(view.level == 1 && view.substep == amr::FineSubstep::First,
                        "fine callback failure must occur after the first fine commit");
                ++fineCalls;
                throw std::runtime_error("fine callback failure");
            });
    } catch (std::runtime_error const& error) {
        caught = std::string(error.what()) == "fine callback failure";
    }
    require(caught && fineCalls == 1 && synchronizedCalls == 0,
            "partial fine advance published a synchronized view or lost the callback exception");
    auto resumed = owner->evolve(evolution,1,false,synchronized);
    require(resumed.error.find("invalid_run_state") != std::string::npos && synchronizedCalls == 0,
            "partial fine advance remained resumable");
    auto prepared = runtime::prepare_amr_evolution(*owner,intent(),BGKDynamicsSelection{});
    require(!prepared.ok() && !prepared.errors.empty() &&
            prepared.errors.front().find("invalid_run_state") != std::string::npos,
            "partial fine advance accepted a new preparation");
}

template<class T>
void scalar_time_checks()
{
    for (T units : {T(1e-9),T(1),T(1e9)})
        require(amr::detail::representable_evolution_time<T>(units,units/T(100),1000),
                "C2 ordinary accumulated time range rejected");
    require(!amr::detail::representable_evolution_time<T>(
                std::numeric_limits<T>::max(),std::numeric_limits<T>::max(),1),
            "C2 overflowing time endpoint accepted");
}

} // namespace

void lifecycle_checks()
{
    scalar_time_checks<float>();
    scalar_time_checks<double>();
    require(!amr::detail::representable_evolution_time<float>(0,1,16777218),
            "C2 float accumulation can stagnate at 2^24");
    require(amr::detail::representable_evolution_time<double>(0,1,16777218),
            "C2 representable double accumulation rejected");
    closure_checks();
    auto continuous = bgk_run(false);
    auto segmented = bgk_run(true);
    compare(continuous,segmented,false,"C2 continuous versus segmented BGK");
    identity_checks(1);
    identity_checks(3);
    injection_checks();
    fine_callback_failure_checks();
    amrex::Print() << Descriptor::name()
        << " C2 preparation/binding/collision lifecycle: PASS; multilevel_validation=SEPARATE_C4\n";
}

} // namespace lbm::tests::amr_evolution
