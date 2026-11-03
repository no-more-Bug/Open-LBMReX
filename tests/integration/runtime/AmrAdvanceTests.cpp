#include "integration/runtime/AmrEvolutionTests.H"
#include "integration/runtime/AmrEvolutionTestCase.H"
#include "integration/runtime/AmrAdvanceEvidence.H"
#include "integration/runtime/AmrDonorEvidence.H"
#include "integration/runtime/AmrMatrixFixtures.H"
#include "lbm/dynamics/BGKDynamicsSelection.H"
#include "lbm/dynamics/DynamicsModel.H"
#include <AMReX_Print.H>
#include <optional>

namespace lbm::tests::amr_evolution {
namespace {

struct RunResult {
    std::vector<Frame> frames;
    std::vector<std::vector<int>> mapping;
};

class MappingScope {
public:
    explicit MappingScope(amrex::DistributionMapping::Strategy strategy)
        : previous_(amrex::DistributionMapping::strategy()) { amrex::DistributionMapping::strategy(strategy); }
    ~MappingScope() { amrex::DistributionMapping::strategy(previous_); }
private:
    amrex::DistributionMapping::Strategy previous_;
};

RunResult run_case(Fixture const& fixture, Real tau, bool uniform,
                           int steps, bool split, int sampling, bool emptyFine = false,
                           amrex::DistributionMapping::Strategy strategy=amrex::DistributionMapping::SFC,
                           std::string const& group="advance")
{
    MappingScope mappingScope(strategy);
    std::string const key=fixture.name+(uniform ? "-uniform" : "-wave")+"-tau"+std::to_string(tau)+
        "-n"+std::to_string(steps)+"-split"+std::to_string(split)+"-sample"+std::to_string(sampling)+
        (strategy==amrex::DistributionMapping::ROUNDROBIN ? "-rr" : "-sfc");
    std::string const directory="artifacts/evolution-"+group+"/"+key;
    AdvanceEvidence evidence(directory,fixture.name,steps,tau,uniform);
    RunResult result;
    EvolutionProfile const profile{uniform};
    auto owner=make_simulation(fixture,profile);
    auto prepared=runtime::prepare_amr_evolution(*owner,evolution_intent(tau),BGKDynamicsSelection{});
    require(prepared.ok(),fixture.name+": evolution preparation");
    auto evolution=std::move(prepared).take();
    std::optional<amr_reference::HierarchyReference> reference;
    std::vector<Frame> initial, final;
    owner->observe(true,[&](EvolutionView const& view) {
        initial=capture(view);
        reference.emplace(initial_reference(view,tau,profile));
        for (auto const& level : view.levels)
            result.mapping.emplace_back(level.mapping.ProcessorMap().begin(),level.mapping.ProcessorMap().end());
        for (std::size_t lev=0; lev<view.levels.size(); ++lev) {
            check_populations(initial[lev].populations,reference->levels()[lev].current,fixture.name+": initial");
            check_macros(initial[lev],reference->levels()[lev].current);
            close(evolution.level(int(lev)).dtPhys,reference->levels()[lev].dt,"independent physical timestep");
            close(evolution.level(int(lev)).lbm.omega,reference->levels()[lev].omega,"independent relaxation");
        }
        evidence.record(view,initial,reference->levels());
        if (group=="matrix") {
            // RR supplies the remote-origin cases; SFC may align adjacent-level owners.
            bool const remote=fixture.requireRemoteDonor && strategy==amrex::DistributionMapping::ROUNDROBIN;
            write_donor_evidence(view,fixture.name,directory,remote);
        }
        if constexpr (Descriptor::d==3) {
            if (!uniform) {
                Real low=1, high=-1;
                auto const& macro=initial[0].macro.values;
                for (std::size_t n=3; n<macro.size(); n+=4) {
                    low=std::min(low,macro[n]); high=std::max(high,macro[n]);
                }
                require(high-low>Real(0.001),"D3 engineering initial state must vary with z");
            }
        }
        if (emptyFine) {
            auto const& fine=view.levels[1];
            require(fine.boxes.size()==1,fixture.name+": expected one actual fine Box");
            int localBoxes=0;
            for (amrex::MFIter it(fine.populations.multifab()); it.isValid(); ++it) ++localBoxes;
            int empty=localBoxes==0;
            int emptyRank=empty ? amrex::ParallelDescriptor::MyProc() : -1;
            amrex::ParallelDescriptor::ReduceIntSum(empty);
            amrex::ParallelDescriptor::ReduceIntMax(emptyRank);
            require(empty==amrex::ParallelDescriptor::NProcs()-1,"E1 actual empty fine ranks");
            amrex::Print() << "advance_empty_fine sample=" << fixture.name << " empty_ranks=" << empty
                           << " empty_rank=" << emptyRank << '\n';
        }
    });
    int synchronizedCalls=0, referenceStep=0;
    std::size_t cursor=0;
    std::vector<int> fineCalls(owner->numLevels(),0);
    std::vector<amr_reference::ReferenceSample> samples;
    auto next_reference=[&](int coarsestStep) {
        if (coarsestStep==referenceStep) return;
        require(coarsestStep==referenceStep+1,"reference coarsest schedule skipped a step");
        samples=reference->advance(); cursor=0; referenceStep=coarsestStep;
    };
    auto fine=[&](amr::FineSubstepView<Descriptor> const& view) {
        next_reference(view.coarsestStep);
        require(cursor<samples.size(),"unexpected fine callback");
        auto const& expected=samples[cursor++];
        require(view.level==expected.level && view.fine.step==expected.step &&
            view.coarsestStep==expected.coarsestStep &&
            (view.substep==amr::FineSubstep::Second)==expected.second && !view.fine.macro,
            fixture.name+": fine phase/clock identity");
        close(view.fine.physicalTime,expected.time,"fine callback time",reference->levels()[view.level].dt);
        ++fineCalls[view.level];
        auto reentrant=owner->evolve(evolution,0,false,[](EvolutionView const&) {});
        require(reentrant.error.find("invalid_run_state")!=std::string::npos,"fine observer evolve reentry");
        if (sampling==2 && view.level!=owner->numLevels()-1) return;
        auto actual=static_hierarchy::snapshot(view.fine.populations.multifab(),view.fine.geometry.Domain());
        check_populations(actual,expected.populations,fixture.name+": fine L"+std::to_string(view.level)+" step "+std::to_string(expected.step));
    };
    auto synchronized=[&](EvolutionView const& view) {
        next_reference(view.step);
        ++synchronizedCalls;
        require(view.step==synchronizedCalls && view.physicalTime==view.levels[0].physicalTime,
                fixture.name+": synchronized identity");
        if (sampling) require(cursor==samples.size(),"missing fine callbacks before synchronized view");
        final=capture(view);
        for (int lev=0; lev<owner->numLevels(); ++lev) {
            require(view.levels[lev].step==view.step*(1 << lev) &&
                    view.levels[lev].physicalTime==view.physicalTime,"local step or common time mismatch");
            close(view.physicalTime,reference->levels()[lev].time,"reference synchronized time",evolution.level(0).dtPhys);
            check_populations(final[lev].populations,reference->levels()[lev].current,
                fixture.name+": synchronized L"+std::to_string(lev)+" step "+std::to_string(view.step));
            check_macros(final[lev],reference->levels()[lev].current);
            if (uniform) {
                auto const subject=fixture.name+": constant state";
                for (bool macro : {false,true}) {
                    auto const& values=macro ? final[lev].macro.values : final[lev].populations.values;
                    auto const& before=macro ? initial[lev].macro.values : initial[lev].populations.values;
                    for (std::size_t n=0; n<values.size(); ++n)
                        close(values[n],before[n],subject,macro && n%4 ? Real(0.02) : Real(1));
                }
            }
        }
        evidence.record(view,final,reference->levels());
    };
    require(owner->evolve(evolution,0,true,synchronized,fine).ok() && synchronizedCalls==0 && cursor==0,
            "zero step called an observer");
    auto advance=[&](int count) {
        auto result=sampling ? owner->evolve(evolution,count,true,synchronized,fine)
                             : owner->evolve(evolution,count,true,synchronized);
        require(result.ok(),fixture.name+": "+result.error);
    };
    if (split) { advance(1); advance(steps-1); }
    else advance(steps);
    require(synchronizedCalls==steps,"synchronized callback count");
    for (int lev=1; lev<owner->numLevels(); ++lev) {
        int minimum=fineCalls[lev], maximum=fineCalls[lev];
        amrex::ParallelDescriptor::ReduceIntMin(minimum);
        amrex::ParallelDescriptor::ReduceIntMax(maximum);
        require(minimum==maximum && minimum==(sampling ? steps*(1 << lev) : 0),"rank fine callback count/order");
    }
    amrex::Print() << "advance_pass sample=" << fixture.name << " uniform=" << uniform
        << " tau_finest=" << tau << " coarse_steps=" << steps << " split=" << split
        << " sampling=" << sampling << " levels=" << owner->numLevels()
        << " ranks=" << amrex::ParallelDescriptor::NProcs() << '\n';
    result.frames=std::move(final);
    return result;
}

template<class D>
struct CountedCollision : LocalDynamicsModel<D> {
    Real increment;
    AMREX_GPU_HOST_DEVICE explicit CountedCollision(Real value) noexcept : increment(value) {}
    template<class Cell>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void collide(Cell&& cell) const noexcept { cell[0]+=increment; }
};
struct CountedSelection {
    int* preparations;
    template<class D> using Dynamics=CountedCollision<D>;
    template<class D>
    DynamicsPreparationResult<Dynamics<D>> prepare(config::ResolvedLbmConfig const&) const
    {
        int const lev=(*preparations)++;
        DynamicsPreparationResult<Dynamics<D>> result;
        result.dynamics.emplace(Real(lev+1)/1024);
        return result;
    }
};

void collision_schedule()
{
    auto owner=make_simulation(central_fixture(3,8),EvolutionProfile{true});
    int preparations=0;
    auto prepared=runtime::prepare_amr_evolution(*owner,evolution_intent(0.65),CountedSelection{&preparations});
    require(prepared.ok() && preparations==3,"collision selection must prepare each level once");
    auto evolution=std::move(prepared).take();
    std::vector<Frame> initial;
    owner->observe(false,[&](EvolutionView const& view) { initial=capture(view); });
    int synchronizedCalls=0, fineCalls=0;
    auto check=[&](int lev, amr::LbmAmrLevelView<Descriptor> const& level) {
        auto values=static_hierarchy::snapshot(level.populations.multifab(),level.geometry.Domain());
        auto const& before=initial[lev].populations;
        for (std::size_t n=0; n<values.values.size(); ++n)
            close(values.values[n],before.values[n]+(n%Descriptor::q==0 ? Real(level.step*(lev+1))/1024 : 0),
                  "injected per-level collision count");
    };
    auto synchronized=[&](EvolutionView const& view) {
        ++synchronizedCalls;
        require(preparations==3,"collision preparation repeated inside evolution");
        for (int lev=0; lev<owner->numLevels(); ++lev) check(lev,view.levels[lev]);
    };
    auto fine=[&](amr::FineSubstepView<Descriptor> const& view) { ++fineCalls; check(view.level,view.fine); };
    require(owner->evolve(evolution,1,false,synchronized,fine).ok(),"injected first segment");
    require(owner->evolve(evolution,3,false,synchronized,fine).ok(),"injected continued segment");
    require(synchronizedCalls==4 && fineCalls==24 && preparations==3,"injected level schedule");
    amrex::Print() << "advance_collision_schedule levels=3 coarse_steps=4 local_steps=4,8,16: PASS\n";
}

template<class T>
void synchronized_time_checks()
{
    auto accepted=[](T b,T target,T d,T actual,T fine,int count) {
        return amr::detail::synchronized_evolution_time(b,target,d,actual,fine,count).first;
    };
    int const exponent=std::numeric_limits<T>::digits<30 ? 40 : 400;
    T const infinity=std::numeric_limits<T>::infinity();
    for (int e : {-exponent,0,exponent}) {
        T const unit=std::ldexp(T(1),e);
        for (int count : {1,4,8}) {
            T const begin=T(0.1)*unit, dt=T(0.2)*unit, fine=dt/count;
            T actual=begin;
            for (int n=0; n<count; ++n) actual+=fine;
            require(accepted(begin,begin+dt,dt,actual,fine,count),"legal repeated clock additions rejected");
        }
        T near=std::nextafter(unit,infinity), far=unit;
        for (int n=0; n<64; ++n) far=std::nextafter(far,infinity);
        require(accepted(T(0),unit,unit,near,unit/4,4),"one ULP clock difference rejected");
        require(!accepted(T(0),unit,unit,far,unit/4,4),"64 ULP clock difference accepted");
    }
    for (T actual : {T(0.75),T(1.25),T(0.5),T(2)})
        require(!accepted(T(0),T(1),T(1),actual,T(0.25),4),"missed/extra step or wrong dt accepted");
    T const begin=std::ldexp(T(1),std::numeric_limits<T>::digits-1), target=begin+16;
    for (T offset : {T(-1),T(1)})
        require(accepted(begin,target,T(16),target+offset,T(4),4),"quarter-step tolerance boundary rejected");
    for (T offset : {T(-2),T(2)})
        require(!accepted(begin,target,T(16),target+offset,T(4),4),"quarter-step tolerance cap missing");
    require(!accepted(T(1),T(2),T(1),T(1),T(0.25),4) &&
            !accepted(T(1),T(1),T(1),T(2),T(0.25),4),"nonadvancing clock accepted");
    for (T bad : {std::numeric_limits<T>::quiet_NaN(),infinity,-infinity})
        require(!accepted(T(0),T(1),T(1),bad,T(0.25),4) &&
                !accepted(T(0),bad,T(1),T(1),T(0.25),4),"nonfinite clock accepted");
}

} // namespace

void advance_checks()
{
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts/evolution-advance");
    amrex::ParallelDescriptor::Barrier();
    require(std::numeric_limits<Real>::digits==53,"engineering matrix requires FP64");
    synchronized_time_checks<float>(); synchronized_time_checks<double>();
    int const n=Descriptor::d == 2 ? 32 : 16;
    auto const e1=central_fixture(2,n), e2=central_fixture(3,8);
    run_case(central_fixture(1,8),0.65,false,2,false,1);
    for (Real tau : {Real(0.65),Real(0.8)}) {
        run_case(e1,tau,true,2,false,1,true);
        run_case(e2,tau,true,2,false,1);
    }
    run_case(e1,0.65,false,4,false,1,true);
    auto const continuous=run_case(e2,0.65,false,4,false,1).frames;
    compare_frames(run_case(e2,0.65,false,4,true,1).frames,continuous,"continuous versus 1+3");
    compare_frames(run_case(e2,0.65,false,4,false,0).frames,continuous,"sampling disabled");
    compare_frames(run_case(e2,0.65,false,4,false,2).frames,continuous,"finest-only consumer");
    for (bool periodicShift : {false,true})
        run_case(four_level_fixture(periodicShift),0.65,false,2,false,1);
    collision_schedule();
    amrex::Print() << Descriptor::name() << " C4 recursive advance/reference/clocks/sampling: PASS\n";
}

void matrix_checks()
{
    if (amrex::ParallelDescriptor::IOProcessor()) std::filesystem::remove_all("artifacts/evolution-matrix");
    amrex::ParallelDescriptor::Barrier();
    require(std::numeric_limits<Real>::digits==53,"engineering matrix requires FP64");
    using DM=amrex::DistributionMapping;
    auto const sharedThrough=through_fixture(8).name;
    for (auto const& fixture : matrix_fixtures())
        if (fixture.name!=sharedThrough) // This run is shared with E7 below.
            run_case(fixture,0.65,false,2,false,1,false,DM::ROUNDROBIN,"matrix");
    int const largeBox=Descriptor::d==2 ? 32 : 16;
    for (bool through : {false,true}) {
        auto fixture=[&](int box) { return through ? through_fixture(box) : central_fixture(3,box); };
        auto const coarse=run_case(fixture(largeBox),0.65,false,2,false,1,false,DM::SFC,"matrix");
        auto const fine=run_case(fixture(8),0.65,false,2,false,1,false,DM::SFC,"matrix");
        auto const alternate=run_case(fixture(8),0.65,false,2,false,1,false,DM::ROUNDROBIN,"matrix");
        compare_frames(fine.frames,coarse.frames,"E7 same actual A / different Box decomposition");
        compare_frames(alternate.frames,fine.frames,"E7 same actual A / different DM");
        bool differentBoxes=false;
        for (std::size_t lev=0; lev<fine.frames.size(); ++lev) {
            differentBoxes=differentBoxes || fine.frames[lev].populations.boxes!=coarse.frames[lev].populations.boxes;
            require(fine.frames[lev].populations.boxes==alternate.frames[lev].populations.boxes,
                    "E7 DM comparison changed ordered Boxes");
        }
        require(differentBoxes,"E7 selected grid sizes produced identical actual Boxes");
        require(amrex::ParallelDescriptor::NProcs()==1 || alternate.mapping!=fine.mapping,
                "E7 selected strategies produced identical actual owners");
        amrex::Print() << "matrix_equivalence sample=" << fixture(8).name
            << " actual_owner_change=" << (alternate.mapping!=fine.mapping) << '\n';
    }
    amrex::Print() << Descriptor::name() << " C6 layout/DM matrix: PASS\n";
}

} // namespace lbm::tests::amr_evolution
