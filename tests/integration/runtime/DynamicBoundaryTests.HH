#ifndef LBM_TESTS_DYNAMICBOUNDARYTESTS_HH
#define LBM_TESTS_DYNAMICBOUNDARYTESTS_HH

#include "integration/runtime/DynamicBoundaryTestSupport.H"
#include "integration/runtime/DynamicBoundaryForceTests.HH"
#include <stdexcept>

namespace lbm::tests::dynamic_amr::dynamic_boundary {
template<class Descriptor>
void run_relayout_checks()
{
    auto prior=amrex::DistributionMapping::strategy();
    amrex::DistributionMapping::strategy(amrex::DistributionMapping::ROUNDROBIN);
    auto spec=boundary_spec<Descriptor>(true);
    spec.regridInterval=0;
    auto layout=amr::prepare_dynamic_layout(spec,Descriptor::d);
    require(layout.ok(),"relayout dynamic geometry");
    auto bspec=boundaries(2);
    auto profile=reference::BoundaryEvolutionProfile<Descriptor>{};
    int const ebBaseline=amrex::EB2::IndexSpace::size();
    LayoutOwner<Descriptor,decltype(profile),RegionTagging> changed(*layout.layout,profile,RegionTagging{0},bspec);
    amr::StaticHierarchySpec fixedSpec;
    static_cast<amr::HierarchyGeometrySpec&>(fixedSpec)=spec;
    Real const z=spec.probHi[2];
    fixedSpec.regions={{1,{{2,2,0},{14,14,z}}},{2,{{8,8,0},{12,12,z}}}};
    auto prepared=amr::prepare_static_hierarchy<Descriptor>(fixedSpec,profile,bspec);
    require(prepared.ok(),"matching static boundary hierarchy");
    auto fixed=prepared.take();
    check_owners(changed); check_owners(*fixed);
    changed.observe(true,[&](auto const& view) {
        require(view.forceBatch.empty(),"initial boundary force must be empty");
        reference::check_boundary_initial(view,true);
    });
    compare_owners(changed,*fixed,false,"dynamic N0/static initial");
    changed.relayout(0);
    compare_owners(changed,*fixed,false,"identical layout leaves populations unchanged");
    changed.relayout(1);
    check_owners(changed);
    require(amrex::EB2::IndexSpace::size()==ebBaseline,"rebuild retained temporary EB resources");
    compare_owners(changed,*fixed,false,"same coverage BA reblock");
    auto pa=runtime::prepare_amr_evolution(changed,boundary_intent(),BGKDynamicsSelection{});
    auto pb=runtime::prepare_amr_evolution(*fixed,boundary_intent(),BGKDynamicsSelection{});
    require(pa.ok() && pb.ok(),"relayout prepared evolution");
    auto a=std::move(pa).take(),b=std::move(pb).take();
    std::vector<Real> traceA,traceB;
    int stepsA=0,stepsB=0;
    auto observeA=[&](auto const& view) {
        require(view.step==++stepsA,"relayout force steps");
        append_force_trace(traceA,view.forceBatch);
        if (view.step==1) reference::check_domain_first(view.levels[0],true);
    };
    auto observeB=[&](auto const& view) {
        require(view.step==++stepsB,"static reference force steps");
        append_force_trace(traceB,view.forceBatch);
    };
    amrex::Long remote=0;
    auto fine=[&](auto const& view) {
        if (view.fine.step!=1) return;
        std::vector<boundary::InternalBoundarySpec> owned;
        for (int index=0;index<3;++index)
            if ((index<2 ? 1 : 2)==view.level) owned.push_back(bspec.internal[index]);
        remote+=reference::check_internal_first<Descriptor>(view.fine,owned);
    };
    profiling::AmrTimestepProfiler profiler(profiling::TimestepProfilingMode::StageBreakdown,0,3,3);
    require(changed.evolve(a,1,true,observeA,fine,&profiler).ok(),"first step after BA rebuild");
    require(fixed->evolve(b,1,true,observeB).ok(),"static first step");
    if (amrex::ParallelDescriptor::NProcs()>1)
        require(remote>0,"reblocked YMS/Bouzidi oracle must cross actual rank boundaries");
    compare_owners(changed,*fixed,false,"continued after BA rebuild");
    compare_traces(traceA,traceB);

    changed.relayout(2); // Same coverage and BA, new DM; all fine FABs on rank zero.
    check_owners(changed);
    changed.observe(false,[&](auto const& view) {
        require(view.forceBatch.empty(),"observe after DM rebuild replayed force");
        auto const& owners=view.levels[2].mapping.ProcessorMap();
        int empty=std::find(owners.begin(),owners.end(),amrex::ParallelDescriptor::MyProc())==owners.end();
        amrex::ParallelDescriptor::ReduceIntSum(empty);
        require(empty==amrex::ParallelDescriptor::NProcs()-1,"DM fixture must exercise empty fine ranks");
    });
    // Comparison is after one non-equilibrium step, so reapplying Guo
    // initialization during metadata rebuilding would change this L0 field.
    compare_owners(changed,*fixed,false,"DM rebuild must not initialize Guo or populations");
    require(changed.evolve(a,2,true,observeA,nullptr,&profiler).ok(),"continued after DM rebuild");
    require(fixed->evolve(b,2,true,observeB).ok(),"static continuation");
    compare_traces(traceA,traceB);
    compare_owners(changed,*fixed,false,"N0 static full field and force compatibility");
    profiler.finalize();
    require(profiler.localDurations().boundaryPreparation.calls==0 &&
            profiler.localDurations().hierarchy.calls==6,"N0 evolve must not rebuild boundaries");
    require(amrex::EB2::IndexSpace::size()==ebBaseline,"DM rebuild retained EB resources");
    // Publication moved into the guarded scope with force draining/regrid.
    // A throwing observer must not leave a resumable, partly delivered step.
    bool threw=false;
    try {
        changed.evolve(a,1,false,[](auto const&) { throw std::runtime_error("observer fixture"); });
    } catch (std::runtime_error const&) {
        threw=true;
    }
    require(threw && !changed.evolve(a,1,false,[](auto const&) {}).ok(),
            "observer failure must leave the dynamic owner Failed");

    amrex::DistributionMapping::strategy(prior);
    amrex::Print()<<"dynamic_boundary_relayout_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()
                  <<" remote_donors="<<remote<<'\n';
}

template<class Descriptor>
auto domain_run(std::vector<int> const& segments)
{
    auto spec=boundary_spec<Descriptor>(false);
    auto bspec=boundaries(2,false);
    auto prepared=amr::prepare_dynamic_hierarchy<Descriptor>(
        spec,reference::BoundaryEvolutionProfile<Descriptor>{},DomainPatchTagging{},bspec);
    for (auto const& error:prepared.errors) amrex::Print()<<error<<'\n';
    require(prepared.ok() && prepared.simulation->numLevels()==1,"domain lifecycle starts on L0");
    auto owner=prepared.take();
    owner->observe(false,[](auto const& view) { require(view.forceBatch.empty(),"domain initial force"); });
    auto parameters=runtime::prepare_amr_evolution(*owner,boundary_intent(),BGKDynamicsSelection{});
    require(parameters.ok(),"domain lifecycle evolution");
    auto evolution=std::move(parameters).take();
    profiling::AmrTimestepProfiler profiler(profiling::TimestepProfilingMode::StageBreakdown,0,7,3);
    std::array<int,7> const counts{2,3,3,1,2,3,3};
    amrex::BoxArray previous;
    int observed=0;
    for (int count:segments) {
        auto run=owner->evolve(evolution,count,true,[&](auto const& view) {
            require(view.step==++observed && view.levels.size()==std::size_t(counts[observed-1]),
                    "domain-only new/move/remove/restore timeline");
            close(view.physicalTime,observed*0.005,"domain lifecycle physical time");
            require(view.forceBatch.empty(),"domain-only case must not fabricate internal force");
            if (view.step==1) reference::check_domain_first(view.levels[0],true);
            if (view.step==3) require(view.levels[1].boxes!=previous,"domain-only patch failed to move");
            if (view.levels.size()>1) previous=view.levels[1].boxes;
            for (int lev=1;lev<int(view.levels.size());++lev)
                for (int box=0;box<view.levels[lev].boxes.size();++box) {
                    auto const coarse=amrex::coarsen(view.levels[lev].boxes[box],
                        amrex::IntVect(1<<lev,1<<lev,Descriptor::d==3 ? 1<<lev : 1));
                    require(coarse.smallEnd(0)>=4 && coarse.smallEnd(1)>=4 &&
                            coarse.bigEnd(0)<=60 && coarse.bigEnd(1)<=60,
                            "dynamic fine coverage reached the four-cell domain guard");
                }
        },nullptr,&profiler);
        require(run.ok(),run.error);
    }
    require(observed==7,"domain lifecycle endpoint");
    profiler.finalize();
    require(profiler.localDurations().boundaryPreparation.calls==6 &&
            profiler.localDurations().hierarchy.calls==14,
            "domain regrid opportunities and actual rebuilds differ");
    return owner;
}

template<class Descriptor>
void run_domain_checks()
{
    auto continuous=domain_run<Descriptor>({7});
    auto segmented=domain_run<Descriptor>({3,4});
    compare_owners(*continuous,*segmented,true,"nonperiodic domain continuous/segmented");
    amrex::Print()<<"dynamic_boundary_domain_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<'\n';
}
} // namespace lbm::tests::dynamic_amr::dynamic_boundary
#endif
