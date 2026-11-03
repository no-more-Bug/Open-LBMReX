#ifndef LBM_TESTS_DYNAMICBOUNDARYFORCETESTS_HH
#define LBM_TESTS_DYNAMICBOUNDARYFORCETESTS_HH

#include "integration/runtime/DynamicBoundaryTestSupport.H"

namespace lbm::tests::dynamic_amr::dynamic_boundary {
inline void check_label(boundary::InternalForceStep const& step,int level,int nativeStep,int coarseStep)
{
    require(step.level==level && step.populationStep==nativeStep && step.coarsestStep==coarseStep,
            "dynamic force sample identity changed");
    Real const dt=level==1 ? 0.0025 : 0.00125;
    close(step.beginPhysicalTime,(nativeStep-1)*dt,"dynamic force begin time");
    close(step.endPhysicalTime,nativeStep*dt,"dynamic force end time");
    close(step.samplePhysicalTime(),(nativeStep-0.5)*dt,"dynamic force midpoint");
}

template<class Descriptor>
void check_batch(boundary::InternalForceBatch const& batch,
                  std::array<Velocity,3> const& unit,int grouping,int coarseStep)
{
    require(batch.boundarySteps.size()==6,"expected two L1 and four L2 native samples");
    for (int sample=0;sample<6;++sample) {
        int const level=sample<2 ? 1 : 2;
        int const nativeStep=sample<2 ? 2*coarseStep-1+sample : 4*coarseStep-3+sample-2;
        auto const& row=batch.boundarySteps[sample];
        check_label(row.step,level,nativeStep,coarseStep);
        require(row.boundaryForces.size()==std::size_t(level==1 ? 2 : 1),"dynamic force boundary row shape");
        for (int entry=0;entry<int(row.boundaryForces.size());++entry) {
            int const b=level==1 ? entry : 2;
            require(row.boundaryForces[entry].boundaryId.value==4+b,"sparse BoundaryId lost");
            auto const expected=reference::expected_force<Descriptor>(unit,b,nativeStep);
            for (int d=0;d<3;++d)
                reference::force_close(row.boundaryForces[entry].forceLb[d],expected[d],
                                       "native MEM oracle across actual regrid");
        }
    }
    require(batch.objectSteps.size()==std::size_t(grouping==1 ? 6 : 2),"dynamic Object step row count");
    for (int sample=0;sample<int(batch.objectSteps.size());++sample) {
        int const level=sample<2 ? 1 : 2;
        int const nativeStep=sample<2 ? 2*coarseStep-1+sample : 4*coarseStep-3+sample-2;
        auto const& row=batch.objectSteps[sample];
        check_label(row.step,level,nativeStep,coarseStep);
        int const object=grouping==1 ? (level==1 ? 0 : 1) : 1;
        require(row.objectForces.size()==1 && row.objectForces[0].objectId.value==object,
                "complete same-level Object identity");
        Velocity expected{};
        for (int b=0;b<3;++b) {
            bool const include=grouping==1 ? (level==1 ? b<2 : b==2) : b==1;
            if (include) {
                auto const term=reference::expected_force<Descriptor>(unit,b,nativeStep);
                for (int d=0;d<3;++d) expected[d]+=term[d];
            }
        }
        for (int d=0;d<3;++d)
            reference::force_close(row.objectForces[0].forceLb[d],expected[d],"same-level Object force sum");
    }
    require(batch.objectIntervals.size()==std::size_t(grouping==2),"cross-level interval shape");
    if (grouping==2) {
        auto const& row=batch.objectIntervals[0];
        require(row.coarsestStep==coarseStep && row.objectForces.size()==1 &&
                row.objectForces[0].objectId.value==0,"cross-level interval identity");
        close(row.beginPhysicalTime,(coarseStep-1)*0.005,"cross-level interval begin");
        close(row.endPhysicalTime,coarseStep*0.005,"cross-level interval end");
        Velocity expected{};
        // Independent weights: two L1 steps and four L2 steps per L0 interval.
        for (int n=2*coarseStep-1;n<=2*coarseStep;++n) {
            auto const term=reference::expected_force<Descriptor>(unit,0,n);
            for (int d=0;d<3;++d) expected[d]+=term[d]/2;
        }
        for (int n=4*coarseStep-3;n<=4*coarseStep;++n) {
            auto const term=reference::expected_force<Descriptor>(unit,2,n);
            for (int d=0;d<3;++d) expected[d]+=term[d]/4;
        }
        for (int d=0;d<3;++d)
            reference::force_close(row.objectForces[0].forceLb[d],expected[d],"L0 interval-weighted force");
    }
}

template<class Descriptor>
auto force_run(int grouping,std::vector<int> const& segments,std::vector<Real>& trace)
{
    auto spec=boundary_spec<Descriptor>(true);
    auto bspec=boundaries(grouping);
    auto prepared=amr::prepare_dynamic_hierarchy<Descriptor>(
        spec,reference::ForceRestProfile{},ExternalPatchTagging{},bspec);
    for (auto const& error:prepared.errors) amrex::Print()<<error<<'\n';
    require(prepared.ok(),"dynamic force hierarchy preparation");
    auto owner=prepared.take();
    check_owners(*owner);
    auto parameters=runtime::prepare_amr_evolution(*owner,boundary_intent(),reference::ForceProbeSelection{});
    require(parameters.ok(),"dynamic synthetic force parameters");
    auto evolution=std::move(parameters).take();
    require(evolution.level(1).dxLb()==2 && evolution.level(2).dxLb()==1,"fixed force lattice units");
    std::array<Velocity,3> unit;
    for (int b=0;b<3;++b)
        unit[b]=reference::force_reference<Descriptor>(owner->geometry(b<2 ? 1 : 2),bspec.internal[b]);
    int completed=0;
    amrex::Long initialFine=0;
    owner->observe(false,[&](auto const& view) {
        require(view.forceBatch.empty(),"initial observe must not fabricate force");
        initialFine=view.levels[2].boxes.numPts();
    });
    profiling::AmrTimestepProfiler profiler(profiling::TimestepProfilingMode::StageBreakdown,0,4,3);
    auto observer=[&](auto const& view) {
        require(view.step==++completed,"missing or duplicate L0 force delivery");
        check_owners(*owner);
        check_batch<Descriptor>(view.forceBatch,unit,grouping,view.step);
        append_force_trace(trace,view.forceBatch);
        auto const count=view.levels[2].boxes.numPts();
        require((view.step<=2 ? count>initialFine : count==initialFine),
                "force publication must accompany the actual post-regrid layout");
    };
    require(owner->evolve(evolution,0,false,observer).ok() && completed==0,"zero-step force no-op");
    for (int count:segments) {
        auto run=owner->evolve(evolution,count,false,observer,nullptr,&profiler);
        require(run.ok(),run.error);
        owner->observe(false,[](auto const& view) {
            require(view.forceBatch.empty(),"standalone observe replayed an old batch");
        });
    }
    require(completed==4,"dynamic force endpoint");
    profiler.finalize();
    auto const record=profiler.performanceRecord();
    require(record.has_value(),"missing boundary preparation performance record");
    auto const& local=profiler.localDurations();
    require(local.boundaryPreparation.calls==2 && local.hierarchy.calls==8 &&
            record->criticalRank.boundaryPreparation.calls==2 &&
            record->rankMax.boundaryPreparation.calls==2,
            "count actual boundary rebuilds, not four regrid opportunities");
    require(std::isfinite(record->rankMax.boundaryPreparation.seconds) &&
            record->rankMax.boundaryPreparation.seconds>0,"rebuild timing missing");
    amrex::Print()<<"dynamic_boundary_preparation d="<<Descriptor::d<<" grouping="<<grouping
        <<" rebuilds="<<record->rankMax.boundaryPreparation.calls
        <<" hierarchy_calls="<<local.hierarchy.calls
        <<" critical_seconds="<<record->criticalRank.boundaryPreparation.seconds
        <<" max_seconds="<<record->rankMax.boundaryPreparation.seconds<<'\n';
    return owner;
}

template<class Descriptor>
void run_force_checks()
{
    auto prior=amrex::DistributionMapping::strategy();
    amrex::DistributionMapping::strategy(amrex::DistributionMapping::ROUNDROBIN);
    std::vector<Real> sameLevel,continuousTrace,segmentedTrace;
    auto same=force_run<Descriptor>(1,{4},sameLevel);
    same.reset();
    auto continuous=force_run<Descriptor>(2,{4},continuousTrace);
    auto segmented=force_run<Descriptor>(2,{1,3},segmentedTrace);
    compare_traces(continuousTrace,segmentedTrace);
    compare_owners(*continuous,*segmented,true,"dynamic boundary force continuous/segmented");
    amrex::DistributionMapping::strategy(prior);
    amrex::Print()<<"dynamic_boundary_force_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<'\n';
}
} // namespace lbm::tests::dynamic_amr::dynamic_boundary
#endif
