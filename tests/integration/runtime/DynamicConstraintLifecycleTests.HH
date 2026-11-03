#ifndef LBM_TESTS_DYNAMICCONSTRAINTLIFECYCLETESTS_HH
#define LBM_TESTS_DYNAMICCONSTRAINTLIFECYCLETESTS_HH

#include "integration/runtime/DynamicConstraintLifecycleTests.H"
#include "integration/runtime/DynamicConstraintGridTests.HH"
#include "integration/runtime/DynamicRegridTests.HH"

#include <limits>

namespace lbm::tests::dynamic_amr::constraint_lifecycle {

struct PeriodicProfile {
    bool third;
    Real zLength;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const
    {
        Real const twoPi = 6.2831853071795864769;
        return {1,0.01*std::sin(twoPi*(third ? p[2]/zLength : p[1])),
                0.02*std::sin(twoPi*p[0]),third ? 0.015*std::sin(twoPi*p[1]) : 0};
    }
};

struct LinearProfile {
    bool third;
    AMREX_GPU_HOST_DEVICE
    initialization::MacroscopicState operator()(amrex::GpuArray<Real,3> const& p, Real) const
    {
        return {1,third ? p[2]/64 : 0,third ? p[0]/64 : p[0]/32,
                third ? p[0]/256+p[2]/128 : 0};
    }
};

struct CheckedRule {
    int* failures;
    int* calls;
    Real indicator;
    template<class Descriptor>
    AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
    {
        amrex::Gpu::Atomic::Add(calls,1);
        if (cell.positionPhys[0] < 0.25 || cell.positionPhys[0] >= 0.75 ||
            cell.rho != cell.rho) {
            amrex::Gpu::Atomic::Add(failures,1);
            return false;
        }
        Real const normalization=Descriptor::d == 3 ? 256*indicator/5 : 32*indicator;
        return amr::VorticityTaggingRule{{20,10},normalization}(cell);
    }
};

template<class Predicate>
void check_collated(amrex::TagBoxArray const& tags, amrex::IntVect n,
                    Predicate wanted, std::string const& label)
{
    amrex::Gpu::PinnedVector<amrex::IntVect> actual;
    tags.collate(actual); // Every rank participates, even if it owns no cells.
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    auto less = [](auto const& a,auto const& b) {
        for (int d=0; d<3; ++d) if (a[d] != b[d]) return a[d] < b[d];
        return false;
    };
    std::sort(actual.begin(),actual.end(),less);
    std::vector<amrex::IntVect> expected;
    for (int i=0; i<n[0]; ++i) for (int j=0; j<n[1]; ++j) for (int k=0; k<n[2]; ++k)
        if (wanted(i,j,k)) expected.emplace_back(i,j,k);
    require(actual.size() == expected.size() && std::equal(actual.begin(),actual.end(),expected.begin()),
            label+": global raw valid-cell tags differ");
}

template<class Descriptor>
void check_raw_tags()
{
    auto spec = constraint_grids::spec_for<Descriptor>(1);
    spec.constraints = {{constraint_grids::rectangle(spec,0,8,0,32),1,1},
                        {constraint_grids::rectangle(spec,24,32,0,32),0,0}};
    auto layout = constraints::prepared(spec,Descriptor::d);
    auto const& geom = layout.geometry;
    auto const n = geom.Domain().length();
    for (bool split : {false,true}) {
        amrex::BoxArray boxes(geom.Domain());
        if (split) boxes.maxSize(amrex::IntVect(16,32,n[2]));
        amrex::Vector<int> owners(boxes.size(),amrex::ParallelDescriptor::NProcs()-1);
        if (split) owners[0] = 0;
        amrex::DistributionMapping mapping(owners);
        LatticeField<Descriptor> populations(boxes,mapping);
        operators::EquilibriumPopulationInitializationOperator<Descriptor,LinearProfile>(
            LinearProfile{Descriptor::d == 3},geom).apply(populations,0);
        amrex::MultiFab velocity(boxes,mapping,3,stencil_extent<Descriptor>());
        amrex::iMultiFab covered(boxes,mapping,1,0);
        auto const dx = geom.CellSizeArray();
        Real const poison = std::numeric_limits<Real>::quiet_NaN();
        for (amrex::MFIter mfi(velocity); mfi.isValid(); ++mfi) {
            auto const u = velocity.array(mfi);
            amrex::ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                Real const x=(i+0.5)*dx[0],z=(k+0.5)*dx[2];
                u(i,j,k,0)=Descriptor::d == 3 ? z/64 : 0;
                u(i,j,k,1)=Descriptor::d == 3 ? x/64 : x/32;
                u(i,j,k,2)=Descriptor::d == 3 ? x/256+z/128 : 0;
            });
            auto const f = populations.multifab().array(mfi);
            auto const mask = covered.array(mfi);
            amrex::ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                mask(i,j,k) = i>=16;
                if (i<8 || i>=24)
                    for (int q=0; q<Descriptor::q; ++q) f(i,j,k,q)=poison;
            });
        }
        for (Real eta : {25.,20.,15.,10.}) {
            amrex::TagBoxArray tags(boxes,mapping,0), optional(boxes,mapping,0);
            tags.setVal(amrex::TagBox::CLEAR);
            optional.setVal(amrex::TagBox::CLEAR);
            amrex::Gpu::DeviceScalar<int> failures(0), calls(0);
            amr::tag_valid_cells<Descriptor>(populations.multifab(),&velocity,covered,geom,0,0,
                CheckedRule{failures.dataPtr(),calls.dataPtr(),eta},tags,&*layout.constraints,&optional);
            int failed=failures.dataValue(), called=calls.dataValue();
            amrex::ParallelDescriptor::ReduceIntSum(failed);
            amrex::ParallelDescriptor::ReduceIntSum(called);
            require(failed==0 && called==16*n[1]*n[2],
                    "geometry short circuit invoked rule in poisoned/required/forbidden cells");
            auto dynamic = [eta](int i,int,int) {
                return i>=8 && i<24 && (eta>20 || (i>=16 && eta>10));
            };
            check_collated(optional,n,dynamic,"optional 20/10 hysteresis");
            check_collated(tags,n,[dynamic](int i,int j,int k) {
                return i<8 || dynamic(i,j,k);
            },"required/optional/forbidden combination");
        }
    }
}

struct CurlRule {
    int* failures;
    int* calls;
    template<class Descriptor>
    AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
    {
        auto const u=cell.velocityLb;
        auto const dx=cell.inverseCellSizePhys;
        auto const p=cell.populations;
        int const i=p.i,j=p.j,k=p.k;
        Real const wz=((u(i+1,j,k,1)-u(i-1,j,k,1))*dx[0]-
                       (u(i,j+1,k,0)-u(i,j-1,k,0))*dx[1])/2;
        bool wrong=std::abs(wz-(Descriptor::d == 3 ? Real(1)/64 : Real(1)/32))>2.e-12;
        if constexpr (Descriptor::d == 3) {
            Real const wx=((u(i,j+1,k,2)-u(i,j-1,k,2))*dx[1]-
                           (u(i,j,k+1,1)-u(i,j,k-1,1))*dx[2])/2;
            Real const wy=((u(i,j,k+1,0)-u(i,j,k-1,0))*dx[2]-
                           (u(i+1,j,k,2)-u(i-1,j,k,2))*dx[0])/2;
            wrong=wrong || std::abs(wx)>2.e-12 || std::abs(wy-Real(3)/256)>2.e-12;
        }
        if (wrong) amrex::Gpu::Atomic::Add(failures,1);
        amrex::Gpu::Atomic::Add(calls,1);
        return amr::VorticityTaggingRule{{19,10},Descriptor::d == 3 ? Real(1024) : Real(640)}(cell);
    }
};

template<class Descriptor>
void check_valid_interface_curl()
{
    auto spec = constraint_grids::spec_for<Descriptor>();
    if constexpr (Descriptor::d == 3) {
        spec.nCell[2]=16;
        spec.probHi[2]=0.5;
    }
    auto layout = constraints::prepared(spec,Descriptor::d);
    auto const coarseGeom = layout.geometry;
    auto const fineGeom = amrex::refine(coarseGeom,spec.refRatio);
    amrex::BoxArray coarseBoxes(coarseGeom.Domain());
    coarseBoxes.maxSize(amrex::IntVect(8,8,Descriptor::d == 3 ? 4 : 1));
    amrex::Box fineRegion(amrex::IntVect(8,8,Descriptor::d == 3 ? 4 : 0),
                          amrex::IntVect(23,23,Descriptor::d == 3 ? 11 : 0));
    fineRegion.refine(spec.refRatio);
    amrex::BoxArray fineBoxes(fineRegion);
    fineBoxes.maxSize(amrex::IntVect(8,8,Descriptor::d == 3 ? 4 : 1));
    LatticeField<Descriptor> coarse(coarseBoxes,amrex::DistributionMapping(coarseBoxes));
    LatticeField<Descriptor> fine(fineBoxes,amrex::DistributionMapping(fineBoxes));
    LinearProfile const profile{Descriptor::d == 3};
    operators::EquilibriumPopulationInitializationOperator<Descriptor,LinearProfile>(profile,coarseGeom).apply(coarse,0);
    operators::EquilibriumPopulationInitializationOperator<Descriptor,LinearProfile>(profile,fineGeom).apply(fine,0);
    amr::SynchronizedVelocity<Descriptor> velocity(
        {{coarseGeom,coarse.multifab()},{fineGeom,fine.multifab()}},spec.refRatio);
    auto host=host_field_snapshot(velocity.level(1));
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const u=host.const_array(mfi);
        for (amrex::BoxIterator cell(mfi.fabbox()); cell.ok(); ++cell) {
            auto const p=cell();
            Real const x=(p[0]+0.5)*fineGeom.CellSize(0),z=(p[2]+0.5)*fineGeom.CellSize(2);
            close(u(p[0],p[1],p[2],0),Descriptor::d == 3 ? z/64 : 0,"valid ux donor/halo");
            close(u(p[0],p[1],p[2],1),Descriptor::d == 3 ? x/64 : x/32,"valid uy donor/halo");
            close(u(p[0],p[1],p[2],2),Descriptor::d == 3 ? x/256+z/128 : 0,"valid uz donor/halo");
        }
    }
    amrex::iMultiFab covered(fineBoxes,fine.multifab().DistributionMap(),1,0);
    covered.setVal(0);
    amrex::TagBoxArray tags(fineBoxes,fine.multifab().DistributionMap(),0);
    amrex::TagBoxArray optional(fineBoxes,fine.multifab().DistributionMap(),0);
    tags.setVal(amrex::TagBox::CLEAR); optional.setVal(amrex::TagBox::CLEAR);
    amrex::Gpu::DeviceScalar<int> failures(0),calls(0);
    amr::tag_valid_cells<Descriptor>(fine.multifab(),&velocity.level(1),covered,fineGeom,1,0,
        CurlRule{failures.dataPtr(),calls.dataPtr()},tags,&*layout.constraints,&optional);
    int failed=failures.dataValue(),called=calls.dataValue();
    amrex::ParallelDescriptor::ReduceIntSum(failed);
    amrex::ParallelDescriptor::ReduceIntSum(called);
    require(failed==0 && called==fineRegion.numPts(),"analytic curl at every valid fine cell including interfaces");
    check_collated(optional,fineGeom.Domain().length(),[fineRegion](int i,int j,int k) {
        return fineRegion.contains(i,j,k);
    },"valid interface optional tags");
}

template<class Descriptor>
void run_tagging_checks()
{
    check_raw_tags<Descriptor>();
    check_valid_interface_curl<Descriptor>();
    amrex::Print()<<"constrained_tagging_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<'\n';
}

struct FirstCallbackTagging {
    static constexpr bool needsVelocity=false;
    std::shared_ptr<std::array<int,3>> calls;
    struct Rule {
        bool first;
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const&) const { return first; }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int lev) const { return {++(*calls)[lev]==1}; }
};

template<class Descriptor>
class InitialCallbackOwner : public amr::LbmAmrSimulation<Descriptor,PeriodicProfile,FirstCallbackTagging> {
    using Base=amr::LbmAmrSimulation<Descriptor,PeriodicProfile,FirstCallbackTagging>;
public:
    std::array<bool,3> created{},alive{},cleared{};
    InitialCallbackOwner(amr::DynamicHierarchyLayout layout, PeriodicProfile profile, FirstCallbackTagging tagging)
        : Base(std::move(layout),profile,std::move(tagging))
    { require(this->initialize({}).empty(),"last-callback initialization"); }
protected:
    void MakeNewLevelFromScratch(int lev,Real time,amrex::BoxArray const& boxes,
                                  amrex::DistributionMapping const& mapping) override
    {
        Base::MakeNewLevelFromScratch(lev,time,boxes,mapping);
        created[lev]=alive[lev]=true;
    }
    void ClearLevel(int lev) override
    {
        Base::ClearLevel(lev);
        alive[lev]=false; cleared[lev]=true;
    }
};

template<class Owner>
void check_initial_view(Owner& owner)
{
    owner.observe(true,[&](auto const& view) {
        require(view.step==0 && view.physicalTime==0 &&
                view.levels.size()==std::size_t(owner.numLevels()),"initial clocks/level publication");
        for (auto const& lev : view.levels) {
            require(lev.populations.multifab().boxArray()==lev.boxes &&
                    lev.populations.multifab().DistributionMap()==lev.mapping &&
                    lev.macro && lev.macro->multifab().boxArray()==lev.boxes &&
                    lev.macro->multifab().DistributionMap()==lev.mapping,
                    "final initial BA/DM/fields differ");
            auto host=host_field_snapshot(lev.macro->multifab());
            for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
                auto const a=host.const_array(mfi);
                for (amrex::BoxIterator p(mfi.validbox()); p.ok(); ++p)
                    for (int c=0; c<host.nComp(); ++c)
                        require(std::isfinite(a(p()[0],p()[1],p()[2],c)),"initial field is not finite");
            }
        }
    });
}

template<class Descriptor>
void run_initialization_checks()
{
    auto spec=constraint_grids::spec_for<Descriptor>(2,16);
    spec.regridInterval=0;
    auto layout=constraints::prepared(spec,Descriptor::d);
    layout.mesh.iterate_on_new_grids=true;
    auto calls=std::make_shared<std::array<int,3>>();
    PeriodicProfile const profile{Descriptor::d == 3,spec.probHi[2]};
    InitialCallbackOwner<Descriptor> owner(layout,profile,FirstCallbackTagging{calls});
    require((*calls)[0]>=2 && (*calls)[1]>=2 && owner.created[1] && owner.created[2],
            "AMReX did not exercise early and late callbacks with an initial L2");
    require(owner.numLevels()==1 && owner.numAllowedLevels()==3 &&
            owner.alive[0] && !owner.alive[1] && !owner.alive[2] &&
            owner.cleared[1] && owner.cleared[2],
            "last empty callbacks must replace early requests and clear rejected tail storage");
    check_initial_view(owner);
    auto prepared=runtime::prepare_amr_evolution(owner,evolution_intent(),BGKDynamicsSelection{});
    require(prepared.ok(),"tail-removal evolution preparation");
    auto evolution=std::move(prepared).take();
    require(owner.evolve(evolution,2,true,[](auto const& view) {
        require(view.levels.size()==1,"cleared tail reappeared during frozen advance");
    }).ok(),"advance after rejected initial tail");

    spec.constraints={{constraint_grids::rectangle(spec,0,16,6,10),1,1}};
    auto fixed=amr::prepare_dynamic_hierarchy<Descriptor>(spec,profile,FullTagging{});
    require(fixed.ok() && fixed.simulation->numLevels()==3,"hard band and outside L2 initialization");
    check_initial_view(*fixed.simulation);
    fixed.simulation->observe(false,[&](auto const& view) {
        constraints::check_cells(view.levels[1].boxes,view.levels[1].geometry.Domain().length(),
            [](auto) { return true; },"full L1 from hard plus optional tags");
        constraints::check_cells(view.levels[2].boxes,view.levels[2].geometry.Domain().length(),
            [](auto p) { return p[1]<24 || p[1]>=40; },"exact fixed-L1 band excludes initial L2");
    });
    amrex::Print()<<"constrained_initialize_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<'\n';
}

template<class Descriptor>
auto constrained_timeline(std::vector<int> const& segments)
{
    auto spec=constraint_grids::spec_for<Descriptor>(2,16);
    spec.regridInterval=2;
    spec.constraints={{constraint_grids::rectangle(spec,0,16,6,10),0,1}};
    Real const dt=0.1/spec.nCell[0];
    auto calls=std::make_shared<std::array<int,3>>();
    auto prepared=amr::prepare_dynamic_hierarchy<Descriptor>(spec,
        PeriodicProfile{Descriptor::d == 3,spec.probHi[2]},TimedTagging{dt,calls});
    require(prepared.ok() && prepared.simulation->numLevels()==1,"constrained timeline start");
    auto owner=prepared.take();
    auto parameters=runtime::prepare_amr_evolution(*owner,evolution_intent(),BGKDynamicsSelection{});
    require(parameters.ok(),"constrained timeline parameters");
    auto evolution=std::move(parameters).take();
    std::array<int,16> const counts{1,2,2,3,3,3,3,3,3,1,1,2,2,3,3,3};
    int observed=0;
    amrex::BoxArray previous;
    for (int count : segments) {
        auto run=owner->evolve(evolution,count,true,[&](auto const& view) {
            ++observed;
            require(view.step==observed && view.levels.size()==std::size_t(counts[observed-1]),
                    "constrained creation/removal/regrowth or segmented regrid cadence");
            close(view.physicalTime,observed*dt,"constrained physical time");
            for (int lev=0; lev<int(view.levels.size()); ++lev) {
                require(view.levels[lev].step==observed*(1<<lev),"new level native clock");
                close(evolution.level(lev).dtPhys,dt/(1<<lev),"allowed-level unit scale");
            }
            if (view.step==6) require(view.levels[1].boxes!=previous,"constrained patch did not move");
            if (view.step==8) require(view.levels[1].boxes.numPts()<previous.numPts(),"constrained patch did not shrink");
            if (view.levels.size()>1) previous=view.levels[1].boxes;
            if (view.levels.size()==3)
                for (int b=0; b<view.levels[2].boxes.size(); ++b) {
                    auto const box=view.levels[2].boxes[b];
                    require(box.bigEnd(1)<24 || box.smallEnd(1)>=40,"regrid entered fixed maximum band");
                }
        });
        require(run.ok(),"constrained timeline evolution: "+run.error);
    }
    require(observed==16,"constrained timeline endpoint");
    return owner;
}

template<class Descriptor>
void run_regrid_checks()
{
    auto continuous=constrained_timeline<Descriptor>({16});
    auto segmented=constrained_timeline<Descriptor>({7,9});
    compare_owners(*continuous,*segmented,true,"constrained continuous/segmented");

    auto spec=constraint_grids::spec_for<Descriptor>(1,16);
    spec.regridInterval=0;
    auto const band=constraint_grids::rectangle(spec,0,16,4,12);
    spec.constraints={{band,1,1}};
    PeriodicProfile const profile{Descriptor::d == 3,spec.probHi[2]};
    auto frozen=amr::prepare_dynamic_hierarchy<Descriptor>(spec,profile,RegionTagging{0});
    amr::StaticHierarchySpec staticSpec;
    static_cast<amr::HierarchyGeometrySpec&>(staticSpec)=spec;
    staticSpec.regions={{1,band}};
    auto fixed=amr::prepare_static_hierarchy<Descriptor>(staticSpec,profile);
    require(frozen.ok() && fixed.ok(),"constrained frozen/static preparation");
    compare_owners(*frozen.simulation,*fixed.simulation,false,"constrained frozen/static initial");
    auto advance=[](auto& owner) {
        auto parameters=runtime::prepare_amr_evolution(owner,evolution_intent(),BGKDynamicsSelection{});
        require(parameters.ok(),"frozen/static parameter preparation");
        auto evolution=std::move(parameters).take();
        require(owner.evolve(evolution,2,true,[](auto const&) {}).ok(),"frozen/static advance");
    };
    advance(*frozen.simulation); advance(*fixed.simulation);
    compare_owners(*frozen.simulation,*fixed.simulation,false,"constrained frozen/static final");
    amrex::Print()<<"constrained_regrid_pass d="<<Descriptor::d
                  <<" ranks="<<amrex::ParallelDescriptor::NProcs()<<'\n';
}
} // namespace lbm::tests::dynamic_amr::constraint_lifecycle
#endif
