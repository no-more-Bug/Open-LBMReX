#ifndef LBM_TESTS_DYNAMICBOUNDARYTESTSUPPORT_H
#define LBM_TESTS_DYNAMICBOUNDARYTESTSUPPORT_H

#include "integration/runtime/DynamicBoundaryTests.H"
#include "integration/runtime/DynamicRegridTests.HH"
#include "integration/runtime/AmrBoundaryForceTests.H"

namespace lbm::tests::dynamic_amr::dynamic_boundary {
namespace reference = ::lbm::tests::integration::amr_boundary_test;
using Velocity = reference::Velocity;

template<class Descriptor>
amr::DynamicHierarchySpec boundary_spec(bool withSolids)
{
    auto fixed=reference::evolution_spec<Descriptor>(3,true);
    amr::DynamicHierarchySpec spec;
    static_cast<amr::HierarchyGeometrySpec&>(spec)=fixed;
    spec.maxLevel=2;
    spec.regridInterval=1;
    spec.coarseNestingMarginCells=2;
    spec.tagBufferCells={amrex::IntVect(0)};
    auto const z=spec.probHi[2];
    spec.constraints={{{{0,0,0},{1,16.25,z}},0,0},
                      {{{15.25,0,0},{16.25,16.25,z}},0,0},
                      {{{0,0,0},{16.25,1,z}},0,0},
                      {{{0,15.25,0},{16.25,16.25,z}},0,0}};
    if (withSolids) {
        spec.constraints.push_back({{{2,2,0},{14,14,z}},1,2});
        spec.constraints.push_back({{{2,2,0},{8,6,z}},1,1});
        spec.constraints.push_back({{{8,8,0},{12,12,z}},2,2});
    }
    return spec;
}

inline amr::AmrBoundarySpec boundaries(int grouping, bool withSolids=true)
{
    auto spec=reference::evolution_boundaries(3,true);
    spec.collectForces=true;
    spec.internal.clear();
    if (withSolids) {
        spec.internal={reference::cylinder("L1_a",3.0625,4.0625,0.5),
                       reference::cylinder("L1_b",6.0625,4.0625,0.5),
                       reference::cylinder("L2",10.03125,10.03125,0.25)};
        spec.internal[1].scheme=spec.internal[2].scheme=boundary::BoundaryScheme::Bouzidi;
        for (int b=0; b<3; ++b)
            spec.internal[b].objectName=grouping==1 ? (b<2 ? "L1_pair" : "L2_one") :
                (b==1 ? "L1_only" : "across");
    }
    return spec;
}

inline config::LbmConfigIntent boundary_intent()
{
    auto intent=evolution_intent();
    intent.uRefLb=0.04; intent.uRefPhys=2;
    return intent;
}

struct ExternalPatchTagging {
    static constexpr bool needsVelocity=false;
    struct Rule {
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            int const step=int(cell.physicalTime/0.005+0.5);
            return cell.level==1 && (step==1 || step==2) &&
                cell.positionPhys[0]>=3 && cell.positionPhys[0]<6 &&
                cell.positionPhys[1]>=9 && cell.positionPhys[1]<12;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {}; }
};

struct DomainPatchTagging {
    static constexpr bool needsVelocity=false;
    struct Rule {
        template<class Descriptor>
        AMREX_GPU_HOST_DEVICE bool operator()(amr::TaggingCell<Descriptor> const& cell) const
        {
            int const step=int(cell.physicalTime/0.005+0.5);
            if (step==0 || step==4) return false;
            Real const lo=step==3 ? 5 : 4;
            return cell.positionPhys[0]>=lo && cell.positionPhys[0]<lo+8 &&
                cell.positionPhys[1]>=4 && cell.positionPhys[1]<12;
        }
    };
    std::vector<std::string> validate(int) const { return {}; }
    Rule rule(int) const { return {}; }
};

template<class Owner>
void check_owners(Owner const& owner)
{
    require(owner.numLevels()==3,"protected boundaries require three active levels");
    for (int b=0; b<3; ++b)
        require(owner.internalBoundaryLevel({4+b})==(b<2 ? 1 : 2),
                "BoundaryId owner changed across regrid");
}

inline void append_step(std::vector<Real>& trace, boundary::InternalForceStep const& step)
{
    trace.insert(trace.end(),{Real(step.level),Real(step.populationStep),Real(step.coarsestStep),
                             step.beginPhysicalTime,step.endPhysicalTime});
}

// A compact value trace is only for two-run equality, not a force oracle.
inline void append_force_trace(std::vector<Real>& trace, boundary::InternalForceBatch const& batch)
{
    trace.insert(trace.end(),{Real(batch.boundarySteps.size()),Real(batch.objectSteps.size()),
                             Real(batch.objectIntervals.size())});
    for (auto const& row : batch.boundarySteps) {
        append_step(trace,row.step);
        trace.push_back(row.boundaryForces.size());
        for (auto const& force : row.boundaryForces) {
            trace.push_back(force.boundaryId.value);
            trace.insert(trace.end(),force.forceLb.begin(),force.forceLb.end());
        }
    }
    for (auto const& row : batch.objectSteps) {
        append_step(trace,row.step);
        trace.push_back(row.objectForces.size());
        for (auto const& force : row.objectForces) {
            trace.push_back(force.objectId.value);
            trace.insert(trace.end(),force.forceLb.begin(),force.forceLb.end());
        }
    }
    for (auto const& row : batch.objectIntervals) {
        trace.insert(trace.end(),{Real(row.coarsestStep),row.beginPhysicalTime,row.endPhysicalTime,
                                 Real(row.objectForces.size())});
        for (auto const& force : row.objectForces) {
            trace.push_back(force.objectId.value);
            trace.insert(trace.end(),force.forceLb.begin(),force.forceLb.end());
        }
    }
}

inline void compare_traces(std::vector<Real> const& a,std::vector<Real> const& b)
{
    require(a.size()==b.size(),"force trace sizes differ");
    for (std::size_t i=0;i<a.size();++i)
        reference::force_close(a[i],b[i],"full native/Object/interval force trace");
}

template<class Descriptor,class Profile,class Tagging>
class LayoutOwner : public amr::LbmAmrSimulation<Descriptor,Profile,Tagging> {
    using Base=amr::LbmAmrSimulation<Descriptor,Profile,Tagging>;
public:
    LayoutOwner(amr::DynamicHierarchyLayout layout,Profile profile,Tagging tagging,
                 amr::AmrBoundarySpec const& spec)
        : Base(std::move(layout),profile,tagging)
    { require(this->initialize(spec).empty(),"dynamic boundary owner initialization"); }

    void relayout(int mode)
    {
        std::vector<amrex::BoxArray> boxes;
        std::vector<amrex::DistributionMapping> maps;
        Real time=0;
        this->observe(false,[&](auto const& view) {
            time=view.physicalTime;
            for (int lev=0;lev<int(view.levels.size());++lev) {
                auto const& level=view.levels[lev];
                boxes.push_back(level.boxes); maps.push_back(level.mapping);
                if (!lev || mode==0) continue;
                if (mode==1) {
                    boxes.back().maxSize(amrex::IntVect(8,8,Descriptor::d==3 ? 4 : 1));
                    require(boxes.back()!=level.boxes,"relayout fixture must change BA");
                }
                amrex::Vector<int> owners(boxes.back().size(),0);
                if (mode==1)
                    for (int b=0;b<boxes.back().size();++b)
                        owners[b]=(b+1)%amrex::ParallelDescriptor::NProcs();
                maps.back()=amrex::DistributionMapping(owners);
            }
        });
        this->installHierarchy(boxes,maps,time);
    }
};
} // namespace lbm::tests::dynamic_amr::dynamic_boundary
#endif
