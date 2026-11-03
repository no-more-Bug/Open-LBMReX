#ifndef LBM_TESTS_POPULATIONREMAPTESTS_HH
#define LBM_TESTS_POPULATIONREMAPTESTS_HH

#include "integration/runtime/DynamicAmrTestSupport.H"
#include "common/HostFieldSnapshot.H"
#include "lbm/amr/PopulationRemap.H"

#include <AMReX_BoxIterator.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <array>
#include <memory>
#include <vector>

namespace lbm::tests::dynamic_amr::remap {

AMREX_GPU_HOST_DEVICE inline Real encoded(int lev, int i, int j, int k, int q)
{
    // Deliberately not an equilibrium distribution; every direction carries distinct data.
    return 1+0.1*lev+0.01*q+0.001*(q+1)*i+0.0003*j+0.00007*k;
}

template<class Descriptor>
amrex::Box region(int lo, int hi)
{
    return amrex::Box(amrex::IntVect(lo,lo,Descriptor::d == 3 ? lo : 0),
                      amrex::IntVect(hi,hi,Descriptor::d == 3 ? hi : 0));
}

template<class Descriptor>
amrex::IntVect child(amrex::IntVect const& p, int x = 0, int y = 0, int z = 0)
{
    return amrex::IntVect(2*p[0]+x,2*p[1]+y,Descriptor::d == 3 ? 2*p[2]+z : 0);
}

template<class Descriptor>
bool covered(std::vector<amrex::Box> const& regions, int lev, amrex::IntVect const& p)
{
    return lev+1 < static_cast<int>(regions.size()) && regions[lev+1].contains(child<Descriptor>(p));
}

// Host oracle enumerates descendants directly, independently of AMReX masks/transfers.
template<class Descriptor>
Real old_composite(std::vector<amrex::Box> const& oldRegions, int lev,
                   amrex::IntVect const& p, int q)
{
    if (!covered<Descriptor>(oldRegions,lev,p)) return encoded(lev,p[0],p[1],p[2],q);
    Real sum = 0;
    for (int z = 0; z < (Descriptor::d == 3 ? 2 : 1); ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                sum += old_composite<Descriptor>(oldRegions,lev+1,child<Descriptor>(p,x,y,z),q);
    return sum/(Descriptor::d == 3 ? 8 : 4);
}

template<class Descriptor>
Real expected(std::vector<amrex::Box> const& oldRegions,
              std::vector<amrex::Box> const& newRegions, int lev,
              amrex::IntVect const& p, int q)
{
    if (lev < static_cast<int>(oldRegions.size()) && oldRegions[lev].contains(p)) {
        if (covered<Descriptor>(oldRegions,lev,p) && !covered<Descriptor>(newRegions,lev,p))
            return old_composite<Descriptor>(oldRegions,lev,p,q);
        return encoded(lev,p[0],p[1],p[2],q);
    }
    require(lev > 0,"new L0 cells are outside the fixed domain");
    return expected<Descriptor>(oldRegions,newRegions,lev-1,
        amrex::IntVect(p[0]/2,p[1]/2,Descriptor::d == 3 ? p[2]/2 : 0),q);
}

template<class Descriptor>
std::unique_ptr<amrex::MultiFab> field(amrex::Box const& area, int maxSize, int ownerShift)
{
    amrex::BoxArray boxes(area);
    boxes.maxSize(amrex::IntVect(maxSize,maxSize,Descriptor::d == 3 ? maxSize : 1));
    amrex::Vector<int> owners(boxes.size());
    for (int b = 0; b < boxes.size(); ++b)
        owners[b] = (b+ownerShift)%amrex::ParallelDescriptor::NProcs();
    auto result = std::make_unique<amrex::MultiFab>(
        boxes,amrex::DistributionMapping(owners),Descriptor::q,1);
    result->setVal(-777);
    return result;
}

template<class Descriptor>
void encode(amrex::MultiFab& values, int lev)
{
    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi) {
        auto const out = values.array(mfi);
        amrex::ParallelFor(mfi.validbox(),Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i,int j,int k,int q) {
                out(i,j,k,q)=encoded(lev,i,j,k,q);
            });
    }
}

template<class Descriptor>
std::array<Real,4> balance(std::vector<std::unique_ptr<amrex::MultiFab>> const& fields,
                          std::vector<amrex::Box> const& regions)
{
    std::array<long double,4> local{};
    for (int lev = 0; lev < static_cast<int>(fields.size()); ++lev) {
        auto host = host_field_snapshot(*fields[lev]);
        Real const spacing = 1.0/(8*(1<<lev));
        Real const volume = spacing*spacing*(Descriptor::d == 3 ? spacing : 0.125);
        for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const f = host.const_array(mfi);
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const& p = cell();
                if (covered<Descriptor>(regions,lev,p)) continue;
                for (int q = 0; q < Descriptor::q; ++q) {
                    auto const contribution = static_cast<long double>(f(p[0],p[1],p[2],q))*volume;
                    local[0] += contribution;
                    for (int d = 0; d < 3; ++d) local[d+1] += contribution*Descriptor::c(q,d);
                }
            }
        }
    }
    std::array<Real,4> result{};
    for (int d = 0; d < 4; ++d) result[d] = static_cast<Real>(local[d]);
    amrex::ParallelDescriptor::ReduceRealSum(result.data(),4);
    return result;
}

template<class Descriptor>
void scenario(std::string const& name, std::vector<amrex::Box> const& oldRegions,
              std::vector<amrex::Box> const& newRegions, int newMaxSize, bool retainCoarse)
{
    std::vector<std::unique_ptr<amrex::MultiFab>> oldFields;
    std::vector<amrex::MultiFab const*> oldViews;
    for (int lev = 0; lev < static_cast<int>(oldRegions.size()); ++lev) {
        oldFields.push_back(field<Descriptor>(oldRegions[lev],4,lev));
        encode<Descriptor>(*oldFields.back(),lev);
        oldViews.push_back(oldFields.back().get());
    }
    auto const before = balance<Descriptor>(oldFields,oldRegions);
    amr::PopulationRemap<Descriptor> transfer(
        oldViews,amrex::IntVect(2,2,Descriptor::d == 3 ? 2 : 1));
    std::vector<std::unique_ptr<amrex::MultiFab>> newFields;
    for (int lev = 0; lev < static_cast<int>(newRegions.size()); ++lev) {
        newFields.push_back(field<Descriptor>(newRegions[lev],newMaxSize,lev+1));
    }
    if (retainCoarse) newFields[0] = std::move(oldFields[0]);
    if (name == "owners" && amrex::ParallelDescriptor::NProcs() > 1) {
        require(newFields[1]->boxArray() == oldFields[1]->boxArray(),"owner test changed boxes");
        require(newFields[1]->DistributionMap() != oldFields[1]->DistributionMap(),
                "owner test did not change ownership");
    }
    for (int lev = 0; lev < static_cast<int>(newFields.size()); ++lev) {
        auto const nextBoxes = lev+1 < static_cast<int>(newFields.size())
            ? newFields[lev+1]->boxArray() : amrex::BoxArray{};
        if (lev == 0 && retainCoarse)
            transfer.backfillUncovered(lev,*newFields[lev],nextBoxes);
        else
            transfer.fillNewLevel(lev,*newFields[lev],lev > 0 ? newFields[lev-1].get() : nullptr,nextBoxes);
        auto host = host_field_snapshot(*newFields[lev]);
        for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const f = host.const_array(mfi);
            for (amrex::BoxIterator cell(mfi.validbox()); cell.ok(); ++cell) {
                auto const& p = cell();
                for (int q = 0; q < Descriptor::q; ++q)
                    close(f(p[0],p[1],p[2],q),expected<Descriptor>(oldRegions,newRegions,lev,p,q),
                          "full population oracle: "+name);
            }
        }
    }
    auto const after = balance<Descriptor>(newFields,newRegions);
    for (int d = 0; d < 4; ++d)
        require(std::isfinite(after[d]) && std::abs(after[d]-before[d]) < 2.e-11,
                "composite mass/momentum: "+name);
    amrex::Print() << "population_remap_pass case=" << name << " d=" << Descriptor::d
                   << " ranks=" << amrex::ParallelDescriptor::NProcs()
                   << " mass_delta=" << after[0]-before[0] << '\n';
}

template<class Descriptor>
void run_checks()
{
    std::vector<amrex::Box> const oldRegions{
        region<Descriptor>(0,7),region<Descriptor>(4,11),region<Descriptor>(10,17)};
    scenario<Descriptor>("coarsen_all",oldRegions,{oldRegions[0]},4,true);
    scenario<Descriptor>("move_shrink_expand",oldRegions,
        {oldRegions[0],region<Descriptor>(2,9),region<Descriptor>(8,11)},4,true);
    scenario<Descriptor>("reblock",oldRegions,oldRegions,8,false);
    scenario<Descriptor>("owners",oldRegions,oldRegions,4,false);
    scenario<Descriptor>("add_levels",{oldRegions[0]},oldRegions,8,true);
}
} // namespace lbm::tests::dynamic_amr::remap
#endif
