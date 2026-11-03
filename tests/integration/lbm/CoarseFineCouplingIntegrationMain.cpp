#include "common/AmrPushReference.H"
#include "common/HostFieldSnapshot.H"
#include "lbm/amr/CoarseFineCoupling.H"
#include "lbm/descriptors/D2Q9Descriptor.H"
#include "lbm/descriptors/D3Q27Descriptor.H"
#include "lbm/operators/level/PullStreamingOperator.H"

#include <AMReX.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace {
using namespace lbm;
namespace reference = lbm::tests::amr_reference;
using reference::Cell;
using reference::Real;
using reference::VolumeGrid;
#ifdef LBM_DESCRIPTOR_D3Q27
using Descriptor = D3Q27Descriptor;
#else
using Descriptor = D2Q9Descriptor;
#endif
using Field = LatticeField<Descriptor>;
constexpr Real sentinel = -12345;

void require(bool condition, std::string const& message)
{
    if (!condition) amrex::Abort(message);
}
Cell cell(amrex::IntVect const& p) { return {p[0],p[1],p[2]}; }
amrex::IntVect index(Cell const& p) { return {p[0],p[1],p[2]}; }
bool same(Real a, Real b) { return a == b || (std::isnan(a) && std::isnan(b)); }
Real marker(Cell const& p, int q, bool fine)
{
    int const x=p[0]+8, y=p[1]+8, z=p[2]+8;
    return (fine ? 32 : 4) + Real((q+1)*(1+x+7*y+43*z)+3*x*x+5*y*y+11*z*z+q*x*y)/1024;
}

// Upload a tests-owned valid state while leaving existing ghost phases intact.
void write_valid(Field& field, VolumeGrid const& values)
{
    auto host = lbm::tests::host_field_snapshot(field.multifab());
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const a=host.array(mfi);
        for (amrex::BoxIterator it(mfi.validbox()); it.ok(); ++it)
            for (int q=0; q<Descriptor::q; ++q) a(it(),q)=values.at(cell(it()),q).value;
        amrex::Gpu::htod_memcpy(field.multifab()[mfi].dataPtr(), host[mfi].dataPtr(),
                                host[mfi].size()*sizeof(Real));
    }
}
void unchanged(Field const& field, amrex::MultiFab const& before, std::string const& label)
{
    auto after=lbm::tests::host_field_snapshot(field.multifab());
    for (amrex::MFIter mfi(after); mfi.isValid(); ++mfi)
        for (amrex::BoxIterator it(after[mfi].box()); it.ok(); ++it)
            for (int q=0; q<Descriptor::q; ++q)
                require(same(after.const_array(mfi)(it(),q),before.const_array(mfi)(it(),q)),label);
}
void check_region(Field const& field, VolumeGrid const& expected, amrex::IntVect const& grow,
                  bool checkOuter, std::string const& label)
{
    auto host=lbm::tests::host_field_snapshot(field.multifab());
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        auto const region=amrex::grow(mfi.validbox(),grow);
        auto const box=checkOuter ? host[mfi].box() : region;
        for (amrex::BoxIterator it(box); it.ok(); ++it)
            for (int q=0; q<Descriptor::q; ++q) {
                auto const want=region.contains(it()) ? expected.at(cell(it()),q).value : sentinel;
                require(same(host.const_array(mfi)(it(),q),want),label);
            }
    }
}

amrex::BoxArray coverage(amrex::Box const& domain, int kind)
{
    int constexpr octants=1<<Descriptor::d;
    amrex::BoxList boxes;
    if (kind<octants) {
        auto lo=domain.smallEnd(), hi=domain.bigEnd();
        for (int d=0; d<Descriptor::d; ++d) {
            if ((kind>>d)&1) lo[d]=0;
            else hi[d]=-1;
        }
        boxes.push_back(amrex::Box(lo,hi));
    } else if (kind<octants+Descriptor::d) {
        auto box=domain;
        box.setBig(kind-octants,-1);
        boxes.push_back(box);
    } else if (kind==octants+Descriptor::d) {
        auto lo=domain.smallEnd(), hi=domain.bigEnd();
        for (int d=0; d<Descriptor::d; ++d) { lo[d]=-1; hi[d]=0; }
        boxes=amrex::BoxArray(amrex::Box(lo,hi)).complementIn(domain);
    } else if (kind==octants+Descriptor::d+1) {
        for (int start : {-4,0}) {
            auto box=domain;
            box.setSmall(0,start); box.setBig(0,start+1);
            boxes.push_back(box);
        }
    } else {
        boxes.push_back(domain);
    }
    return amrex::BoxArray(std::move(boxes));
}
amrex::DistributionMapping owners(amrex::BoxArray const& boxes, int offset)
{
    amrex::Vector<int> ranks(boxes.size());
    for (int b=0; b<boxes.size(); ++b) ranks[b]=(b+offset)%amrex::ParallelDescriptor::NProcs();
    return amrex::DistributionMapping(std::move(ranks));
}
struct Counts {
    long oneFineRestGhost=0, oneFineRestCoarse=0, complete=0, partial=0, remoteExplosion=0;
};

void run_case(int kind, bool poison, reference::Stencil const& stencil,
              std::ofstream& ledger, Counts& counts)
{
    constexpr int dim=Descriptor::d;
    Cell const ratio{2,2,dim==3?2:1};
    Cell const clo{-4,-4,dim==3?-4:0}, size{8,8,dim==3?8:1};
    Cell const flo{-8,-8,dim==3?-8:0}, fsize{16,16,dim==3?16:1};
    amrex::Box const domain(index(clo),amrex::IntVect(3,3,dim==3?3:0));
    amrex::IntVect const r=index(ratio), v=stencil_extent<Descriptor>();
    amrex::RealBox physical({0.,0.,0.},{1.,1.,dim==3?1.:1./16});
    amrex::Array<int,3> periodic{1,1,1};
    amrex::Geometry const cgeom(domain,physical,0,periodic);
    amrex::Geometry const fgeom(amrex::refine(domain,r),physical,0,periodic);
    auto const covered=coverage(domain,kind);
    amrex::BoxArray coarseBoxes(domain); coarseBoxes.maxSize(4);
    auto fineBoxes=covered; fineBoxes.refine(r); fineBoxes.maxSize(4);
    auto const coarseOwners=owners(coarseBoxes,0), fineOwners=owners(fineBoxes,1);
    Field coarseStar(coarseBoxes,coarseOwners,v), coarseNext(coarseBoxes,coarseOwners,v);
    Field fineFirst(fineBoxes,fineOwners,2*v), fineNext(fineBoxes,fineOwners,2*v);
    amr::CoarseFineCoupling<Descriptor> coupling(cgeom,coarseStar,fgeom,fineFirst,r);

    VolumeGrid coarse(clo,size,Descriptor::q), initial(flo,fsize,Descriptor::q);
    for (int n=0; n<coarse.cells(); ++n) {
        auto const p=coarse.cell(n);
        for (int q=0; q<Descriptor::q; ++q) coarse.at(p,q).value=marker(p,q,false);
    }
    for (int n=0; n<initial.cells(); ++n) {
        auto const p=initial.cell(n), c=reference::coarse_cell(p,ratio);
        bool const fine=covered.contains(index(c));
        for (int q=0; q<Descriptor::q; ++q)
            initial.at(p,q)={fine?marker(p,q,true):coarse.at(c,q).value,
                             static_cast<unsigned char>(fine?1:0)};
    }
    auto const first=reference::push(initial,stencil);
    auto collided=first;
    for (int n=0; n<collided.cells(); ++n) {
        auto const p=collided.cell(n);
        if (!covered.contains(index(reference::coarse_cell(p,ratio)))) continue;
        for (int q=0; q<Descriptor::q; ++q) {
            collided.at(p,q).value+=Real(q+1)/4+Real(p[0]*p[0]+5*p[1]+p[2]*p[2])/256;
            collided.at(p,q).phase=2;
        }
    }
    auto const second=reference::push(collided,stencil);
    auto coarseInput=coarse;
    if (poison) {
        for (int n=0; n<coarseInput.cells(); ++n) {
            auto const p=coarseInput.cell(n);
            if (covered.contains(index(p)))
                for (int q=0; q<Descriptor::q; ++q)
                    coarseInput.at(p,q).value=std::numeric_limits<Real>::quiet_NaN();
        }
    }
    coarseStar.multifab().setVal(sentinel); coarseNext.multifab().setVal(sentinel);
    fineFirst.multifab().setVal(sentinel); fineNext.multifab().setVal(sentinel);
    write_valid(coarseStar,coarseInput); write_valid(fineFirst,initial);
    coarseStar.fillBoundary(cgeom,v);
    auto const coarseBefore=lbm::tests::host_field_snapshot(coarseStar.multifab());
    operators::PullStreamingOperator<Descriptor> pull;
    pull.apply(coarseStar,coarseNext);
    fineFirst.fillBoundary(fgeom,2*v);
    coupling.explode(coarseStar,fineFirst);
    check_region(fineFirst,initial,2*v,false,"C3 explosion/first-fill source mismatch");
    unchanged(coarseStar,coarseBefore,"C3 explosion modified coarse source");

    // These ghost values were checked above; count genuinely remote coarse donors.
    for (amrex::MFIter mfi(fineFirst.multifab()); mfi.isValid(); ++mfi) {
        for (amrex::BoxIterator it(amrex::grow(mfi.validbox(),2*v)); it.ok(); ++it) {
            auto const c=reference::coarse_cell(initial.wrap(cell(it())),ratio);
            if (covered.contains(index(c))) continue;
            auto const hits=coarseBoxes.intersections(amrex::Box(index(c),index(c)));
            if (coarseOwners[hits.front().first]!=amrex::ParallelDescriptor::MyProc())
                ++counts.remoteExplosion;
        }
    }
    auto const fineBefore=lbm::tests::host_field_snapshot(fineFirst.multifab());
    pull.apply(fineFirst,fineNext,v);
    unchanged(fineFirst,fineBefore,"C3 first pull modified its source");
    check_region(fineNext,first,v,true,"C3 first pull/G1/outer preservation mismatch");
    write_valid(fineNext,collided);
    fineNext.fillBoundary(fgeom,v);
    check_region(fineNext,collided,v,true,"C3 second fill overwrote G1 or missed F1*");
    pull.apply(fineNext,fineFirst);
    check_region(fineFirst,second,amrex::IntVect(0),false,"C3 second fine result mismatch");

    if (poison) {
        // F2 is not an input to coalesce; only the valid+v ring of Q may be exported.
        fineFirst.multifab().setVal(std::numeric_limits<Real>::quiet_NaN());
        auto host=lbm::tests::host_field_snapshot(fineNext.multifab());
        for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
            auto const inner=amrex::grow(mfi.validbox(),v);
            auto const a=host.array(mfi);
            for (amrex::BoxIterator it(host[mfi].box()); it.ok(); ++it)
                if (!inner.contains(it()))
                    for (int q=0; q<Descriptor::q; ++q) a(it(),q)=std::numeric_limits<Real>::quiet_NaN();
            amrex::Gpu::htod_memcpy(fineNext.multifab()[mfi].dataPtr(),host[mfi].dataPtr(),
                                    host[mfi].size()*sizeof(Real));
        }
    }
    auto const sourceBefore=lbm::tests::host_field_snapshot(fineNext.multifab());
    auto const coarseTmp=lbm::tests::host_field_snapshot(coarseNext.multifab());
    coupling.coalesce(fineNext,coarseNext);
    unchanged(fineNext,sourceBefore,"C3 coalescence modified F1*/G1 source");
    auto const actual=lbm::tests::host_field_snapshot(coarseNext.multifab());
    int const volumes=ratio[0]*ratio[1]*ratio[2];
    for (amrex::MFIter mfi(actual); mfi.isValid(); ++mfi) {
        auto const a=actual.const_array(mfi), before=coarseTmp.const_array(mfi);
        for (amrex::BoxIterator it(actual[mfi].box()); it.ok(); ++it) {
            auto const c=cell(it());
            for (int q=0; q<Descriptor::q; ++q) {
                if (!mfi.validbox().contains(it()) || covered.contains(it())) {
                    require(same(a(it(),q),before(it(),q)),"C3 changed a ghost or covered coarse cell");
                    continue;
                }
                std::array<int,3> phases{};
                std::string phaseCodes;
                Real expected=0;
                for (int z=0; z<ratio[2]; ++z)
                for (int y=0; y<ratio[1]; ++y)
                for (int x=0; x<ratio[0]; ++x) {
                    auto const volume=second.at({ratio[0]*c[0]+x,ratio[1]*c[1]+y,ratio[2]*c[2]+z},q);
                    expected+=volume.value/volumes;
                    ++phases[volume.phase]; phaseCodes+=char('0'+volume.phase);
                }
                require(std::isfinite(a(it(),q)) && a(it(),q)==expected,
                        "C3 push-volume/coalescence mismatch case="+std::to_string(kind)+
                        " q="+std::to_string(q)+" cell="+std::to_string(c[0])+","+
                        std::to_string(c[1])+","+std::to_string(c[2]));
                if (phases[0]==volumes) continue;
                counts.oneFineRestGhost+=(phases[2]==1 && phases[1]==volumes-1);
                counts.oneFineRestCoarse+=(phases[2]==1 && phases[0]==volumes-1);
                counts.complete+=(phases[0]==0); counts.partial+=(phases[0]>0);
                ledger<<kind<<','<<poison<<','<<c[0]<<','<<c[1]<<','<<c[2]<<','<<q<<','
                      <<phaseCodes<<','<<phases[2]<<','<<phases[1]<<','<<phases[0]<<','
                      <<expected<<','<<a(it(),q)<<'\n';
            }
        }
    }
    amrex::Print()<<"exchange_case="<<kind<<" poison="<<poison<<" status=pass\n";
}
} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        auto const stencil=reference::stencil(Descriptor::d);
        for (int q=0; q<Descriptor::q; ++q)
            for (int d=0; d<3; ++d)
                require(stencil.directions[q][d]==Descriptor::c(q,d),"C3 independent direction identity mismatch");
        std::ofstream ledger("exchange-ledger-rank"+std::to_string(amrex::ParallelDescriptor::MyProc())+".csv");
        require(ledger.good(),"C3 cannot create exchange ledger");
        ledger<<std::setprecision(std::numeric_limits<Real>::max_digits10);
        ledger<<"case,poison,i,j,k,q,volume_phases,f1star,g1,ctmp,expected,actual\n";
        Counts counts;
        for (int kind=0; kind<(1<<Descriptor::d)+Descriptor::d+3; ++kind)
            for (bool poison : {false,true}) run_case(kind,poison,stencil,ledger,counts);
        amrex::ParallelDescriptor::ReduceLongSum(counts.oneFineRestGhost);
        amrex::ParallelDescriptor::ReduceLongSum(counts.oneFineRestCoarse);
        amrex::ParallelDescriptor::ReduceLongSum(counts.complete);
        amrex::ParallelDescriptor::ReduceLongSum(counts.partial);
        amrex::ParallelDescriptor::ReduceLongSum(counts.remoteExplosion);
        require(counts.oneFineRestGhost>0 && counts.oneFineRestCoarse>0 &&
                counts.complete>0 && counts.partial>0,"C3 missing required volume paths");
        if (amrex::ParallelDescriptor::NProcs()>1)
            require(counts.remoteExplosion>0,"C3 no actual remote coarse donor was checked");
        require(ledger.good(),"C3 failed writing exchange ledger");
        amrex::Print()<<"exchange_status=pass dimensions="<<Descriptor::d
            <<" one_fine_rest_g1="<<counts.oneFineRestGhost
            <<" one_fine_rest_ctmp="<<counts.oneFineRestCoarse
            <<" complete="<<counts.complete<<" partial="<<counts.partial
            <<" remote_explosion="<<counts.remoteExplosion<<'\n';
    }
    amrex::Finalize();
}
