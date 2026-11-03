#include "integration/runtime/AmrDonorEvidence.H"
#include <map>
#include <set>

namespace lbm::tests::amr_evolution {
namespace {
using Cell=amr_reference::Cell;
Cell cell(amrex::IntVect const& p) { return {p[0],p[1],p[2]}; }

struct Owners {
    amr_reference::SparseVolumeGrid coordinates;
    std::map<Cell,int> valid;
    explicit Owners(amr::LbmAmrLevelView<Descriptor> const& level)
        : coordinates(cell(level.geometry.Domain().smallEnd()),cell(level.geometry.Domain().length()),1)
    {
        for (int b=0; b<level.boxes.size(); ++b)
            for (amrex::BoxIterator it(level.boxes[b]); it.ok(); ++it)
                valid.emplace(cell(it()),level.mapping[b]);
    }
    auto find(Cell p) const { return valid.find(coordinates.wrap(p)); }
};

enum Count { Explosion, RemoteExplosion, F1, G1, RemoteF1, RemoteG1,
             RemoteZDirection, RemoteDiagonal, RemotePeriodicZ, ZFace, CountSize };
using Counts=std::array<long long,CountSize>;
}

void write_donor_evidence(EvolutionView const& view, std::string const& sample,
                          std::string const& directory, bool requireRemote)
{
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    std::ofstream summary(directory+"/donors.csv"), witnesses(directory+"/donor-witnesses.csv");
    summary << "coarse_level,explosion_copies,remote_explosion_copies,f1_volumes,g1_volumes,remote_f1_volumes,remote_g1_volumes,remote_z_direction,remote_diagonal,remote_periodic_z,z_face_volumes\n";
    witnesses << "coarse_level,kind,target_i,target_j,target_k,read_i,read_j,read_k,origin_i,origin_j,origin_k,wrapped_origin_i,wrapped_origin_j,wrapped_origin_k,q,offset_x,offset_y,offset_z,source_owner,target_owner\n";
    Counts total{};
    auto const stencil=amr_reference::stencil(Descriptor::d);
    for (int lev=0; lev+1<int(view.levels.size()); ++lev) {
        auto const& fine=view.levels[lev+1];
        Owners const coarseOwners(view.levels[lev]), fineOwners(fine);
        Cell const ratio=cell(view.refRatios[lev]);
        std::set<Cell> covered;
        for (auto const& [p,owner] : fineOwners.valid) {
            (void)owner;
            covered.insert(amr_reference::coarse_cell(p,ratio));
        }
        Counts counts{};
        std::set<std::string> recorded;
        auto witness=[&](std::string const& kind, Cell target, Cell read, Cell origin, Cell wrapped,
                         int q, Cell offset, int sourceOwner, int targetOwner) {
            if (!recorded.insert(kind).second) return;
            witnesses << lev << ',' << kind;
            for (auto const& p : {target,read,origin,wrapped})
                for (int d=0; d<3; ++d) witnesses << ',' << p[d];
            witnesses << ',' << q;
            for (int d=0; d<3; ++d) witnesses << ',' << offset[d];
            witnesses << ',' << sourceOwner << ',' << targetOwner << '\n';
        };
        amrex::IntVect const grow(2,2,Descriptor::d==3 ? 2 : 0);
        for (int b=0; b<fine.boxes.size(); ++b) {
            for (amrex::BoxIterator it(amrex::grow(fine.boxes[b],grow)); it.ok(); ++it) {
                Cell const g=cell(it());
                if (fineOwners.find(g)!=fineOwners.valid.end()) continue;
                Cell const origin=amr_reference::coarse_cell(g,ratio);
                auto const source=coarseOwners.find(origin);
                require(source!=coarseOwners.valid.end(),"explosion has no coarse valid origin");
                ++counts[Explosion];
                bool const remote=source->second!=fine.mapping[b];
                counts[RemoteExplosion]+=remote;
                witness(remote ? "remote_explosion" : "explosion",g,origin,origin,source->first,
                        -1,{},source->second,fine.mapping[b]);
            }
        }
        for (auto const& [target,targetOwner] : coarseOwners.valid) {
            if (covered.count(target)) continue;
            for (int q=0; q<Descriptor::q; ++q) {
                auto const e=stencil.directions[q];
                int const components=(e[0]!=0)+(e[1]!=0)+(e[2]!=0);
                for (int z=0; z<ratio[2]; ++z)
                    for (int y=0; y<ratio[1]; ++y)
                        for (int x=0; x<ratio[0]; ++x) {
                            Cell const offset{x,y,z};
                            Cell s{},r{};
                            for (int d=0; d<3; ++d) {
                                s[d]=target[d]*ratio[d]+offset[d]-e[d];
                                r[d]=s[d]-e[d];
                            }
                            auto source=fineOwners.find(s);
                            bool const f1=source!=fineOwners.valid.end();
                            if (!f1) source=fineOwners.find(r);
                            if (source==fineOwners.valid.end()) continue; // Ordinary Ctmp.
                            Cell const origin=f1 ? s : r;
                            bool const remote=source->second!=targetOwner;
                            bool const periodicZ=origin[2]!=source->first[2];
                            ++counts[f1 ? F1 : G1];
                            counts[f1 ? RemoteF1 : RemoteG1]+=remote;
                            counts[RemoteZDirection]+=remote && e[2]!=0;
                            counts[RemoteDiagonal]+=remote && components>=2;
                            counts[RemotePeriodicZ]+=remote && periodicZ;
                            counts[ZFace]+=components==1 && e[2]!=0;
                            std::string const phase=f1 ? "f1" : "g1";
                            witness((remote ? "remote_" : "")+phase,target,s,origin,source->first,
                                    q,offset,source->second,targetOwner);
                            if (remote && e[2]!=0)
                                witness("remote_z_"+phase,target,s,origin,source->first,q,offset,source->second,targetOwner);
                            if (remote && components>=2)
                                witness("remote_diagonal_"+phase,target,s,origin,source->first,q,offset,source->second,targetOwner);
                            if (remote && periodicZ)
                                witness("remote_periodic_z_"+phase,target,s,origin,source->first,q,offset,source->second,targetOwner);
                        }
            }
        }
        summary << lev;
        for (int n=0; n<CountSize; ++n) { summary << ',' << counts[n]; total[n]+=counts[n]; }
        summary << '\n';
    }
    summary.flush(); witnesses.flush();
    require(summary.good() && witnesses.good(),"cannot write donor evidence");
    if (requireRemote && amrex::ParallelDescriptor::NProcs()>1) {
        require(total[RemoteExplosion]>0 && total[RemoteF1]>0 && total[RemoteG1]>0,
                sample+": expected remote explosion, F1* and G1 origins");
        if constexpr (Descriptor::d==3)
            require(total[RemoteZDirection]>0 && total[RemoteDiagonal]>0 && total[RemotePeriodicZ]>0,
                    sample+": expected remote z/diagonal/periodic-z origins");
    }
    if (sample.find("Through-all")!=std::string::npos)
        require(total[Explosion]==0 && total[F1]==0 && total[G1]==0,"full-fine exchange must be empty");
    if constexpr (Descriptor::d==3)
        if (sample.find("Vertex")!=std::string::npos || sample.find("Wrap3")!=std::string::npos)
            require(total[ZFace]>0,"D3 vertex/wrap case has no actual z-facing interface contribution");
}
} // namespace lbm::tests::amr_evolution
