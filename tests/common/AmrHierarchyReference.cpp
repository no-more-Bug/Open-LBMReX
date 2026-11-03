#include "common/AmrHierarchyReference.H"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lbm::tests::amr_reference {

Real equilibrium(Stencil const& s, int q, Real rho, std::array<Real,3> const& u)
{
    auto const& c=s.directions[q];
    Real const cu=c[0]*u[0]+c[1]*u[1]+c[2]*u[2];
    Real const uu=u[0]*u[0]+u[1]*u[1]+u[2]*u[2];
    return s.weights[q]*rho*(1+3*cu+Real(4.5)*cu*cu-Real(1.5)*uu);
}

HierarchyReference::HierarchyReference(std::vector<ReferenceLevel> levels, int dimensions)
    : levels_(std::move(levels)), stencil_(stencil(dimensions)), ratio_{2,2,dimensions==3?2:1}
{}

SparseVolumeGrid HierarchyReference::fine_source(int lev, SparseVolumeGrid const& collided) const
{
    auto const& fine=levels_[lev+1];
    SparseVolumeGrid result(fine.current.lower(),fine.current.size(),fine.current.populations());
    for (auto const& p : fine.support) {
        auto const c=coarse_cell(p,ratio_);
        if (!levels_[lev].current.find(c)) continue; // Only actual coarse valid sources.
        auto const* values=collided.find(c);
        auto* target=result.insert(p);
        for (int q=0; q<result.populations(); ++q) target[q]={values[q].value,0};
    }
    return result;
}

void HierarchyReference::return_volumes(
    int lev, SparseVolumeGrid const& finePushed, SparseVolumeGrid& coarsePushed) const
{
    auto const& coarse=levels_[lev].current;
    auto const& fine=levels_[lev+1].current;
    int const count=ratio_[0]*ratio_[1]*ratio_[2];
    for (std::size_t n=0; n<coarse.cells(); ++n) {
        auto const c=coarse.cell(n);
        Cell const first{ratio_[0]*c[0],ratio_[1]*c[1],ratio_[2]*c[2]};
        if (fine.find(first)) continue; // Covered coarse auxiliary state is not a physical target.
        for (int q=0; q<coarse.populations(); ++q) {
            Real sum=0;
            bool touched=false, missing=false;
            for (int z=0; z<ratio_[2]; ++z)
            for (int y=0; y<ratio_[1]; ++y)
            for (int x=0; x<ratio_[0]; ++x) {
                auto const* volume=finePushed.find({first[0]+x,first[1]+y,first[2]+z});
                if (!volume || volume[q].phase==3) { missing=true; continue; }
                sum+=volume[q].value;
                touched=touched || volume[q].phase==1 || volume[q].phase==2;
            }
            if (touched) {
                if (missing) throw std::runtime_error("reference support missed a returning subvolume");
                coarsePushed.insert(c)[q].value=sum/count;
            }
        }
    }
}

SparseVolumeGrid HierarchyReference::step(
    int lev, SparseVolumeGrid source, unsigned char finePhase,
    std::vector<ReferenceSample>& samples)
{
    auto& level=levels_[lev];
    for (std::size_t n=0; n<level.current.cells(); ++n) {
        auto const p=level.current.cell(n);
        auto const* before=level.current.row(n);
        Real rho=0;
        std::array<Real,3> u{};
        for (int q=0; q<level.current.populations(); ++q) {
            rho+=before[q].value;
            for (int d=0; d<3; ++d) u[d]+=stencil_.directions[q][d]*before[q].value;
        }
        for (auto& component : u) component/=rho;
        auto* after=source.insert(p);
        for (int q=0; q<level.current.populations(); ++q)
            after[q]={before[q].value+level.omega*(equilibrium(stencil_,q,rho,u)-before[q].value),finePhase};
    }
    auto pushed=push(source,stencil_);
    if (lev+1<int(levels_.size())) {
        auto first=step(lev+1,fine_source(lev,source),1,samples);
        auto const& fine=levels_[lev+1];
        samples.push_back({lev+1,fine.step,levels_[0].step+1,false,fine.time,fine.current});
        auto second=step(lev+1,std::move(first),2,samples);
        samples.push_back({lev+1,fine.step,levels_[0].step+1,true,fine.time,fine.current});
        return_volumes(lev,second,pushed);
    }
    // Functional value replacement, independent of the production P/Q lifecycle.
    SparseVolumeGrid next(level.current.lower(),level.current.size(),level.current.populations());
    for (std::size_t n=0; n<level.current.cells(); ++n) {
        auto const p=level.current.cell(n);
        auto const* values=pushed.find(p);
        if (!values) throw std::runtime_error("reference push missed a valid cell");
        auto* target=next.insert(p);
        for (int q=0; q<level.current.populations(); ++q) {
            if (values[q].phase==3 || !std::isfinite(values[q].value))
                throw std::runtime_error("reference push missed a valid population");
            target[q]=values[q];
        }
    }
    level.current=std::move(next);
    ++level.step;
    level.time+=level.dt;
    return pushed;
}

std::vector<ReferenceSample> HierarchyReference::advance()
{
    std::vector<ReferenceSample> samples;
    step(0,levels_[0].current,0,samples);
    Real const common=levels_[0].time;
    for (auto& level : levels_) level.time=common;
    return samples;
}

} // namespace lbm::tests::amr_reference
