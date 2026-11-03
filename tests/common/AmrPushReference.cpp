#include "common/AmrPushReference.H"
#include <limits>

namespace lbm::tests::amr_reference {

Stencil stencil(int dimensions)
{
    if (dimensions == 2) {
        return {{{0,0,0},{-1,1,0},{-1,0,0},{-1,-1,0},{0,-1,0},
                 {1,-1,0},{1,0,0},{1,1,0},{0,1,0}},
                {Real(4)/9,Real(1)/36,Real(1)/9,Real(1)/36,Real(1)/9,
                 Real(1)/36,Real(1)/9,Real(1)/36,Real(1)/9}};
    }
    Stencil result{{{0,0,0},{-1,0,0},{0,-1,0},{0,0,-1},
                    {-1,-1,0},{-1,1,0},{-1,0,-1},{-1,0,1},{0,-1,-1},{0,-1,1},
                    {-1,-1,-1},{-1,-1,1},{-1,1,-1},{-1,1,1},
                    {1,0,0},{0,1,0},{0,0,1},
                    {1,1,0},{1,-1,0},{1,0,1},{1,0,-1},{0,1,1},{0,1,-1},
                    {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1}}, {}};
    for (auto const& c : result.directions) {
        int const norm = c[0]*c[0]+c[1]*c[1]+c[2]*c[2];
        result.weights.push_back(norm == 0 ? Real(8)/27 : norm == 1 ? Real(2)/27 :
                                 norm == 2 ? Real(1)/54 : Real(1)/216);
    }
    return result;
}

VolumeGrid::VolumeGrid(Cell low, Cell size, int populations)
    : low_(low), size_(size), q_(populations),
      data_(static_cast<std::size_t>(cells())*q_)
{}

Cell VolumeGrid::wrap(Cell p) const
{
    for (int d = 0; d < 3; ++d) {
        int shifted = (p[d]-low_[d]) % size_[d];
        if (shifted < 0) shifted += size_[d];
        p[d] = low_[d]+shifted;
    }
    return p;
}

Cell VolumeGrid::cell(int n) const
{
    return {low_[0]+n%size_[0], low_[1]+(n/size_[0])%size_[1],
            low_[2]+n/(size_[0]*size_[1])};
}

int VolumeGrid::index(Cell p, int q) const
{
    p = wrap(p);
    return (((p[2]-low_[2])*size_[1]+p[1]-low_[1])*size_[0]+p[0]-low_[0])*q_+q;
}
Volume& VolumeGrid::at(Cell p, int q) { return data_[index(p,q)]; }
Volume const& VolumeGrid::at(Cell p, int q) const { return data_[index(p,q)]; }

VolumeGrid push(VolumeGrid const& source, Stencil const& stencil)
{
    auto result = source;
    for (int n = 0; n < source.cells(); ++n) {
        auto const p = source.cell(n);
        for (int q = 0; q < source.populations(); ++q) {
            auto destination = p;
            for (int d = 0; d < 3; ++d) destination[d] += stencil.directions[q][d];
            result.at(destination,q) = source.at(p,q);
        }
    }
    return result;
}

Cell coarse_cell(Cell fine, Cell ratio)
{
    for (int d = 0; d < 3; ++d) {
        int const remainder = fine[d] % ratio[d];
        fine[d] = fine[d]/ratio[d] - (remainder < 0 ? 1 : 0);
    }
    return fine;
}

Cell SparseVolumeGrid::wrap(Cell p) const
{
    for (int d=0; d<3; ++d) {
        int relative=(p[d]-low_[d])%size_[d];
        if (relative<0) relative+=size_[d];
        p[d]=low_[d]+relative;
    }
    return p;
}
std::size_t SparseVolumeGrid::key(Cell p) const
{
    p=wrap(p);
    return (std::size_t(p[2]-low_[2])*size_[1]+p[1]-low_[1])*size_[0]+p[0]-low_[0];
}
Volume* SparseVolumeGrid::insert(Cell p)
{
    auto const [it,added]=rows_.emplace(key(p),cells_.size());
    if (added) {
        cells_.push_back(wrap(p));
        data_.resize(cells_.size()*q_,{std::numeric_limits<Real>::quiet_NaN(),3});
    }
    return data_.data()+it->second*q_;
}
Volume* SparseVolumeGrid::find(Cell p)
{
    auto const it=rows_.find(key(p));
    return it==rows_.end() ? nullptr : data_.data()+it->second*q_;
}
Volume const* SparseVolumeGrid::find(Cell p) const
{
    auto const it=rows_.find(key(p));
    return it==rows_.end() ? nullptr : data_.data()+it->second*q_;
}
SparseVolumeGrid push(SparseVolumeGrid const& source, Stencil const& stencil)
{
    SparseVolumeGrid result(source.lower(),source.size(),source.populations());
    for (std::size_t n=0; n<source.cells(); ++n) {
        auto const p=source.cell(n);
        auto const* values=source.row(n);
        for (int q=0; q<source.populations(); ++q) {
            if (values[q].phase==3) continue;
            auto destination=p;
            for (int d=0; d<3; ++d) destination[d]+=stencil.directions[q][d];
            result.insert(destination)[q]=values[q];
        }
    }
    return result;
}

} // namespace lbm::tests::amr_reference
