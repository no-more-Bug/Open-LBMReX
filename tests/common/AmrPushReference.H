#ifndef LBM_TESTS_AMRPUSHREFERENCE_H
#define LBM_TESTS_AMRPUSHREFERENCE_H

#include <AMReX_REAL.H>
#include <array>
#include <vector>
#include <unordered_map>

namespace lbm::tests::amr_reference {

using Cell = std::array<int,3>;
using Real = amrex::Real;

// Independent, literal direction/weight tables. No production Descriptor,
// coupling mask, gather classifier or streaming operator enters this oracle.
struct Stencil {
    std::vector<Cell> directions;
    std::vector<Real> weights;
};
Stencil stencil(int dimensions);

// A small periodic reference lattice; phases travel forward with each volume.
// 0 = ordinary coarse volume, 1 = first fine outflow (G1), 2 = F1*.
struct Volume { Real value = 0; unsigned char phase = 0; };
class VolumeGrid {
public:
    VolumeGrid(Cell low, Cell size, int populations);
    Cell wrap(Cell p) const;
    Cell cell(int index) const;
    int cells() const { return size_[0]*size_[1]*size_[2]; }
    int populations() const { return q_; }
    Volume& at(Cell p, int q);
    Volume const& at(Cell p, int q) const;
private:
    int index(Cell p, int q) const;
    Cell low_, size_;
    int q_;
    std::vector<Volume> data_;
};
VolumeGrid push(VolumeGrid const& source, Stencil const& stencil);
Cell coarse_cell(Cell fine, Cell ratio);

// Compact support for hierarchy tests. Absent populations carry phase 3;
// no full finest-domain population array is allocated.
class SparseVolumeGrid {
public:
    SparseVolumeGrid(Cell low, Cell size, int populations) : low_(low), size_(size), q_(populations) {}
    Cell wrap(Cell p) const;
    Cell lower() const { return low_; }
    Cell size() const { return size_; }
    int populations() const { return q_; }
    std::size_t cells() const { return cells_.size(); }
    Cell cell(std::size_t row) const { return cells_[row]; }
    Volume const* row(std::size_t n) const { return data_.data()+n*q_; }
    Volume* insert(Cell p);
    Volume* find(Cell p);
    Volume const* find(Cell p) const;
private:
    std::size_t key(Cell p) const;
    Cell low_, size_;
    int q_;
    std::vector<Cell> cells_;
    std::vector<Volume> data_;
    std::unordered_map<std::size_t,std::size_t> rows_;
};
SparseVolumeGrid push(SparseVolumeGrid const& source, Stencil const& stencil);

} // namespace lbm::tests::amr_reference
#endif
