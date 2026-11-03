#ifndef LBM_AMR_SYNCHRONIZEDVELOCITY_H
#define LBM_AMR_SYNCHRONIZEDVELOCITY_H

#include "lbm/fields/LatticeField.H"

#include <memory>
#include <vector>

namespace lbm::amr {

struct VelocitySourceLevel {
    amrex::Geometry const& geometry;
    amrex::MultiFab const& populations;
};

/** Temporary lattice-velocity donors and halos. Never writes solver populations. */
template<class Descriptor>
class SynchronizedVelocity {
public:
    SynchronizedVelocity(std::vector<VelocitySourceLevel> const& levels,
                         amrex::IntVect const& ratio);
    ~SynchronizedVelocity();
    amrex::MultiFab const& level(int lev) const { return *velocity_[lev]; }

private:
    std::vector<std::unique_ptr<amrex::MultiFab>> velocity_;
};

} // namespace lbm::amr
#include "lbm/amr/SynchronizedVelocity.HH"
#endif
