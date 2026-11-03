#ifndef LBM_TESTS_NUMERICAL_CAVITYOBSERVATION_H
#define LBM_TESTS_NUMERICAL_CAVITYOBSERVATION_H

#include "CavityCase.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include <array>
#include <fstream>

namespace amr_cavity {

// Own the comparison endpoint; published macros are borrowed and later overwritten.
template<class Descriptor>
class CavityObservation {
public:
    explicit CavityObservation(std::string const& directory);
    Real initialize(lbm::amr::LbmAmrView<Descriptor> const& view);
    void save(lbm::amr::LbmAmrView<Descriptor> const& view);
    void record(lbm::amr::LbmAmrView<Descriptor> const& view);
private:
    // mass, squared velocity change, squared current velocity, and three D3 maxima.
    std::array<Real,6> measure(lbm::amr::LbmAmrView<Descriptor> const& view, bool compare) const;
    std::vector<amrex::BoxArray> covered_;
    std::vector<amrex::MultiFab> previous_;
    Real beginTime_ = 0;
    Real initialMass_ = 0;
    std::ofstream scalars_;
};

} // namespace amr_cavity
#include "CavityObservation.HH"
#endif
