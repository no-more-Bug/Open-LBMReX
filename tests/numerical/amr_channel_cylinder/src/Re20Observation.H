#ifndef LBM_TESTS_NUMERICAL_RE20OBSERVATION_H
#define LBM_TESTS_NUMERICAL_RE20OBSERVATION_H

#include "Re20Case.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include <array>
#include <fstream>
#include <string>
#include <vector>

namespace amr_re20 {

template<class Descriptor>
class Re20Observation {
public:
    Re20Observation(std::string const& directory, Real forceDenominator, lbm::boundary::ObjectId objectId);
    void initialize(lbm::amr::LbmAmrView<Descriptor> const& view);
    void save(lbm::amr::LbmAmrView<Descriptor> const& view);
    void consume(lbm::boundary::InternalForceBatch const& batch);
    void record(lbm::amr::LbmAmrView<Descriptor> const& view, int commonStep);
    void profiles(lbm::amr::LbmAmrView<Descriptor> const& view) const;
    void finish();
private:
    struct FieldMetrics {
        Real changeSquared = 0, speedSquared = 0;
        Real rhoFront = 0, rhoRear = 0, phiIn = 0, phiOut = 0;
        Real uzMax = 0, rhoZMax = 0, velocityZMax = 0;
    };
    FieldMetrics measure(lbm::amr::LbmAmrView<Descriptor> const& view, bool compare) const;
    bool fluid(amrex::IntVect const& cell, amrex::Geometry const& geometry) const;
    std::vector<Real> planeMean(amrex::MultiFab const& host, amrex::Geometry const& geometry) const;
    void writeForce(char const* kind, int id, lbm::boundary::InternalForceStep const& step,
                    amrex::GpuArray<Real,3> const& force);
    void writeFields(FieldMetrics const& metrics);
    std::string directory_;
    Real forceDenominator_;
    lbm::boundary::ObjectId objectId_;
    std::vector<std::vector<unsigned char>> active_;
    std::vector<amrex::MultiFab> previous_;
    Real beginTime_ = 0;
    int beginPopulationStep_ = 0;
    std::array<Real,2> forceIntegral_{};
    int intervalSamples_ = 0;
    bool collectingInterval_ = false;
    std::ofstream scalars_, forces_;
};

} // namespace amr_re20
#include "Re20Observation.HH"
#endif
