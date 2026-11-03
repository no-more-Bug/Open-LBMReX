#ifndef LBM_TESTS_AMR_UNIFORM_FLOW_OBSERVATION_H
#define LBM_TESTS_AMR_UNIFORM_FLOW_OBSERVATION_H
#include "UniformFlowCase.H"
#include "lbm/amr/LbmAmrSimulation.H"
#include <fstream>

namespace amr_uniform_flow {
template<class Descriptor>
class UniformFlowObservation {
public:
    explicit UniformFlowObservation(std::string const& directory);
    void record(lbm::amr::LbmAmrView<Descriptor> const& view);
private:
    std::ofstream scalars_;
};
}
#include "UniformFlowObservation.HH"
#endif
