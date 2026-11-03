#ifndef LBM_TESTS_NUMERICAL_AMRSAMPLEOUTPUT_H
#define LBM_TESTS_NUMERICAL_AMRSAMPLEOUTPUT_H

#include "runtime/LbmAmrPlotfile.H"
#include <fstream>

namespace lbm::tests::numerical {

// Tests-owned native output only; sampling times and scientific policy belong
// to each case. Intermediate fine results are separate single-level files.
template<class Descriptor>
class AmrSampleOutput {
public:
    AmrSampleOutput(std::string directory, amrex::Real velocityReference);
    void layout(amr::LbmAmrView<Descriptor> const& view, std::string const& filename = "layout.csv");
    void synchronized(amr::LbmAmrView<Descriptor> const& view);
    void fine(amr::FineSubstepView<Descriptor> const& view);
private:
    void record(std::string const& kind, int level, int step, amrex::Real time,
                int coarsestStep, std::string const& phase, std::string const& path);
    std::string directory_;
    amrex::Real velocityReference_;
    std::ofstream samples_;
};

} // namespace lbm::tests::numerical
#include "numerical/common/AmrSampleOutput.HH"
#endif
