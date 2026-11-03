#ifndef LBM_TESTS_NUMERICAL_RE100OBSERVATION_H
#define LBM_TESTS_NUMERICAL_RE100OBSERVATION_H

#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <iomanip>

namespace amr_re100 {
class Re100Observation {
public:
    Re100Observation(std::string const& directory, amrex::Real forceDenominator)
        : forceDenominator_(forceDenominator)
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            forces_.open(directory+"/forces.csv");
            forces_ << std::setprecision(17)
                    << "kind,id,level,population_step,coarsest_step,begin_time,end_time,sample_time,Fx_lb,Fy_lb,Fz_lb,Cd,Cl,Cz\n";
            if (!forces_) amrex::Abort("cannot create Re100 force history");
        }
    }
    void consume(lbm::boundary::InternalForceBatch const& batch);
    void finish()
    {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            forces_.flush();
            if (!forces_) amrex::Abort("cannot write Re100 force history");
        }
    }
private:
    void write(char const* kind, int id, lbm::boundary::InternalForceStep const& step,
               amrex::GpuArray<amrex::Real,3> const& force);
    amrex::Real forceDenominator_;
    std::ofstream forces_;
};
} // namespace amr_re100
#include "Re100Observation.HH"
#endif
