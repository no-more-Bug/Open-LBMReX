#ifndef LBM_TESTS_NUMERICAL_RE100OBSERVATION_HH
#define LBM_TESTS_NUMERICAL_RE100OBSERVATION_HH

namespace amr_re100 {
inline void Re100Observation::write(char const* kind, int id, lbm::boundary::InternalForceStep const& step,
                                    amrex::GpuArray<amrex::Real,3> const& force)
{
    forces_ << kind << ',' << id << ',' << step.level << ',' << step.populationStep << ',' << step.coarsestStep
            << ',' << step.beginPhysicalTime << ',' << step.endPhysicalTime << ',' << step.samplePhysicalTime();
    for (auto value : force) forces_ << ',' << value;
    for (auto value : force) forces_ << ',' << value/forceDenominator_;
    forces_ << '\n';
}

inline void Re100Observation::consume(lbm::boundary::InternalForceBatch const& batch)
{
    // One Object on one level: preserve native boundary and Object step samples.
    if (!batch.objectIntervals.empty()) amrex::Abort("unexpected cross-level Re100 Object interval");
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    for (auto const& sample : batch.boundarySteps)
        for (auto const& force : sample.boundaryForces)
            write("boundary", force.boundaryId.value, sample.step, force.forceLb);
    for (auto const& sample : batch.objectSteps)
        for (auto const& force : sample.objectForces)
            write("object", force.objectId.value, sample.step, force.forceLb);
}
} // namespace amr_re100
#endif
