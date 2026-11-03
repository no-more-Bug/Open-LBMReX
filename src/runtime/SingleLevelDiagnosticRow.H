#ifndef LBM_RUNTIME_SINGLELEVELDIAGNOSTICROW_H
#define LBM_RUNTIME_SINGLELEVELDIAGNOSTICROW_H

#include <AMReX_REAL.H>

namespace lbm::runtime {

/**
 * @brief One immutable-by-convention single-level diagnostic observation.
 *
 * Population and macroscopic reductions are complete before this record is
 * created. `timeLb` identifies the current population layer and `timePhys` is
 * its resolved physical-time conversion; output code may serialize the record
 * but must not reinterpret either time or recompute the numerical metrics.
 */
struct DiagnosticRow {
    int step = 0;
    amrex::Real timeLb = 0.0;
    amrex::Real timePhys = 0.0;
    amrex::Real mass = 0.0;
    amrex::Real relativeMassError = 0.0;
    amrex::Real l2Ux = 0.0;
    amrex::Real l2Uy = 0.0;
    amrex::Real l2Uz = 0.0;
    amrex::Real l2U = 0.0;
    amrex::Real l2UExact = 0.0;
    amrex::Real relativeL2U = 0.0;
    amrex::Real linfU = 0.0;
    amrex::Real linfUz = 0.0;
    amrex::Real maxU = 0.0;
    amrex::Real maxRhoError = 0.0;
};

} // namespace lbm::runtime

#endif
