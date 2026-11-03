#ifndef LBM_RUNTIME_SINGLELEVELPROGRAM_H
#define LBM_RUNTIME_SINGLELEVELPROGRAM_H

#include "lbm/boundary/core/BoundarySpec.H"
#include "lbm/config/LbmConfigIntent.H"
#include "lbm/initialization/InitializationSelection.H"
#include "lbm/profiling/TimestepProfiler.H"
#include "runtime/SingleLevelRunIntent.H"
#include "runtime/SingleLevelSetupDiagnostic.H"

#include <AMReX_Array.H>
#include <AMReX_REAL.H>

#include <vector>

namespace lbm::runtime {

template <class Descriptor>
class NoCaseRunSession;

/** AMReX cell-centered index/envelope/periodicity intent. */
struct SingleLevelMeshIntent {
    amrex::Array<int, AMREX_SPACEDIM> nCell{AMREX_D_DECL(64, 64, 1)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> envelopeLo{
        AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> envelopeHi{
        AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{
        AMREX_D_DECL(1, 1, 1)};
    int maxGridSize = 32;
};

/** Parser-free controls for the existing complete-timestep profiler. */
struct SingleLevelProfilingIntent {
    int enabled = 0;
    int warmupSteps = 0;
    int measurementSteps = 0;
    profiling::TimestepProfilingMode mode =
        profiling::TimestepProfilingMode::StageBreakdown;
};

/** Default case preparation for direct programs without benchmark policy. */
struct NoCasePreparation {
    template <class Descriptor>
    using RunSession = NoCaseRunSession<Descriptor>;

    template <class PreparationView>
    SingleLevelCasePreparationResult<
        RunSession<typename PreparationView::DescriptorType>>
    resolve(PreparationView const&) const;
};

/**
 * @brief Parser-free scientific intent entering the single-level assembly seam.
 *
 * Mesh envelope facts remain separate from LBM reference lengths and physical
 * boundary specs. Boundary scheme/profile metadata is the existing authority
 * for physical sampling and integration; this interface does not mirror it.
 */
template <
    class Descriptor,
    class DynamicsSelection,
    class InitializationSelection =
        initialization::ConfiguredInitializationSelection,
    class CasePreparation = NoCasePreparation>
struct SingleLevelProgram {
    SingleLevelMeshIntent mesh;
    config::LbmConfigIntent lbm;
    DynamicsSelection dynamics;
    SingleLevelRunIntent run;
    SingleLevelProfilingIntent profiling;
    InitializationSelection initialization;
    std::vector<boundary::DomainBoundarySpec> domainBoundaries;
    std::vector<boundary::InternalBoundarySpec> internalBoundaries;
    CasePreparation casePreparation;
};

} // namespace lbm::runtime

#endif
