#ifndef LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCE_H
#define LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCE_H

#include "lbm/boundary/core/ResolvedBoundary.H"

#include <AMReX_Array.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <vector>

namespace lbm::boundary {

struct InternalForceStep {
    int level = 0;
    int populationStep = -1;
    int coarsestStep = -1;
    amrex::Real beginPhysicalTime = 0;
    amrex::Real endPhysicalTime = 0;

    amrex::Real samplePhysicalTime() const noexcept
    { return beginPhysicalTime + (endPhysicalTime - beginPhysicalTime) / 2; }
};

struct InternalBoundaryForce {
    BoundaryId boundaryId;
    amrex::GpuArray<amrex::Real, 3> forceLb{0.0, 0.0, 0.0};
};

/** Raw collector result, still expressed in this level's lattice units. */
struct InternalBoundaryForceValues {
    InternalForceStep step;
    std::vector<InternalBoundaryForce> boundaryForces;
};

/** Published samples use the common finest-level lattice force units. */
struct InternalBoundaryForceSample {
    InternalForceStep step;
    std::vector<InternalBoundaryForce> boundaryForces;
};

struct InternalObjectForce {
    ObjectId objectId;
    amrex::GpuArray<amrex::Real, 3> forceLb{0.0, 0.0, 0.0};
};

struct InternalObjectForceSample {
    InternalForceStep step;
    std::vector<InternalObjectForce> objectForces;
};

struct InternalObjectForceIntervalSample {
    int coarsestStep = -1;
    amrex::Real beginPhysicalTime = 0;
    amrex::Real endPhysicalTime = 0;
    std::vector<InternalObjectForce> objectForces;
};

struct InternalForceBatch {
    std::vector<InternalBoundaryForceSample> boundarySteps;
    std::vector<InternalObjectForceSample> objectSteps;
    std::vector<InternalObjectForceIntervalSample> objectIntervals;

    bool empty() const noexcept
    { return boundarySteps.empty() && objectSteps.empty() && objectIntervals.empty(); }
};

/** Sum already reduced contributions for the selected global ObjectIds. */
inline std::vector<InternalObjectForce> sum_internal_object_forces(
    std::vector<InternalBoundaryForce> const& contributions,
    ResolvedBoundarySet const& resolved, std::vector<ObjectId> const& objects)
{
    std::vector<InternalObjectForce> result;
    result.reserve(objects.size());
    for (auto id : objects) {
        InternalObjectForce sum{id};
        auto const& object = resolved.internalObjects[id.value];
        for (int n = 0; n < object.boundaryCount; ++n) {
            const auto boundaryId = resolved.internalObjectBoundaries[object.boundaryOffset + n];
            const auto found = std::find_if(contributions.begin(), contributions.end(),
                [boundaryId](auto const& force) { return force.boundaryId.value == boundaryId.value; });
            if (found != contributions.end())
                for (int d = 0; d < 3; ++d) sum.forceLb[d] += found->forceLb[d];
        }
        result.push_back(sum);
    }
    return result;
}

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::GpuArray<amrex::Real, 3> static_internal_solid_link_force(
    amrex::Real outgoing,
    amrex::Real returned,
    int direction) noexcept
{
    const amrex::Real momentum = outgoing + returned;
    return {
        momentum * static_cast<amrex::Real>(Descriptor::cx(direction)),
        momentum * static_cast<amrex::Real>(Descriptor::cy(direction)),
        momentum * static_cast<amrex::Real>(Descriptor::cz(direction))};
}

} // namespace lbm::boundary

#endif
