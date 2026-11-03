#ifndef LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCECOLLECTOR_H
#define LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCECOLLECTOR_H

#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/boundary/metadata/InternalBoundaryMetadataStorage.H"
#include "lbm/fields/LatticeField.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include <cstddef>
#include <string>
#include <vector>

namespace lbm::boundary {

/** @brief Reduce boundary contributions and return the caller's complete step labels. */
template <class Descriptor>
class InternalSolidForceCollector {
public:
    bool define(
        InternalBoundaryMetadataStorage const& storage,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::vector<BoundaryId> const& ownedBoundaryIds,
        std::string& error);
    void reset() noexcept;

    void recordInternalForceSample(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor> const& completedPopulation,
        InternalForceStep const& step);

    [[nodiscard]] std::vector<InternalBoundaryForceValues>
    takeGlobalBoundaryForceBatch();

private:
    InternalBoundaryMetadataStorage const* storage_ = nullptr;
    amrex::BoxArray boxArray_;
    amrex::DistributionMapping distributionMap_;
    std::vector<BoundaryId> ownedBoundaryIds_;
    std::vector<InternalForceStep> pendingSteps_;
    std::vector<amrex::Real> pendingForces_;
};

} // namespace lbm::boundary

#include "lbm/boundary/internal/force/InternalSolidForceCollector.HH"

#endif
