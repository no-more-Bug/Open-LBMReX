#ifndef LBM_BOUNDARY_INTERNAL_INTERNALSOLIDBOUNDARYSYSTEM_H
#define LBM_BOUNDARY_INTERNAL_INTERNALSOLIDBOUNDARYSYSTEM_H

#include "lbm/boundary/internal/EbGeometryBackend.H"
#include "lbm/boundary/internal/force/InternalSolidForceCollector.H"
#include "lbm/boundary/internal/InternalSolidPopulationExecutor.H"
#include "lbm/boundary/metadata/CenterPhaseField.H"
#include "lbm/boundary/metadata/InternalBoundaryMetadataDiagnostics.H"
#include "lbm/boundary/metadata/InternalBoundaryMetadataStorage.H"

#include <AMReX.H>

#include <memory>
#include <string>
#include <vector>

namespace lbm::boundary {

/** @brief Own independent runtime metadata after temporary geometry preparation. */
template <class Descriptor>
class InternalSolidBoundarySystem {
public:
    InternalSolidBoundarySystem() = default;
    InternalSolidBoundarySystem(InternalSolidBoundarySystem const&) = delete;
    InternalSolidBoundarySystem& operator=(
        InternalSolidBoundarySystem const&) = delete;
    InternalSolidBoundarySystem(InternalSolidBoundarySystem&&) noexcept = default;
    InternalSolidBoundarySystem& operator=(
        InternalSolidBoundarySystem&&) noexcept = default;

    bool define(
        ResolvedBoundarySet const& resolved,
        amrex::Geometry const& geom,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::string& error);
    bool definePrepared(
        ResolvedBoundarySet const& resolved,
        std::vector<BoundaryId> const& ownedBoundaryIds,
        amrex::Geometry const& geom,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::vector<CanonicalCutLinkRecord> const& records,
        CenterPhaseBuildReport const& phaseReport,
        std::string& error);
    void reset() noexcept;

    [[nodiscard]] InternalBoundaryMetadataStorage const& storage() const;
    [[nodiscard]] std::string formatDiagnostics() const;
    [[nodiscard]] bool hasPostStreamingBoundaries() const noexcept
    {
        return ready_ && hasInternal_;
    }

    void applyPostStreamingValidNodes(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor>& streamedDestination,
        amrex::Real physicalTime) const;

    void recordInternalForceSample(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor> const& completedPopulation,
        InternalForceStep const& step);

    [[nodiscard]] std::vector<InternalBoundaryForceValues>
    takeGlobalBoundaryForceBatch();

private:
    struct State {
        InternalBoundaryMetadataStorage storage;
        InternalBoundaryMetadataDiagnostics diagnostics;
    };

    void requireState(char const* requested) const;

    std::unique_ptr<State> state_;
    InternalSolidPopulationExecutor<Descriptor> executor_;
    InternalSolidForceCollector<Descriptor> forceCollector_;
    bool ready_ = false;
    bool hasInternal_ = false;
};

} // namespace lbm::boundary

#include "lbm/boundary/internal/InternalSolidBoundarySystem.HH"

#endif
