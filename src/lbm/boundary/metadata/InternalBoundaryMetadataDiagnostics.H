#ifndef LBM_BOUNDARY_METADATA_INTERNALBOUNDARYMETADATADIAGNOSTICS_H
#define LBM_BOUNDARY_METADATA_INTERNALBOUNDARYMETADATADIAGNOSTICS_H

#include "lbm/boundary/metadata/CenterPhaseField.H"
#include "lbm/boundary/metadata/InternalBoundaryMetadataStorage.H"

#include <string>

namespace lbm::boundary {

struct InternalBoundaryMetadataDiagnostics {
    int boundaryCount = 0;
    int objectCount = 0;
    CenterPhaseBuildReport phase;
    InternalBoundaryMetadataStorageReport storage;
};

InternalBoundaryMetadataDiagnostics build_internal_metadata_diagnostics(
    ResolvedBoundarySet const& resolved,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    CenterPhaseBuildReport const& phase,
    InternalBoundaryMetadataStorageReport const& storage) noexcept;

std::string format_internal_metadata_diagnostics(
    InternalBoundaryMetadataDiagnostics const& data);

} // namespace lbm::boundary

#endif
