#include "lbm/boundary/metadata/InternalBoundaryMetadataDiagnostics.H"

#include <sstream>
#include <algorithm>

namespace lbm::boundary {

InternalBoundaryMetadataDiagnostics build_internal_metadata_diagnostics(
    ResolvedBoundarySet const& resolved,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    CenterPhaseBuildReport const& phase,
    InternalBoundaryMetadataStorageReport const& storage) noexcept
{
    int objectCount = 0;
    for (auto const& object : resolved.internalObjects) {
        for (int n = 0; n < object.boundaryCount; ++n) {
            const auto boundaryId = resolved.internalObjectBoundaries[object.boundaryOffset + n];
            if (std::any_of(ownedBoundaryIds.begin(), ownedBoundaryIds.end(),
                    [&](BoundaryId id) { return id.value == boundaryId.value; })) {
                ++objectCount;
                break;
            }
        }
    }
    return {
        static_cast<int>(ownedBoundaryIds.size()),
        objectCount,
        phase,
        storage};
}

std::string format_internal_metadata_diagnostics(
    InternalBoundaryMetadataDiagnostics const& data)
{
    std::ostringstream os;
    os << "Internal solid geometry metadata:\n"
       << "  boundaries=" << data.boundaryCount
       << " objects=" << data.objectCount
       << " phase_fluid=" << data.phase.fluidCount
       << " phase_solid=" << data.phase.solidCount
       << " on_surface_fluid=" << data.phase.onSurfaceCount << '\n'
       << "  canonical_links_global=" << data.storage.globalRecordCount
       << " canonical_links_local=" << data.storage.localRecordCount
       << " grids=" << data.storage.gridCount << '\n'
       << "  executor_available=1\n";
    return os.str();
}

} // namespace lbm::boundary
