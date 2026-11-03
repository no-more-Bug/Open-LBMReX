#ifndef LBM_BOUNDARY_METADATA_INTERNALBOUNDARYMETADATASTORAGE_H
#define LBM_BOUNDARY_METADATA_INTERNALBOUNDARYMETADATASTORAGE_H

#include "lbm/boundary/core/ResolvedBoundary.H"
#include "lbm/boundary/metadata/InternalBoundaryExecution.H"

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>

#include <string>
#include <vector>

namespace lbm::boundary {

struct InternalBoundaryMetadataStorageReport {
    amrex::Long localRecordCount = 0;
    amrex::Long globalRecordCount = 0;
    int gridCount = 0;
};

/** @brief Immutable rank-local flat host/device metadata storage. */
class InternalBoundaryMetadataStorage {
public:
    InternalBoundaryMetadataStorage() = default;
    InternalBoundaryMetadataStorage(
        InternalBoundaryMetadataStorage const&) = delete;
    InternalBoundaryMetadataStorage& operator=(
        InternalBoundaryMetadataStorage const&) = delete;
    InternalBoundaryMetadataStorage(InternalBoundaryMetadataStorage&&) = delete;
    InternalBoundaryMetadataStorage& operator=(
        InternalBoundaryMetadataStorage&&) = delete;

    bool define(
        std::vector<CanonicalCutLinkRecord> const& source,
        ResolvedBoundarySet const& resolved,
        amrex::Geometry const& geom,
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::string& error);

    [[nodiscard]] InternalBoundaryMetadataView hostView() const noexcept
    {
        return hostView_;
    }
    [[nodiscard]] InternalBoundaryMetadataView deviceView() const noexcept
    {
        return deviceView_;
    }
    [[nodiscard]] InternalBoundaryMetadataStorageReport const& report()
        const noexcept
    {
        return report_;
    }

private:
    void clear() noexcept;

    template <class C, class E, class O>
    static InternalBoundaryMetadataView makeView(
        C const& canonical,
        E const& execution,
        O const& offsets,
        int gridCount) noexcept
    {
        return {
            canonical.empty() ? nullptr : canonical.data(),
            execution.empty() ? nullptr : execution.data(),
            offsets.data(),
            static_cast<amrex::Long>(canonical.size()),
            gridCount};
    }

    amrex::Gpu::HostVector<CanonicalCutLinkRecord> hostCanonical_;
    amrex::Gpu::HostVector<InternalBoundaryExecutionRecord> hostExecution_;
    amrex::Gpu::HostVector<amrex::Long> hostOffsets_;
    amrex::Gpu::DeviceVector<CanonicalCutLinkRecord> deviceCanonical_;
    amrex::Gpu::DeviceVector<InternalBoundaryExecutionRecord> deviceExecution_;
    amrex::Gpu::DeviceVector<amrex::Long> deviceOffsets_;
    InternalBoundaryMetadataView hostView_;
    InternalBoundaryMetadataView deviceView_;
    InternalBoundaryMetadataStorageReport report_;
};

} // namespace lbm::boundary

#endif
