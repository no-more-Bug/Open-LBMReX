#ifndef LBM_BOUNDARY_METADATA_CENTERPHASEFIELD_HH
#define LBM_BOUNDARY_METADATA_CENTERPHASEFIELD_HH

#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>

#include <algorithm>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

namespace lbm::boundary {

inline bool check_internal_links_against_eb(
    amrex::EBFArrayBoxFactory const& factory,
    std::vector<CanonicalCutLinkRecord> const& records,
    std::string& error)
{
    auto const& flags = factory.getMultiEBCellFlagFab();
    amrex::Gpu::DeviceVector<CanonicalCutLinkRecord> deviceRecords(records.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, records.begin(), records.end(), deviceRecords.begin());
    const auto* links = deviceRecords.data();
    amrex::ReduceOps<amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Long> data(ops);
    using Count = typename decltype(data)::Type;
    for (amrex::MFIter mfi(flags); mfi.isValid(); ++mfi) {
        const int grid = mfi.index();
        const auto begin = std::lower_bound(records.begin(), records.end(), grid,
            [](CanonicalCutLinkRecord const& record, int value) { return record.ownerGrid < value; });
        const auto end = std::upper_bound(begin, records.end(), grid,
            [](int value, CanonicalCutLinkRecord const& record) { return value < record.ownerGrid; });
        const amrex::Long offset = begin - records.begin();
        const amrex::Long count = end - begin;
        const auto flagArray = flags.const_array(mfi);
        ops.eval(count, data, [=] AMREX_GPU_DEVICE(amrex::Long n) -> Count {
            auto const& link = links[offset + n];
            const amrex::IntVect fluid(link.fluidCell[0], link.fluidCell[1], link.fluidCell[2]);
            const amrex::IntVect solid(link.solidCell[0], link.solidCell[1], link.solidCell[2]);
            const amrex::IntVect behind = 2 * fluid - solid;
            if (!flagArray.contains(fluid) || !flagArray.contains(solid) || !flagArray.contains(behind)) {
                return {1};
            }
            const auto fluidFlag = flagArray(fluid);
            const auto solidFlag = flagArray(solid);
            const auto behindFlag = flagArray(behind);
            return {static_cast<amrex::Long>(
                fluidFlag.isMultiValued() || solidFlag.isMultiValued() || behindFlag.isMultiValued() ||
                fluidFlag.isCovered() || solidFlag.isRegular() || behindFlag.isCovered())};
        });
    }
    amrex::Long failures = amrex::get<0>(data.value(ops));
    amrex::ParallelDescriptor::ReduceLongSum(failures);
    if (failures != 0) {
        error = "internal cut-link phase roles disagree with native EB flags";
        return false;
    }
    error.clear();
    return true;
}

template <class Descriptor>
bool CenterPhaseField<Descriptor>::build(
    InternalGeometryQueryRegistry const& query,
    amrex::Geometry const& geom,
    std::string& error)
{
    built_ = false;
    report_ = {};
    error.clear();
    amrex::iMultiFab& phase = phase_.multifab();
    amrex::iMultiFab& provenance = provenance_.multifab();
    if (phase.nGrow() != Descriptor::vicinity ||
        provenance.nGrow() != Descriptor::vicinity ||
        phase.boxArray() != provenance.boxArray() ||
        phase.DistributionMap() != provenance.DistributionMap()) {
        error = "center phase requires the canonical cell-centered layout";
        return false;
    }
    PhaseBuildReport geometryReport;
    if (query.fillCenterPhase(
            phase, provenance, geom, geometryReport) !=
            GeometryQueryStatus::Success) {
        error = "internal geometry could not build unique center phase";
        return false;
    }

    for (amrex::MFIter mfi(phase); mfi.isValid(); ++mfi) {
        auto const location = phase.const_array(mfi);
        auto const token = provenance.const_array(mfi);
        auto const output = phase.array(mfi);
        amrex::ParallelFor(
            mfi.fabbox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                const int code = location(i, j, k);
                const int provenanceToken = token(i, j, k);
                if (code == point_location_code(PointLocation::Outside) &&
                    provenanceToken == 0) {
                    output(i, j, k) = center_phase_code(CenterPhase::Fluid);
                } else if (
                    code == point_location_code(PointLocation::Inside) &&
                    provenanceToken == 0) {
                    output(i, j, k) = center_phase_code(CenterPhase::Solid);
                } else if (
                    code == point_location_code(PointLocation::OnSurface) &&
                    is_unique_surface_provenance_token(provenanceToken)) {
                    output(i, j, k) = center_phase_code(CenterPhase::Fluid);
                }
            });
    }
    report_.fluidCount =
        geometryReport.outsideCount + geometryReport.onSurfaceCount;
    report_.solidCount = geometryReport.insideCount;
    report_.onSurfaceCount = geometryReport.onSurfaceCount;
    built_ = true;
    return true;
}

} // namespace lbm::boundary

#endif
