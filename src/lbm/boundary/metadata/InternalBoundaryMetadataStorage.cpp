#include "lbm/boundary/metadata/InternalBoundaryMetadataStorage.H"

#include "lbm/boundary/internal/InternalBoundarySchemeTraits.H"
#include "lbm/boundary/internal/InternalGeometryLayout.H"

#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>

namespace lbm::boundary {
namespace {

bool project(
    CanonicalCutLinkRecord const& canonical,
    InternalBoundarySchemeTraits const& traits,
    int index,
    InternalBoundaryExecutionRecord& output) noexcept
{
    if ((canonical.surfaceCoincidentFluid != 0 &&
         canonical.surfaceCoincidentFluid != 1) ||
        !std::isfinite(canonical.qFromFluid) ||
        (canonical.surfaceCoincidentFluid != 0
             ? canonical.qFromFluid != 0.0
             : !(canonical.qFromFluid > 0.0 &&
                 canonical.qFromFluid < 1.0))) {
        return false;
    }
    output.canonicalIndex = index;
    output.scheme = traits.scheme;
    output.halfLinkPolicy = traits.halfLinkPolicy;
    output.anchorCell = canonical.fluidCell;
    output.direction = canonical.population;
    output.oppositeDirection = canonical.oppositePopulation;
    output.qFromAnchor = canonical.qFromFluid;
    return true;
}

} // namespace

bool InternalBoundaryMetadataStorage::define(
    std::vector<CanonicalCutLinkRecord> const& source,
    ResolvedBoundarySet const& resolved,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::string& error)
{
    clear();
    error.clear();
    int failed = !is_canonical_internal_cell_layout(ba, geom) ||
        ba.size() != dm.size();
    const int rank = amrex::ParallelDescriptor::MyProc();
    std::vector<ResolvedInternalBoundary const*> byId;
    int maximumId = -1;
    for (auto const& boundary : resolved.internal) {
        maximumId = std::max(maximumId, boundary.boundaryId.value);
    }
    byId.resize(static_cast<std::size_t>(maximumId + 1), nullptr);
    for (auto const& boundary : resolved.internal) {
        if (boundary.boundaryId.value < 0 ||
            byId[static_cast<std::size_t>(boundary.boundaryId.value)] !=
                nullptr) {
            failed = 1;
        } else {
            byId[static_cast<std::size_t>(boundary.boundaryId.value)] =
                &boundary;
        }
    }

    std::vector<InternalBoundaryExecutionRecord> execution;
    execution.reserve(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        auto const& record = source[index];
        if (record.ownerGrid < 0 || record.ownerGrid >= ba.size() ||
            dm[record.ownerGrid] != rank ||
            !ba[record.ownerGrid].contains(amrex::IntVect(AMREX_D_DECL(
                record.fluidCell[0], record.fluidCell[1],
                record.fluidCell[2]))) ||
            record.boundaryId < 0 ||
            record.boundaryId >= static_cast<int>(byId.size()) ||
            byId[static_cast<std::size_t>(record.boundaryId)] == nullptr ||
            byId[static_cast<std::size_t>(record.boundaryId)]->objectId.value !=
                record.objectId ||
            (index > 0 && !canonical_cut_link_key_less(
                source[index - 1], record))) {
            failed = 1;
            continue;
        }
        auto const traits = internal_boundary_scheme_traits(
            byId[static_cast<std::size_t>(record.boundaryId)]->scheme);
        InternalBoundaryExecutionRecord projected;
        if (!traits || !project(record, *traits, static_cast<int>(index), projected)) {
            failed = 1;
            continue;
        }
        execution.push_back(projected);
    }
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed != 0) {
        error = "internal metadata source/layout/identity contract failed";
        return false;
    }

    hostCanonical_.resize(source.size());
    std::copy(source.begin(), source.end(), hostCanonical_.begin());
    hostExecution_.resize(execution.size());
    std::copy(execution.begin(), execution.end(), hostExecution_.begin());
    hostOffsets_.resize(static_cast<std::size_t>(ba.size()) + 1U);
    amrex::Long cursor = 0;
    for (int grid = 0; grid < ba.size(); ++grid) {
        hostOffsets_[static_cast<std::size_t>(grid)] = cursor;
        while (cursor < static_cast<amrex::Long>(source.size()) &&
               source[static_cast<std::size_t>(cursor)].ownerGrid == grid) {
            ++cursor;
        }
    }
    hostOffsets_.back() = cursor;

    deviceCanonical_.resize(hostCanonical_.size());
    deviceExecution_.resize(hostExecution_.size());
    deviceOffsets_.resize(hostOffsets_.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, hostCanonical_.begin(), hostCanonical_.end(),
        deviceCanonical_.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, hostExecution_.begin(), hostExecution_.end(),
        deviceExecution_.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, hostOffsets_.begin(), hostOffsets_.end(),
        deviceOffsets_.begin());
    report_.localRecordCount = static_cast<amrex::Long>(source.size());
    report_.globalRecordCount = report_.localRecordCount;
    report_.gridCount = ba.size();
    amrex::ParallelDescriptor::ReduceLongSum(report_.globalRecordCount);
    if (report_.globalRecordCount == 0) {
        clear();
        error = "internal geometry produced no cut links";
        return false;
    }
    hostView_ = makeView(
        hostCanonical_, hostExecution_, hostOffsets_, ba.size());
    deviceView_ = makeView(
        deviceCanonical_, deviceExecution_, deviceOffsets_, ba.size());
    return true;
}

void InternalBoundaryMetadataStorage::clear() noexcept
{
    hostCanonical_.clear();
    hostExecution_.clear();
    hostOffsets_.clear();
    deviceCanonical_.clear();
    deviceExecution_.clear();
    deviceOffsets_.clear();
    hostView_ = {};
    deviceView_ = {};
    report_ = {};
}

} // namespace lbm::boundary
