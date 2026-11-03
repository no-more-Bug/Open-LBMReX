#include "lbm/boundary/internal/InternalGeometryQuery.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <cmath>
#include <limits>

namespace lbm::boundary {

bool InternalGeometryQueryRegistry::empty() const noexcept
{
    return leaves_.empty();
}

GeometryQueryStatus InternalGeometryQueryRegistry::fillCenterPhase(
    amrex::iMultiFab& locations,
    amrex::iMultiFab& provenance,
    amrex::Geometry const& geom,
    PhaseBuildReport& report) const
{
    report = {};
    if (&locations == &provenance || !locations.ixType().cellCentered() ||
        !provenance.ixType().cellCentered() || locations.nComp() != 1 ||
        provenance.nComp() != 1 ||
        locations.nGrowVect() != provenance.nGrowVect() ||
        locations.boxArray() != provenance.boxArray() ||
        locations.DistributionMap() != provenance.DistributionMap()) {
        return GeometryQueryStatus::InvalidRequest;
    }
    locations.setVal(point_location_code(PointLocation::Outside));
    provenance.setVal(static_cast<int>(SurfaceProvenanceCode::None));
    amrex::iMultiFab leafLocation(
        locations.boxArray(), locations.DistributionMap(), 1, locations.nGrowVect());
    for (std::size_t leafIndex = 0; leafIndex < leaves_.size(); ++leafIndex) {
        if (leafIndex >=
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return GeometryQueryStatus::InvalidRequest;
        }
        auto const& cylinder = leaves_[leafIndex];
        leafLocation.setVal(-1);
        if (cylinder.fillCenterPhase(leafLocation, geom) !=
            GeometryQueryStatus::Success) {
            return GeometryQueryStatus::InvalidRequest;
        }
        amrex::Gpu::streamSynchronize();
        const int token = surface_provenance_token(
            static_cast<int>(leafIndex));
        for (amrex::MFIter mfi(locations); mfi.isValid(); ++mfi) {
            auto const aggregate = locations.array(mfi);
            auto const aggregateProvenance = provenance.array(mfi);
            auto const leaf = leafLocation.const_array(mfi);
            amrex::ParallelFor(
                mfi.fabbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    const int leafCode = leaf(i, j, k);
                    const int current = aggregate(i, j, k);
                    if (leafCode == point_location_code(PointLocation::Outside)) {
                        return;
                    }
                    if (current != point_location_code(PointLocation::Outside) ||
                        !is_point_location_code(leafCode)) {
                        aggregate(i, j, k) = -1;
                        aggregateProvenance(i, j, k) =
                            static_cast<int>(SurfaceProvenanceCode::Ambiguous);
                        return;
                    }
                    aggregate(i, j, k) = leafCode;
                    aggregateProvenance(i, j, k) =
                        leafCode == point_location_code(PointLocation::OnSurface)
                        ? token
                        : static_cast<int>(SurfaceProvenanceCode::None);
                });
        }
    }

    amrex::ReduceOps<
        amrex::ReduceOpSum,
        amrex::ReduceOpSum,
        amrex::ReduceOpSum,
        amrex::ReduceOpSum> ops;
    amrex::ReduceData<
        amrex::Long,
        amrex::Long,
        amrex::Long,
        amrex::Long> data(ops);
    using Counts = typename decltype(data)::Type;
    for (amrex::MFIter mfi(locations); mfi.isValid(); ++mfi) {
        const auto valid = mfi.validbox();
        auto const values = locations.const_array(mfi);
        auto const tokens = provenance.const_array(mfi);
        ops.eval(
            mfi.fabbox(),
            data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> Counts {
                const int value = values(i, j, k);
                const bool counted = valid.contains(amrex::IntVect(i, j, k));
                const bool invalid = !is_point_location_code(value) ||
                    tokens(i, j, k) ==
                        static_cast<int>(SurfaceProvenanceCode::Ambiguous);
                return {
                    counted && value == point_location_code(PointLocation::Outside),
                    counted && value == point_location_code(PointLocation::Inside),
                    counted && value == point_location_code(PointLocation::OnSurface),
                    invalid};
            });
    }
    Counts const local = data.value(ops);
    amrex::Long counts[4]{
        amrex::get<0>(local), amrex::get<1>(local),
        amrex::get<2>(local), amrex::get<3>(local)};
    amrex::ParallelDescriptor::ReduceLongSum(counts, 4);
    report = {counts[0], counts[1], counts[2]};
    return counts[3] == 0 ? GeometryQueryStatus::Success
                           : GeometryQueryStatus::InvalidRequest;
}

GeometryQueryStatus InternalGeometryQueryRegistry::decodeSurfaceProvenance(
    int token,
    GeometrySurfaceProvenance& result) const noexcept
{
    result = {};
    if (!is_unique_surface_provenance_token(token)) {
        return GeometryQueryStatus::InvalidRequest;
    }
    const int index = surface_provenance_leaf_index(token);
    if (index < 0 || static_cast<std::size_t>(index) >= leaves_.size()) {
        return GeometryQueryStatus::InvalidRequest;
    }
    auto const& cylinder = leaves_[static_cast<std::size_t>(index)];
    result = {cylinder.boundaryId(), cylinder.objectId(), true, true};
    return GeometryQueryStatus::Success;
}

GeometryQueryStatus InternalGeometryQueryRegistry::classifyDisjoint(
    amrex::RealArray const& point,
    amrex::Geometry const& geom,
    CenterPointClassification& result,
    int& insideLeaf) const
{
    result = {};
    insideLeaf = -1;
    int insideCount = 0;
    int surfaceCount = 0;
    for (std::size_t index = 0; index < leaves_.size(); ++index) {
        auto const& cylinder = leaves_[index];
        const SurfaceTolerance tolerance = cylinder.surfaceTolerance(geom);
        PointLocation location = PointLocation::Outside;
        if (!tolerance.valid ||
            cylinder.classifyPoint(point, tolerance.physicalLength, location) !=
                GeometryQueryStatus::Success) {
            return GeometryQueryStatus::InvalidRequest;
        }
        if (location == PointLocation::Inside) {
            ++insideCount;
            insideLeaf = static_cast<int>(index);
        } else if (location == PointLocation::OnSurface) {
            ++surfaceCount;
            result.surface = {
                cylinder.boundaryId(), cylinder.objectId(), true, true};
        }
    }
    if (insideCount > 1 || surfaceCount > 1 ||
        (insideCount != 0 && surfaceCount != 0)) {
        return GeometryQueryStatus::InvalidRequest;
    }
    result.location = insideCount == 1 ? PointLocation::Inside
        : surfaceCount == 1             ? PointLocation::OnSurface
                                        : PointLocation::Outside;
    if (surfaceCount != 1) {
        result.surface = {};
    }
    return GeometryQueryStatus::Success;
}

GeometryQueryStatus InternalGeometryQueryRegistry::classifyPoint(
    amrex::RealArray const& point,
    amrex::Geometry const& geom,
    CenterPointClassification& result) const
{
    int ignored = -1;
    return classifyDisjoint(point, geom, result, ignored);
}

bool InternalGeometryQueryRegistry::makeEbCylinderFunctions(
    std::vector<amrex::EB2::CylinderIF>& functions) const
{
    functions.clear();
    functions.reserve(leaves_.size());
    for (auto const& cylinder : leaves_) {
        functions.push_back(cylinder.implicitFunction());
    }
    return true;
}

std::string InternalGeometryQueryBuildResult::errorMessage() const
{
    std::string message = "Internal geometry query validation failed:";
    for (std::string const& error : errors) {
        message += "\n  - " + error;
    }
    return message;
}

InternalGeometryQueryBuildResult build_internal_geometry_query(
    ResolvedBoundarySet const& resolved, amrex::Geometry const& geom)
{
    InternalGeometryQueryBuildResult result;
    result.query.leaves_.reserve(resolved.internal.size());
    for (ResolvedInternalBoundary const& boundary : resolved.internal) {
        if (boundary.geometry == "cylinder" && boundary.axis == 2 &&
            std::isfinite(boundary.radiusPhys) && boundary.radiusPhys > 0.0) {
            auto canonical = boundary;
            int first[2]{0, 0};
            int last[2]{0, 0};
            for (int axis = 0; axis < 2; ++axis) {
                if (!geom.isPeriodic(axis)) {
                    continue;
                }
                const amrex::Real period = geom.ProbLength(axis);
                amrex::Real offset = std::fmod(boundary.centerPhys[axis] - geom.ProbLo(axis), period);
                if (offset < 0.0) {
                    offset += period;
                }
                canonical.centerPhys[axis] = geom.ProbLo(axis) + offset;
                const amrex::Real center = canonical.centerPhys[axis];
                const amrex::Real radius = canonical.radiusPhys;
                const amrex::Real halo = geom.CellSize(axis);
                first[axis] = static_cast<int>(std::ceil((geom.ProbLo(axis) - halo - center - radius) / period));
                last[axis] = static_cast<int>(std::floor((geom.ProbHi(axis) + halo - center + radius) / period));
            }
            // A z-cylinder is invariant under z translation. Enumerating z
            // images would duplicate the same leaf and make provenance ambiguous.
            for (int x = first[0]; x <= last[0]; ++x) {
                for (int y = first[1]; y <= last[1]; ++y) {
                    auto image = canonical;
                    image.centerPhys[0] += x * geom.ProbLength(0);
                    image.centerPhys[1] += y * geom.ProbLength(1);
                    result.query.leaves_.emplace_back(image);
                }
            }
        } else {
            result.errors.push_back(
                "BoundaryId=" + std::to_string(boundary.boundaryId.value) +
                " lacks a supported Stage 4.5 analytic-cylinder geometry");
        }
    }
    return result;
}

} // namespace lbm::boundary
