#ifndef LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDMETADATAINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDMETADATAINTEGRATIONTESTS_H

#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/internal/EbGeometryBackend.H"
#include "lbm/boundary/internal/InternalSolidBoundarySystem.H"
#include "lbm/boundary/metadata/InternalCutLinkCompiler.H"

#include <AMReX_EB2.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::integration {
namespace internal_solid_metadata_test {

inline void require(bool condition, char const* message)
{
    if (!condition) {
        amrex::Abort(message);
    }
}

inline bool contains(
    std::vector<std::string> const& errors,
    std::string const& text)
{
    for (auto const& error : errors) {
        if (error.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

inline void require_close(
    amrex::Real actual,
    amrex::Real expected,
    char const* message)
{
    const amrex::Real tolerance =
        64.0 * std::numeric_limits<amrex::Real>::epsilon() *
        (1.0 + std::abs(expected));
    require(std::abs(actual - expected) <= tolerance, message);
}

inline bool same_cell(
    amrex::GpuArray<int, 3> const& lhs,
    amrex::GpuArray<int, 3> const& rhs) noexcept
{
    return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
}


template <class Descriptor>
amrex::Geometry geometry(
    bool periodicZ = true,
    bool periodicX = false,
    bool periodicY = false)
{
    constexpr int nx = 64;
    constexpr int ny = 64;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    amrex::Box const domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, nz - 1)));
    amrex::RealBox const physical(
        {AMREX_D_DECL(0.0, 0.0, 0.0)},
        {AMREX_D_DECL(16.0, 16.0, static_cast<amrex::Real>(nz))});
    amrex::Array<int, AMREX_SPACEDIM> const periodic{
        AMREX_D_DECL(periodicX, periodicY, periodicZ)};
    return amrex::Geometry(
        domain, &physical, amrex::CoordSys::cartesian, periodic.data());
}

inline boundary::InternalBoundarySpec cylinder(
    std::string name,
    amrex::Real x,
    amrex::Real y,
    amrex::Real radius = 1.0,
    std::string objectName = {})
{
    boundary::InternalBoundarySpec spec;
    spec.name = std::move(name);
    spec.geometry = "cylinder";
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.hasPhysical = true;
    spec.scheme = boundary::BoundaryScheme::Yms;
    spec.profile = boundary::BoundaryProfileKind::None;
    spec.objectName = std::move(objectName);
    spec.centerPhys = {x, y, 0.5};
    spec.hasCenterPhys = true;
    spec.radiusPhys = radius;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

template <class Descriptor>
boundary::BoundaryResolveResult resolve(
    amrex::Geometry const& geom,
    std::vector<boundary::InternalBoundarySpec> specs)
{
    return boundary::resolve_boundary_specs<Descriptor>(
        {}, specs, geom);
}

template <class Descriptor>
void check_resolve(amrex::Geometry const& geom)
{
    std::vector<boundary::InternalBoundarySpec> sharedObject{
        cylinder("first_shared", 4.0, 4.0, 1.0, "shared_object"),
        cylinder("second_shared", 8.0, 4.0, 1.0, "shared_object")};
    auto valid = resolve<Descriptor>(geom, std::move(sharedObject));
    require(valid.ok() && valid.resolved.internalObjects.size() == 1 &&
                valid.resolved.internalObjects[0].boundaryCount == 2,
            "valid multi-boundary object resolution failed");

    auto expectInvalid = [&](boundary::InternalBoundarySpec spec,
                             std::string const& text,
                             amrex::Geometry const& inputGeom) {
        auto result = resolve<Descriptor>(inputGeom, {std::move(spec)});
        require(!result.ok() && contains(result.errors, text),
                "internal resolver negative mismatch");
    };
    auto noScheme = cylinder("no_scheme", 4.0, 4.0);
    noScheme.scheme.reset();
    expectInvalid(std::move(noScheme), "scheme is required", geom);
    auto stl = cylinder("stl", 4.0, 4.0);
    stl.geometry = "stl";
    expectInvalid(std::move(stl), "project adapter is not implemented", geom);
    auto periodic = resolve<Descriptor>(geometry<Descriptor>(true, true, false),
        {cylinder("xperiodic", 4.0, 4.0)});
    require(periodic.ok(), "transverse periodic cylinder definition must be supported");
    if constexpr (Descriptor::d == 3) {
        expectInvalid(cylinder("nonperiodic_z", 4.0, 4.0),
                      "requires periodic z", geometry<Descriptor>(false));
    }
    auto overlap = resolve<Descriptor>(
        geom, {cylinder("a", 4.0, 4.0), cylinder("b", 5.0, 4.0)});
    require(!overlap.ok() && contains(overlap.errors, "partial_overlap"),
            "overlapping cylinders must fail");
    auto near = resolve<Descriptor>(
        geom, {cylinder("a2", 4.0, 4.0), cylinder("b2", 6.5, 4.0)});
    require(!near.ok() && contains(near.errors, "insufficient_clearance"),
            "near cylinders must fail");
}

template <class Descriptor>
void check_definition_and_level_constraints(amrex::Geometry const& geom)
{
    auto fineDomain = geom.Domain();
    fineDomain.refine(amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 1));
    const auto physical = geom.ProbDomain();
    const auto periodic = geom.isPeriodicArray();
    const amrex::Geometry fine(fineDomain, &physical, amrex::CoordSys::cartesian, periodic.data());
    auto small = boundary::resolve_boundary_definitions<Descriptor>(
        {}, {cylinder("fine_owner", 4.0, 4.0, 0.625)}, geom);
    require(small.ok(), "global definitions must not apply coarse resolution to a fine owner");
    const std::vector<boundary::BoundaryId> first{{0}};
    require(contains(boundary::validate_internal_boundaries_on_level(small.resolved, first, geom),
                     "R/h_max") &&
                boundary::validate_internal_boundaries_on_level(small.resolved, first, fine).empty(),
            "resolution must use the owning level spacing");
    require(!resolve<Descriptor>(geom, {cylinder("single", 4.0, 4.0, 0.625)}).ok(),
            "single-level resolution must still compose the level check");
    auto nearWall = boundary::resolve_boundary_definitions<Descriptor>(
        {}, {cylinder("near_wall", 1.0, 4.0, 0.625)}, geom);
    require(nearWall.ok() &&
                contains(boundary::validate_internal_boundaries_on_level(nearWall.resolved, first, geom), "non-periodic xlo") &&
                boundary::validate_internal_boundaries_on_level(nearWall.resolved, first, fine).empty(),
            "non-periodic domain clearance must use owner spacing");

    auto pair = boundary::resolve_boundary_definitions<Descriptor>({},
        {cylinder("a", 4.0, 4.0), cylinder("b", 6.5, 4.0)}, geom);
    require(pair.ok(), "global disjoint geometry must not apply a level gap");
    const std::vector<boundary::BoundaryId> second{{1}};
    const std::vector<boundary::BoundaryId> both{{0}, {1}};
    require(boundary::validate_internal_boundaries_on_level(pair.resolved, first, geom).empty() &&
                boundary::validate_internal_boundaries_on_level(pair.resolved, second, geom).empty() &&
                contains(boundary::validate_internal_boundaries_on_level(pair.resolved, both, geom),
                         "insufficient_clearance") &&
                boundary::validate_internal_boundaries_on_level(pair.resolved, both, fine).empty(),
            "only same-level owners share a spacing-dependent pair gap");

    for (auto const& relation : std::vector<std::pair<boundary::InternalBoundarySpec, std::string>>{
             {cylinder("coincident", 4.0, 4.0), "coincident"},
             {cylinder("tangent", 6.0, 4.0), "tangent"},
             {cylinder("contained", 4.1, 4.0, 0.25), "containment"},
             {cylinder("overlap", 5.0, 4.0), "partial_overlap"}}) {
        auto invalid = boundary::resolve_boundary_definitions<Descriptor>(
            {}, {cylinder("base", 4.0, 4.0), relation.first}, geom);
        require(!invalid.ok() && contains(invalid.errors, relation.second),
                "global geometry rejection must precede ownership resolution");
    }
    const auto periodicGeom = geometry<Descriptor>(true, true, false);
    auto seamOverlap = boundary::resolve_boundary_definitions<Descriptor>({},
        {cylinder("left", 0.5, 4.0), cylinder("right", 15.5, 4.0)}, periodicGeom);
    require(!seamOverlap.ok() && contains(seamOverlap.errors, "partial_overlap"),
            "global cylinder validity must include periodic neighbors");
    auto selfOverlap = boundary::resolve_boundary_definitions<Descriptor>(
        {}, {cylinder("self", 8.0, 8.0, 8.0)}, periodicGeom);
    require(!selfOverlap.ok() && contains(selfOverlap.errors, "periodic_self_intersection"),
            "a cylinder touching its own periodic image must be rejected");
    auto selfGap = resolve<Descriptor>(geometry<Descriptor>(true, true, true),
        {cylinder("self_gap", 8.0, 8.0, 7.75)});
    require(selfGap.ok(), "a disjoint self image must not introduce a new three-cell spacing gate");
}

template <class Descriptor>
void check_periodic_query(bool periodicY)
{
    constexpr amrex::Real h = 0.25;
    const int nz = Descriptor::d == 3 ? 4 : 1;
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(31, 31, nz - 1));
    const amrex::RealBox physical({0.0, 0.0, 0.0}, {32.0 * h, 32.0 * h, nz * h});
    const amrex::Array<int, AMREX_SPACEDIM> periodic{1, periodicY, 1};
    const amrex::Geometry geom(domain, &physical, amrex::CoordSys::cartesian, periodic.data());
    const amrex::Real diagonal = periodicY ? std::sqrt(2.0) : 1.0;
    auto resolved = resolve<Descriptor>(geom, {cylinder("seam", 4.25 * h,
        (periodicY ? 4.25 : 16.5) * h, 4.6 * diagonal * h)});
    require(resolved.ok(), resolved.error_message().c_str());
    // Nonzero IDs ensure an image is never mistaken for a new boundary/object.
    resolved.resolved.internal[0].boundaryId = boundary::BoundaryId{7};
    resolved.resolved.internal[0].objectId = boundary::ObjectId{3};
    auto built = boundary::build_internal_geometry_query(resolved.resolved, geom);
    require(built.ok(), built.errorMessage().c_str());
    boundary::LinkIntersectionRequest request;
    request.fluidPoint = {31.5 * h, (periodicY ? 31.5 : 16.5) * h, 0.5 * h};
    request.solidPoint = {32.5 * h, (periodicY ? 32.5 : 16.5) * h,
        (Descriptor::d == 3 ? 1.5 : 0.5) * h};
    auto hit = built.query.intersectLink(request, geom);
    require(hit.contractValid(request) && hit.provenance.boundaryId.value == 7 &&
                hit.provenance.objectId.value == 3,
            "periodic short link must keep its original identities");
    require_close(hit.q, 0.15, "periodic link q must come from the short unwrapped segment");
    require_close(hit.intersection[0], 31.65 * h, "periodic hit x must remain in local unwrapped coordinates");
    require_close(hit.intersection[1], (periodicY ? 31.65 : 16.5) * h, "periodic hit y mismatch");
    require_close(hit.intersection[2], (Descriptor::d == 3 ? 0.65 : 0.5) * h, "periodic hit z mismatch");
    require_close(hit.normalSolidToFluid[0], -1.0 / diagonal, "periodic normal x mismatch");
    require_close(hit.normalSolidToFluid[1], periodicY ? -1.0 / diagonal : 0.0, "periodic normal y mismatch");
    require_close(hit.normalSolidToFluid[2], 0.0, "z-cylinder normal must remain transverse");
    boundary::CenterPointClassification point;
    require(built.query.classifyPoint(request.fluidPoint, geom, point) == boundary::GeometryQueryStatus::Success &&
                point.location == boundary::PointLocation::Outside,
            "periodic fluid classification mismatch");
    require(built.query.classifyPoint(request.solidPoint, geom, point) == boundary::GeometryQueryStatus::Success &&
                point.location == boundary::PointLocation::Inside,
            "periodic solid classification mismatch");
    const amrex::RealArray equivalentSolid{0.5 * h, (periodicY ? 0.5 : 16.5) * h, 0.5 * h};
    require(built.query.classifyPoint(equivalentSolid, geom, point) == boundary::GeometryQueryStatus::Success &&
                point.location == boundary::PointLocation::Inside,
            "periodically equivalent points must have the same phase");
    const amrex::RealArray expectedSurface{31.65 * h, (periodicY ? 31.65 : 16.5) * h,
        (Descriptor::d == 3 ? 0.65 : 0.5) * h};
    require(built.query.classifyPoint(expectedSurface, geom, point) == boundary::GeometryQueryStatus::Success &&
                point.location == boundary::PointLocation::OnSurface && point.surface.unique &&
                point.surface.boundaryId.value == 7 && point.surface.objectId.value == 3,
            "periodic surface provenance mismatch");
    std::vector<amrex::EB2::CylinderIF> functions;
    require(built.query.makeEbCylinderFunctions(functions) && functions.size() == (periodicY ? 4u : 2u),
            "EB must receive transverse combination images without duplicate z images");

    if (!periodicY) {
        resolved.resolved.internal[0].radiusPhys = 4.75 * h;
        auto surfaceQuery = boundary::build_internal_geometry_query(resolved.resolved, geom);
        amrex::BoxArray boxes(domain);
        boxes.maxSize(16);
        amrex::DistributionMapping mapping(boxes);
        amrex::iMultiFab locations(boxes, mapping, 1, 0);
        amrex::iMultiFab tokens(boxes, mapping, 1, 0);
        boundary::PhaseBuildReport report;
        require(surfaceQuery.query.fillCenterPhase(locations, tokens, geom, report) == boundary::GeometryQueryStatus::Success,
                "periodic bulk classification failed");
        const amrex::IntVect cell(31, 16, 0);
        const amrex::Box surfaceCell(cell, cell);
        boundary::GeometrySurfaceProvenance provenance;
        const int token = static_cast<int>(tokens.sum(surfaceCell, 0));
        require(locations.sum(surfaceCell, 0) == boundary::point_location_code(boundary::PointLocation::OnSurface) &&
                    surfaceQuery.query.decodeSurfaceProvenance(token, provenance) == boundary::GeometryQueryStatus::Success &&
                    provenance.boundaryId.value == 7 && provenance.objectId.value == 3,
                "periodic leaf token must decode to the original boundary/object");
    }
}

template <class Descriptor>
void check_bulk_on_surface(amrex::Geometry const& geom)
{
    auto resolved =
        resolve<Descriptor>(geom, {cylinder("surface", 8.125, 8.125)});
    require(resolved.ok(), resolved.error_message().c_str());
    auto query = boundary::build_internal_geometry_query(resolved.resolved, geom);
    require(query.ok(), query.errorMessage().c_str());
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(32);
    amrex::DistributionMapping dm(ba);
    boundary::EbGeometryBackend backend;
    std::string error;
    require(backend.define(query.query, geom, ba, dm, error), error.c_str());
    boundary::CenterPhaseField<Descriptor> phase(ba, dm);
    require(phase.build(query.query, geom, error), error.c_str());
    boundary::InternalCutLinkCompiler<Descriptor> compiler;
    require(compiler.build(query.query, phase, geom, {}, error), error.c_str());
    require(boundary::check_internal_links_against_eb(*backend.factory(), compiler.records(), error), error.c_str());

    const amrex::IntVect surfaceCell(AMREX_D_DECL(36, 32, 0));
    const amrex::Box surfaceBox(surfaceCell, surfaceCell);
    int owners = 0;
    for (int boxIndex = 0; boxIndex < ba.size(); ++boxIndex) {
        if (ba[boxIndex].contains(surfaceCell)) {
            ++owners;
        }
    }
    const amrex::Long phaseValue =
        phase.phase().multifab().sum(surfaceBox, 0);
    const amrex::Long tokenValue =
        phase.surfaceProvenance().multifab().sum(surfaceBox, 0);
    require(owners == 1 &&
                phaseValue == boundary::center_phase_code(boundary::CenterPhase::Fluid) &&
                tokenValue > 0 && phase.report().onSurfaceCount > 0,
            "bulk OnSurface cell must remain Fluid with unique provenance");
}

template <class Descriptor>
void check_exact_query(amrex::Geometry const& geom)
{
    auto resolved = resolve<Descriptor>(geom, {cylinder("query", 8.0, 8.0)});
    require(resolved.ok(), resolved.error_message().c_str());
    auto built = boundary::build_internal_geometry_query(resolved.resolved, geom);
    require(built.ok(), built.errorMessage().c_str());

    boundary::CenterPointClassification surface;
    require(
        built.query.classifyPoint({9.0, 8.0, 0.5}, geom, surface) ==
                boundary::GeometryQueryStatus::Success &&
            surface.location == boundary::PointLocation::OnSurface &&
            surface.surface.unique,
        "analytic OnSurface classification failed");

    auto checkLink = [&](amrex::RealArray fluid,
                         amrex::RealArray solid,
                         amrex::Real expectedQ) {
        boundary::LinkIntersectionRequest request;
        request.fluidPoint = fluid;
        request.solidPoint = solid;
        auto hit = built.query.intersectLink(request, geom);
        require(hit.contractValid(request) &&
                    std::abs(hit.q - expectedQ) < 2.0e-12,
                "analytic exact link contract failed");
        return hit;
    };
    auto axis = checkLink({9.5, 8.0, 0.5}, {8.5, 8.0, 0.5}, 0.5);
    require(std::abs(axis.intersection[0] - 9.0) < 2.0e-12 &&
                std::abs(axis.normalSolidToFluid[0] - 1.0) < 2.0e-12,
            "axis hit/normal mismatch");
    checkLink({9.5, 8.5, 0.5}, {8.5, 7.5, 0.5}, 0.5);
    if constexpr (Descriptor::d == 3) {
        checkLink({9.5, 8.5, 0.0}, {8.5, 7.5, 1.0}, 0.5);
    }

    boundary::LinkIntersectionRequest endpoint;
    endpoint.fluidPoint = {9.0, 8.0, 0.5};
    endpoint.solidPoint = {8.0, 8.0, 0.5};
    endpoint.boundaryId = surface.surface.boundaryId;
    endpoint.objectId = surface.surface.objectId;
    endpoint.surfaceCoincidentFluid = true;
    auto endpointHit = built.query.intersectLink(endpoint, geom);
    require(endpointHit.contractValid(endpoint) && endpointHit.q == 0.0,
            "OnSurface q=0 contract failed");

    boundary::AnalyticCylinderGeometry analytic(resolved.resolved.internal[0]);
    auto tangentRoots = analytic.exactSegmentRoots(
        {7.0, 9.0, 0.5}, {9.0, 9.0, 0.5});
    require(tangentRoots.status == boundary::AnalyticCylinderRootStatus::Tangent,
            "analytic tangent must not become an ordinary root");

    auto decimalSurface = resolved;
    decimalSurface.resolved.internal[0].centerPhys = {0.2, 0.2, 0.5};
    decimalSurface.resolved.internal[0].radiusPhys = 0.05;
    auto decimalQuery =
        boundary::build_internal_geometry_query(decimalSurface.resolved, geom);
    require(
        decimalQuery.ok() &&
            decimalQuery.query.segmentTopology(
                {0.25, 0.2, 0.5}, {0.25, 0.195, 0.5},
                true, false, geom) ==
                boundary::GeometryIntersectionStatus::NoHit,
        "authorized decimal surface-endpoint tangency must remain fluid topology");

    auto chordResolved = resolve<Descriptor>(
        geom, {cylinder("short_chord", 7.548197, 8.951803)});
    require(chordResolved.ok(), chordResolved.error_message().c_str());
    auto chordQuery =
        boundary::build_internal_geometry_query(chordResolved.resolved, geom);
    require(
        chordQuery.ok() &&
            chordQuery.query.segmentTopology(
                {8.125, 8.125, 0.5}, {8.375, 8.375, 0.5},
                false, false, geom) ==
                boundary::GeometryIntersectionStatus::MultipleHits,
        "same-phase diagonal link crossing a cylinder must fail closed");

    auto multipleResolved = resolve<Descriptor>(
        geom, {cylinder("first", 5.0, 8.0), cylinder("second", 9.0, 8.0)});
    require(multipleResolved.ok(), multipleResolved.error_message().c_str());
    auto multipleQuery =
        boundary::build_internal_geometry_query(multipleResolved.resolved, geom);
    boundary::LinkIntersectionRequest multiple;
    multiple.fluidPoint = {3.5, 8.0, 0.5};
    multiple.solidPoint = {9.0, 8.0, 0.5};
    require(multipleQuery.ok() &&
                multipleQuery.query.intersectLink(multiple, geom).status ==
                    boundary::GeometryIntersectionStatus::MultipleHits,
            "one segment crossing multiple objects must fail closed");
}

template <class Descriptor>
void check_next_fluid_gate(amrex::Geometry const& geom)
{
    auto requireDonorFixture =
        [&geom](boundary::ResolvedBoundarySet const& resolved,
                amrex::Real anchorX,
                boundary::PointLocation donorLocation) {
        auto query = boundary::build_internal_geometry_query(resolved, geom);
        require(query.ok(), query.errorMessage().c_str());
        boundary::CenterPointClassification anchor;
        boundary::CenterPointClassification solid;
        boundary::CenterPointClassification donor;
        const amrex::RealArray anchorPoint{anchorX, 8.125, 0.5};
        const amrex::RealArray solidPoint{anchorX + 0.25, 8.125, 0.5};
        const amrex::RealArray donorPoint{anchorX - 0.25, 8.125, 0.5};
        boundary::LinkIntersectionRequest link;
        link.fluidPoint = anchorPoint;
        link.solidPoint = solidPoint;
        require(
            query.query.classifyPoint(anchorPoint, geom, anchor) ==
                    boundary::GeometryQueryStatus::Success &&
                query.query.classifyPoint(solidPoint, geom, solid) ==
                    boundary::GeometryQueryStatus::Success &&
                query.query.classifyPoint(donorPoint, geom, donor) ==
                    boundary::GeometryQueryStatus::Success &&
                anchor.location == boundary::PointLocation::Outside &&
                solid.location == boundary::PointLocation::Inside &&
                donor.location == donorLocation &&
                query.query.intersectLink(link, geom).status ==
                    boundary::GeometryIntersectionStatus::UniqueHit,
            "next-fluid negative fixture does not contain the intended cut link");
    };
    auto compile = [&](boundary::ResolvedBoundarySet const& resolved,
                       bool expectedSuccess,
                       auto inspect) {
        auto query = boundary::build_internal_geometry_query(resolved, geom);
        require(query.ok(), query.errorMessage().c_str());
        amrex::BoxArray ba(geom.Domain());
        ba.maxSize(32);
        amrex::DistributionMapping dm(ba);
        boundary::EbGeometryBackend backend;
        std::string error;
        require(
            backend.define(query.query, geom, ba, dm, error),
            error.c_str());
        boundary::CenterPhaseField<Descriptor> phase(ba, dm);
        require(
            phase.build(query.query, geom, error),
            error.c_str());
        boundary::InternalCutLinkCompiler<Descriptor> compiler;
        const bool built = compiler.build(query.query, phase, geom, {}, error);
        if (expectedSuccess) {
            require(built, error.c_str());
            require(boundary::check_internal_links_against_eb(*backend.factory(), compiler.records(), error), error.c_str());
            inspect(compiler.records());
        } else {
            require(
                !built && error.find("cut-link compilation failed") !=
                    std::string::npos,
                "invalid next-fluid donor must fail cut-link setup");
        }
    };
    auto noInspection = [](auto const&) {};

    auto ordinary = resolve<Descriptor>(
        geom, {cylinder("ordinary_donor", 8.0, 8.0)});
    require(ordinary.ok(), ordinary.error_message().c_str());
    compile(ordinary.resolved, true, [&](auto const& records) {
        amrex::Long count = static_cast<amrex::Long>(records.size());
        amrex::ParallelDescriptor::ReduceLongSum(count);
        require(count > 0, "ordinary pure-fluid donor fixture has no links");
        if constexpr (Descriptor::d == 3) {
            amrex::Long periodicDonors = 0;
            for (auto const& record : records) {
                const int nextZ = record.fluidCell[2] -
                    Descriptor::cz(record.population);
                periodicDonors += nextZ < geom.Domain().smallEnd(2) ||
                    nextZ > geom.Domain().bigEnd(2);
            }
            amrex::ParallelDescriptor::ReduceLongSum(periodicDonors);
            require(periodicDonors > 0,
                    "D3 periodic-z next-fluid donor was not exercised");
        }
    });

    auto onSurface = resolve<Descriptor>(
        geom, {cylinder("surface_donor", 8.125, 8.125)});
    require(onSurface.ok(), onSurface.error_message().c_str());
    compile(onSurface.resolved, true, [&](auto const& records) {
        amrex::Long endpointLinks = 0;
        for (auto const& record : records) {
            endpointLinks += record.surfaceCoincidentFluid != 0 &&
                record.qFromFluid == 0.0;
        }
        amrex::ParallelDescriptor::ReduceLongSum(endpointLinks);
        require(endpointLinks > 0,
                "OnSurface q=0 pure-fluid donor was not exercised");
    });

    auto closePair = resolve<Descriptor>(
        geom,
        {cylinder("left_donor", 4.0, 8.125),
         cylinder("right_donor", 12.0, 8.125)});
    require(closePair.ok(), closePair.error_message().c_str());
    // Bypass the resolver's 3-cell clearance only in this focused fixture so
    // the compiler must reject a Solid or OnSurface next-fluid donor itself.
    closePair.resolved.internal[0].centerPhys[0] = 7.0;
    closePair.resolved.internal[1].centerPhys[0] = 9.25;
    requireDonorFixture(
        closePair.resolved, 8.125, boundary::PointLocation::Inside);
    compile(closePair.resolved, false, noInspection);

    closePair.resolved.internal[0].centerPhys[0] = 6.875;
    requireDonorFixture(
        closePair.resolved, 8.125, boundary::PointLocation::OnSurface);
    compile(closePair.resolved, false, noInspection);

    auto outside = resolve<Descriptor>(
        geom, {cylinder("outside_donor", 8.0, 8.125)});
    require(outside.ok(), outside.error_message().c_str());
    outside.resolved.internal[0].centerPhys[0] = 1.25;
    requireDonorFixture(
        outside.resolved, 0.125, boundary::PointLocation::Outside);
    require(!geom.Domain().contains(amrex::IntVect(AMREX_D_DECL(-1, 32, 0))),
            "outside-donor fixture must place x_ff beyond the domain");
    compile(outside.resolved, false, noInspection);
}

template <class Descriptor>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real executor_source_value(int i, int j, int k, int population) noexcept
{
    return 2.0 + 0.01 * static_cast<amrex::Real>(i) +
        0.002 * static_cast<amrex::Real>(j) +
        0.0003 * static_cast<amrex::Real>(k) +
        0.1 * static_cast<amrex::Real>(population);
}

template <class Descriptor>
amrex::Long check_population_executor_with_resolved(
    amrex::Geometry const& geom,
    int maxGridSize,
    boundary::ResolvedBoundarySet resolved,
    bool expectBouzidi,
    bool expectYms)
{
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(maxGridSize);
    amrex::Vector<int> owners(static_cast<std::size_t>(ba.size()), 0);
    const int rankCount = amrex::ParallelDescriptor::NProcs();
    for (int grid = 0; grid < ba.size(); ++grid) {
        owners[static_cast<std::size_t>(grid)] =
            (ba[grid].smallEnd(0) / maxGridSize) % rankCount;
    }
    amrex::DistributionMapping dm(std::move(owners));
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    std::string error;
    require(system.define(resolved, geom, ba, dm, error), error.c_str());
    require(system.hasPostStreamingBoundaries(),
            "internal executor must advertise post-stream work");

    LatticeField<Descriptor> src("executor_src", ba, dm);
    LatticeField<Descriptor> dst("executor_dst", ba, dm);
    for (amrex::MFIter mfi(src.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const data = src.multifab().array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(), Descriptor::q,
            [=] AMREX_GPU_DEVICE(
                int i, int j, int k, int population) noexcept {
                data(i, j, k, population) =
                    executor_source_value<Descriptor>(i, j, k, population);
            });
    }
    src.fillBoundary(geom, false);
    constexpr amrex::Real sentinel = -1234.5;
    dst.multifab().setVal(sentinel);

    amrex::MultiFab expected(ba, dm, Descriptor::q, 0);
    expected.setVal(sentinel);
    auto const host = system.storage().hostView();
    auto const device = system.storage().deviceView();
    const int zLo = geom.Domain().smallEnd(2);
    const int zHi = geom.Domain().bigEnd(2);
    const int zLength = geom.Domain().length(2);

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #endif
    for (amrex::MFIter mfi(expected); mfi.isValid(); ++mfi) {
        const int grid = mfi.index();
        const amrex::Long begin = host.begin(grid);
        const amrex::Long count = host.end(grid) - begin;
        if (count == 0) {
            continue;
        }
        auto const reference = expected.array(mfi);
        amrex::ParallelFor(
            count,
            [=] AMREX_GPU_DEVICE(amrex::Long n) noexcept {
                auto const& canonical = device.canonical[begin + n];
                auto const& execution = device.execution[begin + n];
                const int i = canonical.fluidCell[0];
                const int j = canonical.fluidCell[1];
                const int k = canonical.fluidCell[2];
                const int direction = canonical.population;
                const int opposite = canonical.oppositePopulation;
                int nextK = k - Descriptor::cz(direction);
                if (nextK < zLo) {
                    nextK += zLength;
                } else if (nextK > zHi) {
                    nextK -= zLength;
                }
                const amrex::Real anchorForward =
                    executor_source_value<Descriptor>(i, j, k, direction);
                const amrex::Real anchorOpposite =
                    executor_source_value<Descriptor>(i, j, k, opposite);
                const amrex::Real nextFluidForward =
                    executor_source_value<Descriptor>(
                        i - Descriptor::cx(direction),
                        j - Descriptor::cy(direction),
                        nextK,
                        direction);
                const amrex::Real q = canonical.qFromFluid;
                amrex::Real value;
                if (execution.scheme == boundary::BoundaryScheme::Bouzidi) {
                    if (q < 0.5) {
                        value = 2.0 * q * anchorForward +
                            (1.0 - 2.0 * q) * nextFluidForward;
                    } else {
                        value = anchorForward / (2.0 * q) +
                            (2.0 * q - 1.0) * anchorOpposite / (2.0 * q);
                    }
                } else {
                    value = (q * anchorOpposite + q * anchorForward +
                             (1.0 - q) * nextFluidForward) /
                        (1.0 + q);
                }
                reference(i, j, k, opposite) = value;
            });
    }

    amrex::Long axisLinks = 0;
    amrex::Long faceDiagonalLinks = 0;
    amrex::Long xyzLinks = 0;
    amrex::Long bouzidiLinks = 0;
    amrex::Long ymsLinks = 0;
    amrex::Long repeatedAnchors = 0;
    amrex::Long crossFabDonors = 0;
    amrex::Long crossRankDonors = 0;
    amrex::Long periodicZDiagonalDonors = 0;
    amrex::Long periodicXyzCornerDonors = 0;
    auto gridContaining = [&](amrex::IntVect const& cell) {
        for (int grid = 0; grid < ba.size(); ++grid) {
            if (ba[grid].contains(cell)) {
                return grid;
            }
        }
        return -1;
    };
    require(!resolved.internal.empty(),
            "internal executor requires at least one resolved boundary");
    const int boundaryIdOffset = resolved.internal.front().boundaryId.value;
    for (amrex::Long n = 0; n < host.count; ++n) {
        auto const& canonical = host.canonical[n];
        auto const& execution = host.execution[n];
        require(
            canonical.solidCell[0] ==
                    canonical.fluidCell[0] + Descriptor::cx(canonical.population) &&
                canonical.solidCell[1] ==
                    canonical.fluidCell[1] + Descriptor::cy(canonical.population) &&
                canonical.solidCell[2] ==
                    canonical.fluidCell[2] + Descriptor::cz(canonical.population) &&
                canonical.oppositePopulation ==
                    Descriptor::opposite(canonical.population),
            "canonical fluid-to-solid endpoint/direction mapping mismatch");
        require(
            canonical.boundaryId >= boundaryIdOffset &&
                canonical.boundaryId - boundaryIdOffset <
                    static_cast<int>(resolved.internal.size()) &&
                canonical.objectId ==
                    resolved.internal[static_cast<std::size_t>(
                        canonical.boundaryId - boundaryIdOffset)]
                        .objectId.value &&
                canonical.qFromFluid >= 0.0 &&
                canonical.qFromFluid < 1.0,
            "canonical q/provenance identity mismatch");
        require(
            same_cell(execution.anchorCell, canonical.fluidCell) &&
                execution.direction == canonical.population &&
                execution.oppositeDirection == canonical.oppositePopulation &&
                execution.oppositeDirection ==
                    Descriptor::opposite(execution.direction),
            "internal executor direction/component mapping mismatch");
        const int norm = Descriptor::cNormSqr(execution.direction);
        axisLinks += norm == 1;
        faceDiagonalLinks += norm == 2;
        xyzLinks += norm == 3;
        bouzidiLinks += execution.scheme == boundary::BoundaryScheme::Bouzidi;
        ymsLinks += execution.scheme == boundary::BoundaryScheme::Yms;
        amrex::IntVect nextFluid(AMREX_D_DECL(
            execution.anchorCell[0] - Descriptor::cx(execution.direction),
            execution.anchorCell[1] - Descriptor::cy(execution.direction),
            execution.anchorCell[2] - Descriptor::cz(execution.direction)));
        const bool wrapsZ = nextFluid[2] < geom.Domain().smallEnd(2) ||
            nextFluid[2] > geom.Domain().bigEnd(2);
        if (wrapsZ) {
            const int zLength = geom.Domain().length(2);
            nextFluid[2] += nextFluid[2] < geom.Domain().smallEnd(2)
                ? zLength
                : -zLength;
        }
        const int donorGrid = gridContaining(nextFluid);
        require(donorGrid >= 0, "internal executor donor is outside the BoxArray");
        const bool crossesFab = donorGrid != canonical.ownerGrid;
        crossFabDonors += crossesFab;
        crossRankDonors += crossesFab &&
            dm[donorGrid] != dm[canonical.ownerGrid];
        periodicZDiagonalDonors += wrapsZ && norm == 2;
        periodicXyzCornerDonors += wrapsZ && crossesFab && norm == 3;
        if (n > 0) {
            auto const& previous = host.execution[n - 1];
            const bool sameAnchor =
                same_cell(previous.anchorCell, execution.anchorCell);
            repeatedAnchors += sameAnchor;
            require(
                !sameAnchor ||
                    previous.oppositeDirection != execution.oppositeDirection,
                "one fluid cell has duplicate internal target components");
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(axisLinks);
    amrex::ParallelDescriptor::ReduceLongSum(faceDiagonalLinks);
    amrex::ParallelDescriptor::ReduceLongSum(xyzLinks);
    amrex::ParallelDescriptor::ReduceLongSum(bouzidiLinks);
    amrex::ParallelDescriptor::ReduceLongSum(ymsLinks);
    amrex::ParallelDescriptor::ReduceLongSum(repeatedAnchors);
    amrex::ParallelDescriptor::ReduceLongSum(crossFabDonors);
    amrex::ParallelDescriptor::ReduceLongSum(crossRankDonors);
    amrex::ParallelDescriptor::ReduceLongSum(periodicZDiagonalDonors);
    amrex::ParallelDescriptor::ReduceLongSum(periodicXyzCornerDonors);
    require(
        axisLinks > 0 && faceDiagonalLinks > 0 &&
            (Descriptor::d == 2 || xyzLinks > 0) &&
            (!expectBouzidi || bouzidiLinks > 0) &&
            (!expectYms || ymsLinks > 0) && repeatedAnchors > 0,
        "internal executor direction/scheme/multi-link coverage is incomplete");
    require(ba.size() == 1 || crossFabDonors > 0,
            "multi-FAB executor fixture did not exercise a next-fluid seam donor");
    require(rankCount == 1 || ba.size() == 1 || crossRankDonors > 0,
            "MPI executor fixture did not place next-fluid donor on another rank");
    if constexpr (Descriptor::d == 3) {
        require(periodicZDiagonalDonors > 0 &&
                    (ba.size() == 1 || periodicXyzCornerDonors > 0),
                "D3 executor fixture did not exercise periodic-z corner donors");
    }

    system.applyPostStreamingValidNodes(src, dst, 0.0);
    amrex::MultiFab mismatch(ba, dm, 1, 0);
    mismatch.setVal(0.0);
    for (amrex::MFIter mfi(dst.multifab(), amrex::TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        auto const actual = dst.multifab().const_array(mfi);
        auto const reference = expected.const_array(mfi);
        auto const difference = mismatch.array(mfi);
        amrex::ParallelFor(
            mfi.tilebox(),
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                amrex::Real maximum = 0.0;
                for (int population = 0; population < Descriptor::q;
                     ++population) {
                    maximum = amrex::max(
                        maximum,
                        std::abs(
                            actual(i, j, k, population) -
                            reference(i, j, k, population)));
                }
                difference(i, j, k) = maximum;
            });
    }
    amrex::Gpu::streamSynchronize();
    const amrex::Real maxError = mismatch.max(0);
    require(
        maxError < 512.0 * std::numeric_limits<amrex::Real>::epsilon(),
        "internal executor src-read/dst-write observable result mismatch");
    amrex::Print() << Descriptor::name()
                   << " internal executor decomposition: grids=" << ba.size()
                   << " ranks=" << rankCount
                   << " links=" << system.storage().report().globalRecordCount
                   << " cross_fab_donors=" << crossFabDonors
                   << " cross_rank_donors=" << crossRankDonors
                   << " periodic_z_diagonal_donors="
                   << periodicZDiagonalDonors
                   << " periodic_xyz_corner_donors="
                   << periodicXyzCornerDonors
                   << " max_error=" << maxError << '\n';
    return system.storage().report().globalRecordCount;
}

template <class Descriptor>
amrex::Long check_population_executor(
    amrex::Geometry const& geom,
    int maxGridSize)
{
    auto bouzidi = cylinder("executor_bouzidi", 10.75, 10.75);
    bouzidi.scheme = boundary::BoundaryScheme::Bouzidi;
    auto resolved = resolve<Descriptor>(
        geom, {cylinder("executor_yms", 5.25, 5.25), std::move(bouzidi)});
    require(resolved.ok(), resolved.error_message().c_str());
    return check_population_executor_with_resolved<Descriptor>(
        geom, maxGridSize, std::move(resolved.resolved), true, true);
}

template <class Descriptor>
void check_system(amrex::Geometry const& geom)
{
    auto resolved = resolve<Descriptor>(geom, {cylinder("system", 4.0, 4.0)});
    require(resolved.ok(), resolved.error_message().c_str());
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(32);
    auto gridContaining = [&](amrex::IntVect const& cell) {
        int found = -1;
        for (int grid = 0; grid < ba.size(); ++grid) {
            if (ba[grid].contains(cell)) {
                require(found < 0, "cell belongs to multiple grids");
                found = grid;
            }
        }
        require(found >= 0, "focused cell is outside the BoxArray");
        return found;
    };
    amrex::Vector<int> localizedOwners(
        static_cast<std::size_t>(ba.size()),
        amrex::ParallelDescriptor::NProcs() > 1 ? 1 : 0);
    localizedOwners[static_cast<std::size_t>(gridContaining({16, 16, 0}))] = 0;
    amrex::DistributionMapping dm(std::move(localizedOwners));
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    const int baseline = amrex::EB2::IndexSpace::size();
    std::string error;
    require(system.define(resolved.resolved, geom, ba, dm, error), error.c_str());
    require(amrex::EB2::IndexSpace::size() == baseline, "prepared internal system must release EB before returning");
    auto const report = system.storage().report();
    auto const host = system.storage().hostView();
    require(report.globalRecordCount > 0 &&
                host.count == report.localRecordCount && host.gridOffsets != nullptr,
            "internal metadata system/storage setup failed");
    amrex::Long minLocal = report.localRecordCount;
    amrex::Long maxLocal = report.localRecordCount;
    amrex::ParallelDescriptor::ReduceLongMin(minLocal);
    amrex::ParallelDescriptor::ReduceLongMax(maxLocal);
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        require(minLocal == 0 && maxLocal > 0,
                "MPI local-empty metadata fixture did not split ownership");
    }
    if (host.count == 0) {
        require(host.canonical == nullptr && host.execution == nullptr,
                "empty rank must expose null record pointers");
    }
    for (int grid = 0; grid < host.gridCount; ++grid) {
        require(host.begin(grid) <= host.end(grid),
                "flat metadata offsets are not monotone");
    }
    for (amrex::Long n = 0; n < host.count; ++n) {
        auto const& record = host.canonical[n];
        require(record.ownerGrid >= 0 && record.ownerGrid < ba.size() &&
                    dm[record.ownerGrid] == amrex::ParallelDescriptor::MyProc() &&
                    record.boundaryId == 0 && record.objectId == 0 &&
                    record.qFromFluid >= 0.0 && record.qFromFluid < 1.0,
                "rank-local canonical metadata mismatch");
    }
    std::vector<boundary::CanonicalCutLinkRecord> qOneRecords;
    qOneRecords.reserve(static_cast<std::size_t>(host.count));
    for (amrex::Long n = 0; n < host.count; ++n) {
        qOneRecords.push_back(host.canonical[n]);
    }
    if (!qOneRecords.empty()) {
        qOneRecords.front().qFromFluid = 1.0;
    }
    boundary::InternalBoundaryMetadataStorage qOneStorage;
    std::string qOneError;
    require(
        !qOneStorage.define(
            qOneRecords, resolved.resolved, geom, ba, dm, qOneError) &&
            qOneError.find("source/layout/identity contract failed") !=
                std::string::npos,
        "q=1 canonical metadata must fail storage setup");
    system.reset();
    require(amrex::EB2::IndexSpace::size() == baseline,
            "internal EB lifetime cleanup failed");
    auto seamResolved =
        resolve<Descriptor>(geom, {cylinder("seam", 7.0, 8.0)});
    require(seamResolved.ok(), seamResolved.error_message().c_str());
    const amrex::IntVect fluid(AMREX_D_DECL(32, 31, 0));
    const amrex::IntVect solid(AMREX_D_DECL(31, 31, 0));
    const int fluidGrid = gridContaining(fluid);
    const int solidGrid = gridContaining(solid);
    amrex::Vector<int> seamOwners(static_cast<std::size_t>(ba.size()), 0);
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        seamOwners[static_cast<std::size_t>(solidGrid)] = 1;
    }
    amrex::DistributionMapping seamDm(std::move(seamOwners));
    require(system.define(
                seamResolved.resolved, geom, ba, seamDm, error),
            error.c_str());
    amrex::Long seamLinks = 0;
    amrex::Long crossRankLinks = 0;
    auto const seamView = system.storage().hostView();
    for (amrex::Long n = 0; n < seamView.count; ++n) {
        auto const& record = seamView.canonical[n];
        amrex::IntVect neighbor(AMREX_D_DECL(
            record.solidCell[0], record.solidCell[1], record.solidCell[2]));
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (geom.isPeriodic(d)) {
                const int length = geom.Domain().length(d);
                while (neighbor[d] < geom.Domain().smallEnd(d)) {
                    neighbor[d] += length;
                }
                while (neighbor[d] > geom.Domain().bigEnd(d)) {
                    neighbor[d] -= length;
                }
            }
        }
        const int neighborGrid = gridContaining(neighbor);
        if (neighborGrid != record.ownerGrid) {
            ++seamLinks;
            crossRankLinks += seamDm[neighborGrid] != seamDm[record.ownerGrid];
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(seamLinks);
    amrex::ParallelDescriptor::ReduceLongSum(crossRankLinks);
    require(seamLinks > 0 &&
                (amrex::ParallelDescriptor::NProcs() == 1 || crossRankLinks > 0),
            "FAB/rank seam cut-link ownership was not exercised");
    system.reset();
    require(amrex::EB2::IndexSpace::size() == baseline,
            "repeated internal EB setup leaked IndexSpace");

    auto chordResolved = resolve<Descriptor>(
        geom, {cylinder("short_chord_system", 7.548197, 8.951803)});
    require(chordResolved.ok(), chordResolved.error_message().c_str());
    require(!system.define(chordResolved.resolved, geom, ba, dm, error) &&
                error.find("cut-link compilation failed") != std::string::npos &&
                amrex::EB2::IndexSpace::size() == baseline,
            "same-phase descriptor chord must fail setup without leaking EB");
}

} // namespace internal_solid_metadata_test

template <class Descriptor>
void run_internal_solid_metadata_integration_tests()
{
    using namespace internal_solid_metadata_test;
    auto geom = geometry<Descriptor>();
    check_resolve<Descriptor>(geom);
    check_definition_and_level_constraints<Descriptor>(geom);
    check_periodic_query<Descriptor>(false);
    check_periodic_query<Descriptor>(true);
    check_exact_query<Descriptor>(geom);
    check_bulk_on_surface<Descriptor>(geom);
    check_next_fluid_gate<Descriptor>(geom);
    const amrex::Long singleFabLinks =
        check_population_executor<Descriptor>(geom, 64);
    const amrex::Long multiFabLinks =
        check_population_executor<Descriptor>(geom, 16);
    require(singleFabLinks == multiFabLinks,
            "internal executor link count changed with BoxArray decomposition");
    check_system<Descriptor>(geom);
    amrex::Print() << Descriptor::name()
                   << " internal solid metadata integration tests passed.\n";
}

} // namespace lbm::tests::integration

#endif
