#ifndef LBM_TESTS_INTEGRATION_LBM_AMRBOUNDARYINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_AMRBOUNDARYINTEGRATIONTESTS_H

#include "lbm/amr/AmrBoundarySystem.H"

#include <AMReX_Arena.H>
#include <AMReX_BoxIterator.H>
#include <AMReX_EB2.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>

namespace lbm::tests::integration::amr_boundary_test {

inline void require(bool condition, std::string const& message)
{
    if (!condition) {
        amrex::Abort(message);
    }
}

struct Layout {
    amrex::Geometry geometry;
    amrex::BoxArray boxes;
    amrex::DistributionMapping mapping;
};

template <class Descriptor>
std::vector<Layout> make_layouts(bool corners = false, bool emptyRank = false)
{
    std::vector<Layout> layouts;
    for (int level = 0; level < 3; ++level) {
        const int scale = 1 << level;
        const int n = 64 * scale;
        const int nz = Descriptor::d == 3 ? 4 * scale : 1;
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(n - 1, n - 1, nz - 1));
        const amrex::RealBox physical({0.0, 0.0, 0.0}, {16.0, 16.0, Descriptor::d == 3 ? 1.0 : 0.25});
        const amrex::Array<int, AMREX_SPACEDIM> periodic{1, 1, 1};
        Layout layout;
        layout.geometry = amrex::Geometry(domain, &physical, amrex::CoordSys::cartesian, periodic.data());
        if (level == 0) {
            layout.boxes = amrex::BoxArray(domain);
        } else if (corners) {
            amrex::BoxList regions;
            for (int x : {0, n - 32}) {
                for (int y : {0, n - 32}) {
                    regions.push_back(amrex::Box(amrex::IntVect(x, y, 0), amrex::IntVect(x + 31, y + 31, nz - 1)));
                }
            }
            layout.boxes = amrex::BoxArray(regions);
        } else {
            const int lo = level == 1 ? 24 : 96;
            const int hi = level == 1 ? 103 : 159;
            layout.boxes = amrex::BoxArray(amrex::Box(amrex::IntVect(lo, lo, 0), amrex::IntVect(hi, hi, nz - 1)));
        }
        layout.boxes.maxSize(16);
        amrex::Vector<int> owners(layout.boxes.size());
        for (int grid = 0; grid < layout.boxes.size(); ++grid) {
            owners[grid] = emptyRank ? 0 : grid % amrex::ParallelDescriptor::NProcs();
        }
        layout.mapping = amrex::DistributionMapping(std::move(owners));
        layouts.push_back(std::move(layout));
    }
    return layouts;
}

template <class Descriptor>
std::vector<amr::BoundaryLevelLayout> borrow_layouts(std::vector<Layout> const& layouts)
{
    std::vector<amr::BoundaryLevelLayout> borrowed;
    for (auto const& layout : layouts) {
        borrowed.push_back({layout.geometry, layout.boxes, layout.mapping,
            amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 1)});
    }
    return borrowed;
}

inline boundary::InternalBoundarySpec cylinder(
    std::string name, amrex::Real x, amrex::Real y, amrex::Real radius)
{
    boundary::InternalBoundarySpec spec;
    spec.name = std::move(name);
    spec.objectName = "shared";
    spec.geometry = "cylinder";
    spec.physical = boundary::BoundaryPhysical::NoSlipWall;
    spec.hasPhysical = true;
    spec.scheme = boundary::BoundaryScheme::Yms;
    spec.centerPhys = {x, y, 0.5};
    spec.hasCenterPhys = true;
    spec.radiusPhys = radius;
    spec.hasRadiusPhys = true;
    spec.axis = "z";
    spec.hasAxis = true;
    return spec;
}

template <class Descriptor>
void check_partial_factory()
{
    auto layouts = make_layouts<Descriptor>();
    auto const& layout = layouts[2];
    const auto h = layout.geometry.CellSize(0);
    auto definitions = boundary::resolve_boundary_specs<Descriptor>({},
        {cylinder("partial", 130.5 * h, 130.5 * h, 4.0 * h)}, layout.geometry);
    require(definitions.ok(), definitions.error_message());
    auto query = boundary::build_internal_geometry_query(definitions.resolved, layout.geometry);
    std::string error;
    std::vector<boundary::CanonicalCutLinkRecord> records;
    {
        boundary::CenterPhaseField<Descriptor> phase(layout.boxes, layout.mapping);
        boundary::InternalCutLinkCompiler<Descriptor> compiler;
        require(phase.build(query.query, layout.geometry, error), error);
        require(compiler.build(query.query, phase, layout.geometry, {}, error), error);
        records = compiler.records();
    }
    const int baseline = amrex::EB2::IndexSpace::size();
    {
        boundary::EbGeometryBackend backend;
        require(backend.define(query.query, layout.geometry, layout.boxes, layout.mapping, error), error);
        require(boundary::check_internal_links_against_eb(*backend.factory(), records, error), error);
        require(amrex::EB2::IndexSpace::size() == baseline + 1, "partial factory must own exactly one temporary IndexSpace");
    }
    require(amrex::EB2::IndexSpace::size() == baseline, "partial factory must release its IndexSpace");
    amrex::Long cornerReads = 0;
    for (auto const& link : records) {
        const auto box = layout.boxes[link.ownerGrid];
        int outside = 0;
        for (int axis = 0; axis < Descriptor::d; ++axis) {
            outside += link.solidCell[axis] < box.smallEnd(axis) || link.solidCell[axis] > box.bigEnd(axis);
        }
        cornerReads += outside == Descriptor::d;
    }
    amrex::ParallelDescriptor::ReduceLongSum(cornerReads);
    require(cornerReads > 0, "partial L2 factory fixture must read actual diagonal corner ghost flags");
    amrex::Print() << Descriptor::name() << " partial L2 factory/corner flags passed.\n";
}

template <class Descriptor>
void check_missing_donor_labels()
{
    auto layouts = make_layouts<Descriptor>();
    const auto& geom = layouts[2].geometry;
    const auto h = geom.CellSize(0);
    const int nz = geom.Domain().length(2);
    const amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(15, 15, nz - 1)));
    const amrex::DistributionMapping mapping(amrex::Vector<int>{0});
    auto definitions = boundary::resolve_boundary_definitions<Descriptor>({},
        {cylinder("partial_edge", 17.5 * h, 17.5 * h, 4.0 * h)}, geom);
    require(definitions.ok(), definitions.error_message());
    auto query = boundary::build_internal_geometry_query(definitions.resolved, geom);
    boundary::CenterPhaseField<Descriptor> phase(boxes, mapping);
    std::string error;
    require(phase.build(query.query, geom, error), error);
    amrex::MFInfo hostInfo;
    hostInfo.SetArena(amrex::The_Pinned_Arena());
    amrex::iMultiFab host(boxes, mapping, 1, 1, hostInfo);
    amrex::iMultiFab::Copy(host, phase.phase().multifab(), 0, 0, 1, 1);
    amrex::Gpu::streamSynchronize();
    const amrex::IntVect corner(16, 16, Descriptor::d == 3 ? nz : 0);
    require(!boxes.contains(amrex::IntVect(16, 16, 0)), "label fixture must have no valid donor");
    int sawSolid = 0;
    for (amrex::MFIter mfi(host); mfi.isValid(); ++mfi) {
        sawSolid = host.const_array(mfi)(corner) == boundary::center_phase_code(boundary::CenterPhase::Solid);
    }
    amrex::ParallelDescriptor::ReduceIntMax(sawSolid);
    require(sawSolid == 1, "partial corner labels must be computed without a same-level donor");
    boundary::InternalCutLinkCompiler<Descriptor> compiler;
    require(!compiler.build(query.query, phase, geom, {}, error) &&
                error.find("BoundaryId=0") != std::string::npos && error.find("solid=(") != std::string::npos,
            "a solid endpoint outside the effective level must fail uniformly with its location");
    require(compiler.build(query.query, phase, geom, boxes, error) && compiler.records().empty(),
            "covered anchors must be skipped before link compilation");
}

template <class Descriptor>
amrex::Long expected_links(Layout const& layout, boundary::InternalBoundarySpec const& cylinderSpec)
{
    // Independent circle distances over the known valid layout. No query,
    // compiler, storage records or production reconstruction supplies this count.
    const auto h = layout.geometry.CellSizeArray();
    amrex::Long count = 0;
    for (int grid = 0; grid < layout.boxes.size(); ++grid) {
        for (amrex::BoxIterator bit(layout.boxes[grid]); bit.ok(); ++bit) {
            const auto cell = bit();
            amrex::Real dx = (cell[0] + 0.5) * h[0] - cylinderSpec.centerPhys[0];
            amrex::Real dy = (cell[1] + 0.5) * h[1] - cylinderSpec.centerPhys[1];
            dx = std::remainder(dx, layout.geometry.ProbLength(0));
            dy = std::remainder(dy, layout.geometry.ProbLength(1));
            if (std::hypot(dx, dy) < cylinderSpec.radiusPhys) {
                continue;
            }
            for (int q = 1; q < Descriptor::q; ++q) {
                const auto sx = dx + Descriptor::cx(q) * h[0];
                const auto sy = dy + Descriptor::cy(q) * h[1];
                count += std::hypot(sx, sy) < cylinderSpec.radiusPhys;
            }
        }
    }
    return count;
}

template <class Descriptor>
void check_owners(bool corners, bool emptyRank)
{
    auto layouts = make_layouts<Descriptor>(corners, emptyRank);
    auto levels = borrow_layouts<Descriptor>(layouts);
    amr::AmrBoundarySpec spec;
    const amrex::Real h = layouts[2].geometry.CellSize(0);
    if (corners) {
        spec.internal.push_back(cylinder("corner", 2.5 * h, 2.5 * h, 4.0 * h));
    } else {
        spec.internal.push_back(cylinder("L1", 4.5, 8.0, 0.625));
        spec.internal.push_back(cylinder("L2", 130.5 * h, 130.5 * h, 4.0 * h));
        spec.internal[1].scheme = boundary::BoundaryScheme::Bouzidi;
    }
    amr::AmrBoundarySystem<Descriptor> boundaries;
    const int baseline = amrex::EB2::IndexSpace::size();
    std::string error;
    for (int repetition = 0; repetition < 2; ++repetition) {
        require(boundaries.tryDefine(spec, levels, error), error);
        require(amrex::EB2::IndexSpace::size() == baseline, "prepared levels must retain no EB resources");
        require(boundaries.resolved().internalObjects.size() == 1,
                "different boundaries must retain their shared Object identity across levels");
        for (int level = 0; level < 3; ++level) {
            const auto& layout = layouts[level];
            LatticeField<Descriptor> source("prepared_source", layout.boxes, layout.mapping);
            LatticeField<Descriptor> destination("prepared_destination", layout.boxes, layout.mapping);
            source.multifab().setVal(1.0);
            destination.multifab().setVal(-1.0);
            boundaries.applyPostStreamingValidNodes(level, source, destination, 1.29);
            amrex::MultiFab observed(layout.boxes, layout.mapping, 2, 0);
            for (amrex::MFIter mfi(observed); mfi.isValid(); ++mfi) {
                const auto data = destination.multifab().const_array(mfi);
                const auto counts = observed.array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    amrex::Real changed = 0.0;
                    amrex::Real invalid = 0.0;
                    for (int q = 0; q < Descriptor::q; ++q) {
                        const auto value = data(i, j, k, q);
                        if (value != -1.0) {
                            changed += 1.0;
                            invalid += !(std::abs(value - 1.0) < 1.0e-12);
                        }
                    }
                    counts(i, j, k, 0) = changed;
                    counts(i, j, k, 1) = invalid;
                });
            }
            amrex::Long expected = 0;
            for (std::size_t index = 0; index < spec.internal.size(); ++index) {
                const int owner = corners ? 2 : static_cast<int>(index) + 1;
                require(boundaries.internalBoundaryLevel(boundaries.resolved().internal[index].boundaryId) == owner,
                        "boundary must be owned by exactly its effective level");
                if (owner == level) {
                    expected += expected_links<Descriptor>(layout, spec.internal[index]);
                }
            }
            require(observed.sum(0) == static_cast<amrex::Real>(expected) && observed.sum(1) == 0.0,
                    "owner-level updated component count and constant reconstruction must match");
        }
    }
    amrex::Print() << Descriptor::name() << " prepared ownership/storage passed: periodic_corners="
                   << corners << " empty_rank=" << emptyRank << '\n';
}

template <class Descriptor>
void check_multiple_owner_levels()
{
    std::vector<Layout> layouts;
    for (int level = 0; level < 2; ++level) {
        const int scale = 1 << level;
        const int nz = Descriptor::d == 3 ? 4 * scale : 1;
        const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(32 * scale - 1, 32 * scale - 1, nz - 1));
        const amrex::RealBox physical({0.0, 0.0, 0.0}, {32.0, 32.0, Descriptor::d == 3 ? 4.0 : 1.0});
        const amrex::Array<int, AMREX_SPACEDIM> periodic{1, 1, 1};
        Layout layout;
        layout.geometry = amrex::Geometry(domain, &physical, amrex::CoordSys::cartesian, periodic.data());
        layout.boxes = amrex::BoxArray(level == 0 ? domain :
            amrex::Box(amrex::IntVect(14, 28, 0), amrex::IntVect(21, 31, nz - 1)));
        const int rank = level == 1 && amrex::ParallelDescriptor::NProcs() > 1 ? 1 : 0;
        layout.mapping = amrex::DistributionMapping(amrex::Vector<int>{rank});
        layouts.push_back(std::move(layout));
    }
    amr::AmrBoundarySpec spec;
    spec.internal = {cylinder("two_owners", 15.125, 15.125, 4.5)};
    auto definitions = boundary::resolve_boundary_definitions<Descriptor>({}, spec.internal, layouts[0].geometry);
    require(definitions.ok(), definitions.error_message());
    auto query = boundary::build_internal_geometry_query(definitions.resolved, layouts[0].geometry);
    require(query.ok(), query.errorMessage());
    auto covered = layouts[1].boxes;
    covered.coarsen(amrex::IntVect(2, 2, Descriptor::d == 3 ? 2 : 1));
    std::string error;
    // The fine patch resolves a small cap absent from the coarse cell-center
    // classification. Each level has complete links, yet the boundary is split.
    for (int level = 0; level < 2; ++level) {
        auto const& layout = layouts[level];
        boundary::CenterPhaseField<Descriptor> phase(layout.boxes, layout.mapping);
        boundary::InternalCutLinkCompiler<Descriptor> compiler;
        require(phase.build(query.query, layout.geometry, error), error);
        require(compiler.build(query.query, phase, layout.geometry,
                               level == 0 ? covered : amrex::BoxArray{}, error), error);
        amrex::Long count = 0;
        for (auto const& link : compiler.records()) {
            require(link.boundaryId == definitions.resolved.internal[0].boundaryId.value,
                    "both candidate sets must retain the same BoundaryId");
            ++count;
        }
        amrex::ParallelDescriptor::ReduceLongSum(count);
        require(count > 0, "each effective level must compile links before the global ownership rejection");
        amrex::Print() << Descriptor::name() << " split boundary level=" << level << " compiled_links=" << count << '\n';
    }
    auto levels = borrow_layouts<Descriptor>(layouts);
    amr::AmrBoundarySystem<Descriptor> boundaries;
    const int baseline = amrex::EB2::IndexSpace::size();
    require(!boundaries.tryDefine(spec, levels, error) &&
                error.find("has links on multiple effective levels") != std::string::npos &&
                error.find("BoundaryId=0") != std::string::npos,
            "global ownership must reject a boundary with complete links on two levels");
    require(amrex::EB2::IndexSpace::size() == baseline, "multiple ownership rejection must retain no EB resource");
}

template <class Descriptor>
void check_failed_preparation()
{
    auto layouts = make_layouts<Descriptor>();
    auto levels = borrow_layouts<Descriptor>(layouts);
    amr::AmrBoundarySystem<Descriptor> boundaries;
    amr::AmrBoundarySpec spec;
    std::string error;
    const int baseline = amrex::EB2::IndexSpace::size();
    spec.internal = {cylinder("cross_level", 6.0, 8.0, 0.625)};
    require(!boundaries.tryDefine(spec, levels, error) && error.find("crosses effective levels") != std::string::npos &&
                error.find("BoundaryId=0") != std::string::npos,
            "one boundary spanning effective levels must be rejected");
    require(amrex::EB2::IndexSpace::size() == baseline, "failed ownership must retain no EB resource");
    spec.internal = {cylinder("no_links", 1.03, 1.07, 0.01)};
    require(!boundaries.tryDefine(spec, levels, error) && error.find("no effective cut links") != std::string::npos,
            "declared geometry without effective links must fail ownership");
    spec.internal = {cylinder("valid_again", 8.15625, 8.15625, 0.25)};
    require(boundaries.tryDefine(spec, levels, error), error);
    require(amrex::EB2::IndexSpace::size() == baseline, "recovery after failed preparation must release temporary EB");
}

template <class Descriptor>
void run()
{
    check_partial_factory<Descriptor>();
    check_missing_donor_labels<Descriptor>();
    check_owners<Descriptor>(false, false);
    check_owners<Descriptor>(false, true);
    check_owners<Descriptor>(true, false);
    check_multiple_owner_levels<Descriptor>();
    check_failed_preparation<Descriptor>();
    amrex::Print() << Descriptor::name() << " AMR boundary preparation integration passed.\n";
}

} // namespace lbm::tests::integration::amr_boundary_test
#endif
