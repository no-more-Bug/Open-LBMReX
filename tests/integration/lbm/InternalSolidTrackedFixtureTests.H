#ifndef LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDTRACKEDFIXTURETESTS_H
#define LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDTRACKEDFIXTURETESTS_H

#include "integration/lbm/InternalSolidMetadataIntegrationTests.H"

#include "lbm/boundary/core/BoundaryResolver.H"

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>

#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::integration {
namespace internal_solid_tracked_fixture_test {

using internal_solid_metadata_test::check_population_executor_with_resolved;
using internal_solid_metadata_test::require;
using internal_solid_metadata_test::require_close;

struct TrackedInternalSolidFixture {
    std::string identity;
    amrex::Array<int, AMREX_SPACEDIM> nCell{AMREX_D_DECL(1, 1, 1)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> envelopeLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> envelopeHi{AMREX_D_DECL(1.0, 1.0, 1.0)};
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 1)};
    int maxGridSize = 1;
    int reconstructionSteps = 1;
    std::vector<boundary::DomainBoundarySpec> domain;
    std::vector<boundary::InternalBoundarySpec> internal;
    std::vector<int> expectedObjectIds;
    std::vector<std::string> expectedObjectNames;
    std::vector<int> expectedObjectBoundaryCounts;
    bool expectsBouzidi = false;
    bool expectsYms = false;
};

inline boundary::DomainBoundarySpec
make_wall(std::string name, boundary::DomainSide side, bool guo)
{
    boundary::DomainBoundarySpec wall;
    wall.name = std::move(name);
    wall.location = side;
    wall.physical = boundary::BoundaryPhysical::NoSlipWall;
    if (guo) {
        wall.scheme = boundary::BoundaryScheme::GuoExtrapolation;
    }
    return wall;
}

inline std::vector<boundary::DomainBoundarySpec> make_walls(bool guo)
{
    const std::string prefix = guo ? "guo_" : "";
    return {make_wall(prefix + "left_wall", boundary::DomainSide::XLo, guo),
            make_wall(prefix + "right_wall", boundary::DomainSide::XHi, guo),
            make_wall(prefix + "bottom_wall", boundary::DomainSide::YLo, guo),
            make_wall(prefix + "top_wall", boundary::DomainSide::YHi, guo)};
}

inline boundary::InternalBoundarySpec make_cylinder(std::string name,
                                                    boundary::BoundaryScheme scheme,
                                                    std::string tag,
                                                    std::string objectName,
                                                    amrex::Real x,
                                                    amrex::Real y,
                                                    amrex::Real z)
{
    boundary::InternalBoundarySpec cylinder;
    cylinder.name = std::move(name);
    cylinder.geometry = "cylinder";
    cylinder.physical = boundary::BoundaryPhysical::NoSlipWall;
    cylinder.hasPhysical = true;
    cylinder.scheme = scheme;
    cylinder.profile = boundary::BoundaryProfileKind::None;
    cylinder.tag = std::move(tag);
    cylinder.objectName = std::move(objectName);
    cylinder.centerPhys = {x, y, z};
    cylinder.hasCenterPhys = true;
    cylinder.radiusPhys = 0.16;
    cylinder.hasRadiusPhys = true;
    cylinder.axis = "z";
    cylinder.hasAxis = true;
    return cylinder;
}

inline TrackedInternalSolidFixture make_bouzidi_fixture()
{
    TrackedInternalSolidFixture fixture;
    fixture.identity = "internal_cylinder_bouzidi2d";
    fixture.nCell = {50, 50, 4};
    fixture.envelopeHi = {1.0, 1.0, 0.08};
    fixture.maxGridSize = 25;
    fixture.domain = make_walls(false);
    fixture.internal.push_back(make_cylinder("cylinder_wall",
                                             boundary::BoundaryScheme::Bouzidi,
                                             "stage4_6_bouzidi",
                                             "cylinder",
                                             0.5137,
                                             0.4917,
                                             0.04));
    fixture.expectedObjectIds = {0};
    fixture.expectedObjectNames = {"cylinder"};
    fixture.expectedObjectBoundaryCounts = {1};
    fixture.expectsBouzidi = true;
    return fixture;
}

inline TrackedInternalSolidFixture make_guo_yms_fixture()
{
    TrackedInternalSolidFixture fixture;
    fixture.identity = "internal_cylinder_guo2d";
    fixture.nCell = {65, 65, 4};
    fixture.envelopeLo = {-0.0078125, -0.0078125, 0.0};
    fixture.envelopeHi = {1.0078125, 1.0078125, 0.0625};
    fixture.maxGridSize = 16;
    fixture.domain = make_walls(true);
    fixture.internal.push_back(make_cylinder("cylinder_wall",
                                             boundary::BoundaryScheme::Yms,
                                             "stage4_6_guo_internal",
                                             "cylinder",
                                             0.5137,
                                             0.4917,
                                             0.03125));
    fixture.expectedObjectIds = {0};
    fixture.expectedObjectNames = {"cylinder"};
    fixture.expectedObjectBoundaryCounts = {1};
    fixture.expectsYms = true;
    return fixture;
}

inline TrackedInternalSolidFixture make_metadata_fixture()
{
    TrackedInternalSolidFixture fixture;
    fixture.identity = "internal_cylinder_metadata2d";
    fixture.nCell = {50, 50, 4};
    fixture.envelopeHi = {1.0, 1.0, 0.08};
    fixture.maxGridSize = 25;
    fixture.domain = make_walls(false);
    fixture.internal.push_back(make_cylinder("cylinder_wall",
                                             boundary::BoundaryScheme::Yms,
                                             "stage4_6_yms",
                                             "cylinder",
                                             0.5137,
                                             0.4917,
                                             0.04));
    fixture.expectedObjectIds = {0};
    fixture.expectedObjectNames = {"cylinder"};
    fixture.expectedObjectBoundaryCounts = {1};
    fixture.expectsYms = true;
    return fixture;
}

inline TrackedInternalSolidFixture make_multiple_metadata_fixture()
{
    TrackedInternalSolidFixture fixture;
    fixture.identity = "internal_cylinders_metadata2d";
    fixture.nCell = {100, 80, 4};
    fixture.envelopeHi = {2.0, 1.6, 0.08};
    fixture.maxGridSize = 20;
    fixture.domain = make_walls(false);
    fixture.internal = {make_cylinder("cylinder_a_wall",
                                      boundary::BoundaryScheme::Yms,
                                      "",
                                      "cylinder_a",
                                      0.4537,
                                      0.4471,
                                      0.04),
                        make_cylinder("shared_lower_wall",
                                      boundary::BoundaryScheme::Bouzidi,
                                      "",
                                      "shared_pair",
                                      1.0473,
                                      0.3839,
                                      0.04),
                        make_cylinder("shared_upper_wall",
                                      boundary::BoundaryScheme::Yms,
                                      "",
                                      "shared_pair",
                                      1.0521,
                                      0.8177,
                                      0.04),
                        make_cylinder("cylinder_b_wall",
                                      boundary::BoundaryScheme::Yms,
                                      "",
                                      "cylinder_b",
                                      1.5831,
                                      0.5763,
                                      0.04)};
    fixture.expectedObjectIds = {0, 1, 1, 2};
    fixture.expectedObjectNames = {"cylinder_a", "shared_pair", "cylinder_b"};
    fixture.expectedObjectBoundaryCounts = {1, 2, 1};
    fixture.expectsBouzidi = true;
    fixture.expectsYms = true;
    return fixture;
}

template <class Descriptor>
amrex::Geometry make_geometry(TrackedInternalSolidFixture const& fixture)
{
    amrex::Box const domain(amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
                            amrex::IntVect(AMREX_D_DECL(fixture.nCell[0] - 1,
                                                        fixture.nCell[1] - 1,
                                                        fixture.nCell[2] - 1)));
    amrex::RealBox const envelope(fixture.envelopeLo.data(), fixture.envelopeHi.data());
    return amrex::Geometry(
        domain, &envelope, amrex::CoordSys::cartesian, fixture.isPeriodic.data());
}

template <class Descriptor> void check_fixture(TrackedInternalSolidFixture fixture)
{
    require(fixture.nCell[2] == 4 && fixture.reconstructionSteps == 1,
            "tracked internal-solid depth/horizon identity mismatch");
    auto const geom = make_geometry<Descriptor>(fixture);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        require(geom.Domain().length(direction) == fixture.nCell[direction] &&
                    geom.isPeriodic(direction) ==
                        static_cast<bool>(fixture.isPeriodic[direction]),
                "tracked internal-solid mesh identity mismatch");
        require_close(geom.ProbLo(direction),
                      fixture.envelopeLo[direction],
                      "tracked internal-solid lower envelope mismatch");
        require_close(geom.ProbHi(direction),
                      fixture.envelopeHi[direction],
                      "tracked internal-solid upper envelope mismatch");
    }

    auto resolved = boundary::resolve_boundary_specs<Descriptor>(
        fixture.domain, fixture.internal, geom);
    require(resolved.ok(), resolved.error_message().c_str());
    require(resolved.resolved.domain.size() == fixture.domain.size() &&
                resolved.resolved.internal.size() == fixture.internal.size() &&
                resolved.resolved.internalObjects.size() ==
                    fixture.expectedObjectNames.size(),
            "tracked internal-solid resolved shape mismatch");

    for (std::size_t index = 0; index < fixture.domain.size(); ++index) {
        auto const& actual = resolved.resolved.domain[index];
        auto const expectedScheme = fixture.domain[index].scheme.value_or(
            boundary::BoundaryScheme::HalfwayBounceBack);
        require(actual.boundaryId.value == static_cast<int>(index) &&
                    actual.name == fixture.domain[index].name &&
                    actual.physical == boundary::BoundaryPhysical::NoSlipWall &&
                    actual.scheme == expectedScheme &&
                    actual.schemeOrigin ==
                        (fixture.domain[index].scheme.has_value()
                             ? boundary::BoundarySchemeOrigin::User
                             : boundary::BoundarySchemeOrigin::Default),
                "tracked internal-solid domain identity mismatch");
    }

    for (std::size_t index = 0; index < fixture.internal.size(); ++index) {
        auto const& actual = resolved.resolved.internal[index];
        auto const& expected = fixture.internal[index];
        require(actual.boundaryId.value ==
                        static_cast<int>(fixture.domain.size() + index) &&
                    actual.objectId.value == fixture.expectedObjectIds[index] &&
                    actual.name == expected.name &&
                    actual.objectName == expected.objectName &&
                    actual.tag ==
                        (expected.tag.empty() ? expected.name : expected.tag) &&
                    actual.scheme == expected.scheme.value() && actual.axis == 2,
                "tracked internal-solid BoundaryId/ObjectId identity mismatch");
        require_close(actual.centerPhys[0],
                      expected.centerPhys[0],
                      "tracked internal-solid center-x mismatch");
        require_close(actual.centerPhys[1],
                      expected.centerPhys[1],
                      "tracked internal-solid center-y mismatch");
        require_close(actual.centerPhys[2],
                      expected.centerPhys[2],
                      "tracked internal-solid center-z mismatch");
        require_close(actual.radiusPhys,
                      expected.radiusPhys,
                      "tracked internal-solid radius mismatch");
    }

    for (std::size_t index = 0; index < fixture.expectedObjectNames.size(); ++index) {
        auto const& actual = resolved.resolved.internalObjects[index];
        require(actual.objectId.value == static_cast<int>(index) &&
                    actual.name == fixture.expectedObjectNames[index] &&
                    actual.boundaryCount == fixture.expectedObjectBoundaryCounts[index],
                "tracked internal-solid object aggregation identity mismatch");
    }

    auto const links = check_population_executor_with_resolved<Descriptor>(
        geom,
        fixture.maxGridSize,
        std::move(resolved.resolved),
        fixture.expectsBouzidi,
        fixture.expectsYms);
    require(links > 0, "tracked internal-solid fixture has no cut links");
    amrex::Print() << Descriptor::name() << " tracked internal fixture "
                   << fixture.identity << ": PASS\n";
}

} // namespace internal_solid_tracked_fixture_test

template <class Descriptor> void run_internal_solid_tracked_fixture_tests()
{
    using namespace internal_solid_tracked_fixture_test;
    check_fixture<Descriptor>(make_bouzidi_fixture());
    check_fixture<Descriptor>(make_guo_yms_fixture());
    check_fixture<Descriptor>(make_metadata_fixture());
    check_fixture<Descriptor>(make_multiple_metadata_fixture());
}

} // namespace lbm::tests::integration

#endif
