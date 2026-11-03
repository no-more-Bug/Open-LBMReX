#ifndef LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDFORCEINTEGRATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDFORCEINTEGRATIONTESTS_H

#include "lbm/boundary/core/BoundarySystem.H"
#include "lbm/boundary/internal/force/InternalSolidForce.H"
#include "lbm/boundary/internal/InternalSolidBoundarySystem.H"
#include "integration/lbm/BoundarySystemTestSetup.H"
#include "integration/lbm/InternalSolidMetadataIntegrationTests.H"

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::integration {
namespace internal_solid_force_test {

using internal_solid_metadata_test::cylinder;
using internal_solid_metadata_test::geometry;
using internal_solid_metadata_test::require;
using internal_solid_metadata_test::resolve;

inline boundary::InternalForceStep force_step(int populationStep)
{
    const amrex::Real begin = 1.25 + 0.04*(populationStep-7);
    return {0,populationStep,populationStep,begin,begin+0.04};
}

inline void require_force_close(
    amrex::Real actual,
    amrex::Real expected,
    char const* message)
{
    const amrex::Real tolerance =
        4096.0 * std::numeric_limits<amrex::Real>::epsilon() *
        (1.0 + std::abs(expected));
    require(std::abs(actual - expected) <= tolerance, message);
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real collided_value(
    int i, int j, int k, int population, int sample) noexcept
{
    return 0.25 + 0.1 * sample + 0.002 * i + 0.003 * j +
        0.005 * k + 0.007 * population;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real completed_value(
    int i, int j, int k, int population, int sample) noexcept
{
    return 0.75 + 0.2 * sample + 0.004 * i + 0.006 * j +
        0.008 * k + 0.009 * population;
}

template <class Descriptor>
void fill_force_fields(
    LatticeField<Descriptor>& collided,
    LatticeField<Descriptor>& completed,
    int sample)
{
    for (amrex::MFIter mfi(collided.multifab()); mfi.isValid(); ++mfi) {
        auto const outgoing = collided.multifab().array(mfi);
        auto const returned = completed.multifab().array(mfi);
        amrex::ParallelFor(
            mfi.validbox(), Descriptor::q,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int population) noexcept {
                outgoing(i, j, k, population) =
                    collided_value(i, j, k, population, sample);
                returned(i, j, k, population) =
                    completed_value(i, j, k, population, sample);
            });
    }
}

template <class Descriptor>
std::vector<amrex::Real> expected_boundary_forces(
    boundary::InternalBoundaryMetadataStorage const& storage,
    int boundaryCount,
    int sample)
{
    std::vector<amrex::Real> expected(
        static_cast<std::size_t>(3 * boundaryCount), 0.0);
    auto const view = storage.hostView();
    for (amrex::Long n = 0; n < view.count; ++n) {
        auto const& execution = view.execution[n];
        auto const& canonical = view.canonical[execution.canonicalIndex];
        const amrex::Real momentum = collided_value(
            execution.anchorCell[0], execution.anchorCell[1],
            execution.anchorCell[2], execution.direction, sample) +
            completed_value(
                execution.anchorCell[0], execution.anchorCell[1],
                execution.anchorCell[2], execution.oppositeDirection, sample);
        const std::size_t offset =
            static_cast<std::size_t>(3 * canonical.boundaryId);
        for (int d = 0; d < 3; ++d) {
            expected[offset + static_cast<std::size_t>(d)] += momentum *
                static_cast<amrex::Real>(Descriptor::c(execution.direction, d));
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(
        expected.data(), static_cast<int>(expected.size()));
    return expected;
}

template <class Descriptor>
void check_object_aggregation_and_batch(amrex::Geometry const& geom)
{
    auto sharedA = cylinder("shared_a", 4.0, 4.0);
    auto sharedB = cylinder("shared_b", 8.0, 4.0);
    sharedA.objectName = "shared";
    sharedB.objectName = "shared";
    auto resolved = resolve<Descriptor>(
        geom,
        {std::move(sharedA), std::move(sharedB),
         cylinder("separate", 12.0, 8.0)});
    require(
        resolved.ok() && resolved.resolved.internal.size() == 3 &&
            resolved.resolved.internalObjects.size() == 2 &&
            resolved.resolved.internalObjects[0].boundaryCount == 2 &&
            resolved.resolved.internalObjects[1].boundaryCount == 1 &&
            resolved.resolved.internal[0].name == "shared_a" &&
            resolved.resolved.internal[0].boundaryId.value == 0 &&
            resolved.resolved.internal[0].objectId.value == 0 &&
            resolved.resolved.internal[1].name == "shared_b" &&
            resolved.resolved.internal[1].boundaryId.value == 1 &&
            resolved.resolved.internal[1].objectId.value == 0 &&
            resolved.resolved.internal[2].name == "separate" &&
            resolved.resolved.internal[2].boundaryId.value == 2 &&
            resolved.resolved.internal[2].objectId.value == 1,
        "force object fixture resolution mismatch");

    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(16);
    amrex::DistributionMapping dm(ba);
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    std::string error;
    require(
        system.define(resolved.resolved, geom, ba, dm, error),
        error.c_str());

    amrex::Long boundaryCounts[3]{0, 0, 0};
    amrex::Long repeatedAnchors = 0;
    auto const metadata = system.storage().hostView();
    for (amrex::Long n = 0; n < metadata.count; ++n) {
        auto const& canonical = metadata.canonical[n];
        const int expectedObjectId = canonical.boundaryId < 2 ? 0 : 1;
        require(
            canonical.boundaryId >= 0 && canonical.boundaryId < 3 &&
                canonical.objectId == expectedObjectId,
            "canonical BoundaryId-to-ObjectId mapping mismatch");
        ++boundaryCounts[canonical.boundaryId];
        if (n > 0) {
            auto const& previous = metadata.execution[n - 1];
            auto const& current = metadata.execution[n];
            repeatedAnchors +=
                previous.anchorCell[0] == current.anchorCell[0] &&
                previous.anchorCell[1] == current.anchorCell[1] &&
                previous.anchorCell[2] == current.anchorCell[2];
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(boundaryCounts, 3);
    amrex::ParallelDescriptor::ReduceLongSum(repeatedAnchors);
    require(
        boundaryCounts[0] > 0 && boundaryCounts[1] > 0 &&
            boundaryCounts[2] > 0 && repeatedAnchors > 0,
        "force object/shared-boundary/multiple-link fixture is incomplete");

    LatticeField<Descriptor> collided("force_collided", ba, dm);
    LatticeField<Descriptor> completed("force_completed", ba, dm);
    std::vector<std::vector<amrex::Real>> expected;
    for (int sample = 0; sample < 2; ++sample) {
        fill_force_fields(collided, completed, sample);
        expected.push_back(expected_boundary_forces<Descriptor>(
            system.storage(), 3, sample));
        system.recordInternalForceSample(
            collided, completed, force_step(7 + sample));
    }
    auto batch = system.takeGlobalBoundaryForceBatch();
    require(batch.size() == 2, "force batch sample count mismatch");
    for (std::size_t sample = 0; sample < batch.size(); ++sample) {
        const auto step = force_step(7+static_cast<int>(sample));
        auto const& raw = batch[sample];
        require(raw.step.populationStep == step.populationStep && raw.step.level == 0 &&
                raw.step.coarsestStep == step.coarsestStep && raw.step.beginPhysicalTime == step.beginPhysicalTime &&
                raw.step.endPhysicalTime == step.endPhysicalTime && raw.boundaryForces.size() == 3,
                "collector must preserve complete caller labels and all BoundaryIds");
        for (int boundaryId = 0; boundaryId < 3; ++boundaryId) {
            require(raw.boundaryForces[boundaryId].boundaryId.value == boundaryId,"raw BoundaryId order mismatch");
            for (int d = 0; d < 3; ++d)
                require_force_close(raw.boundaryForces[boundaryId].forceLb[d],expected[sample][3*boundaryId+d],
                                    "raw BoundaryId force mismatch");
        }
        auto objects = boundary::sum_internal_object_forces(raw.boundaryForces,resolved.resolved,{{0},{1}});
        require(objects.size() == 2 && objects[0].objectId.value == 0 && objects[1].objectId.value == 1,
                "Object force identity mismatch");
        for (int d = 0; d < 3; ++d) {
            require_force_close(objects[0].forceLb[d],expected[sample][d]+expected[sample][3+d],"shared Object sum mismatch");
            require_force_close(objects[1].forceLb[d],expected[sample][6+d],"separate Object sum mismatch");
        }
    }
    require(
        system.takeGlobalBoundaryForceBatch().empty(),
        "delivered force batch was not cleared");

    std::vector<boundary::DomainBoundarySpec> walls;
    for (auto const side :
         {boundary::DomainSide::XLo, boundary::DomainSide::XHi,
          boundary::DomainSide::YLo, boundary::DomainSide::YHi}) {
        boundary::DomainBoundarySpec wall;
        wall.name = "force_no_internal_wall_" +
            std::to_string(static_cast<int>(side));
        wall.location = side;
        wall.physical = boundary::BoundaryPhysical::NoSlipWall;
        walls.push_back(std::move(wall));
    }
    boundary::BoundarySystem<Descriptor> noInternal;
    define_boundary_system_for_test(noInternal, walls, geom, ba, dm);
    noInternal.recordInternalForceSample(collided, completed, force_step(12));
    require(
        noInternal.takeGlobalInternalForceBatch().empty(),
        "no-internal force seam must remain empty");
}

} // namespace internal_solid_force_test

template <class Descriptor>
void run_internal_solid_force_integration_tests()
{
    using namespace internal_solid_force_test;
    auto const geom = geometry<Descriptor>();
    check_object_aggregation_and_batch<Descriptor>(geom);
    amrex::Print() << Descriptor::name()
                   << " internal solid force integration tests passed.\n";
}

} // namespace lbm::tests::integration

#endif
