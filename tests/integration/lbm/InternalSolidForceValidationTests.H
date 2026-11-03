#ifndef LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDFORCEVALIDATIONTESTS_H
#define LBM_TESTS_INTEGRATION_LBM_INTERNALSOLIDFORCEVALIDATIONTESTS_H

#include "integration/lbm/InternalSolidForceIntegrationTests.H"

#include <AMReX_DistributionMapping.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lbm::tests::integration {
namespace internal_solid_force_validation_test {

using internal_solid_metadata_test::cylinder;
using internal_solid_metadata_test::require;
using internal_solid_metadata_test::resolve;
using internal_solid_force_test::collided_value;
using internal_solid_force_test::completed_value;
using internal_solid_force_test::fill_force_fields;
using internal_solid_force_test::force_step;
using internal_solid_force_test::require_force_close;

struct ForceCaseResult {
    amrex::GpuArray<amrex::Real, 3> force{0.0, 0.0, 0.0};
    amrex::GpuArray<amrex::Real, 3> expected{0.0, 0.0, 0.0};
    int objectGridCount = 0;
    int objectOwnerRankCount = 0;
};

template <class Descriptor>
amrex::Geometry force_geometry(int nz)
{
    constexpr int nx = 64;
    constexpr int ny = 64;
    const int depth = Descriptor::d == 3 ? nz : 1;
    amrex::Box const domain(
        amrex::IntVect(AMREX_D_DECL(0, 0, 0)),
        amrex::IntVect(AMREX_D_DECL(nx - 1, ny - 1, depth - 1)));
    amrex::RealBox const physical(
        {AMREX_D_DECL(0.0, 0.0, 0.0)},
        {AMREX_D_DECL(16.0, 16.0, static_cast<amrex::Real>(depth))});
    amrex::Array<int, AMREX_SPACEDIM> const periodic{
        AMREX_D_DECL(0, 0, 1)};
    return amrex::Geometry(
        domain, &physical, amrex::CoordSys::cartesian, periodic.data());
}

inline amrex::DistributionMapping force_distribution(
    amrex::BoxArray const& ba,
    amrex::Geometry const& geom,
    bool splitAcrossRanks)
{
    if (!splitAcrossRanks || amrex::ParallelDescriptor::NProcs() == 1) {
        return amrex::DistributionMapping(ba);
    }
    const int split = geom.Domain().smallEnd(0) + geom.Domain().length(0) / 2;
    amrex::Vector<int> owners(static_cast<std::size_t>(ba.size()), 0);
    for (int grid = 0; grid < ba.size(); ++grid) {
        owners[static_cast<std::size_t>(grid)] =
            ba[grid].smallEnd(0) < split ? 0 : 1;
    }
    return amrex::DistributionMapping(std::move(owners));
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real validation_collided_value(int i, int j, int sample) noexcept
{
    return 0.30 + 0.01 * sample + 0.002 * i + 0.001 * j;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real validation_completed_value(int i, int j, int sample) noexcept
{
    return 0.70 + 0.02 * sample + 0.004 * i + 0.002 * j;
}

template <class Descriptor>
void fill_validation_fields(
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
                amrex::ignore_unused(k, population);
                outgoing(i, j, k, population) =
                    validation_collided_value(i, j, sample);
                returned(i, j, k, population) =
                    validation_completed_value(i, j, sample);
            });
    }
}

template <class Descriptor>
amrex::GpuArray<amrex::Real, 3> validation_reference(
    boundary::InternalBoundaryMetadataStorage const& storage,
    int sample)
{
    amrex::GpuArray<amrex::Real, 3> expected{0.0, 0.0, 0.0};
    auto const view = storage.hostView();
    for (amrex::Long n = 0; n < view.count; ++n) {
        auto const& execution = view.execution[n];
        auto const& canonical = view.canonical[execution.canonicalIndex];
        if (canonical.objectId != 0) {
            continue;
        }
        const amrex::Real momentum = validation_collided_value(
            execution.anchorCell[0], execution.anchorCell[1], sample) +
            validation_completed_value(
                execution.anchorCell[0], execution.anchorCell[1], sample);
        for (int d = 0; d < 3; ++d) {
            expected[d] += momentum * static_cast<amrex::Real>(
                Descriptor::c(execution.direction, d));
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(expected.data(), 3);
    return expected;
}

inline void require_force_equal(
    amrex::GpuArray<amrex::Real, 3> const& actual,
    amrex::GpuArray<amrex::Real, 3> const& expected,
    char const* message)
{
    for (int d = 0; d < 3; ++d) {
        require_force_close(actual[d], expected[d], message);
    }
}

template <class Descriptor>
ForceCaseResult run_force_case(
    amrex::Geometry const& geom,
    amrex::IntVect const& maxSize,
    bool splitAcrossRanks,
    int sample,
    int populationStep)
{
    auto resolved = resolve<Descriptor>(
        geom, {cylinder("force_validation", 8.0, 8.0)});
    require(
        resolved.ok() && resolved.resolved.internalObjects.size() == 1,
        "force validation fixture resolution mismatch");
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(maxSize);
    auto dm = force_distribution(ba, geom, splitAcrossRanks);
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    std::string error;
    require(
        system.define(resolved.resolved, geom, ba, dm, error),
        error.c_str());

    auto const metadata = system.storage().hostView();
    amrex::Vector<int> objectGrids(static_cast<std::size_t>(ba.size()), 0);
    for (amrex::Long n = 0; n < metadata.count; ++n) {
        auto const& canonical = metadata.canonical[n];
        if (canonical.objectId == 0) {
            objectGrids[static_cast<std::size_t>(canonical.ownerGrid)] = 1;
        }
    }
    const int localOwner = metadata.count > 0 ? 1 : 0;
    int ownerRankCount = localOwner;
    amrex::ParallelDescriptor::ReduceIntMax(
        objectGrids.data(), static_cast<int>(objectGrids.size()));
    amrex::ParallelDescriptor::ReduceIntSum(ownerRankCount);

    LatticeField<Descriptor> collided("validation_collided", ba, dm);
    LatticeField<Descriptor> completed("validation_completed", ba, dm);
    fill_validation_fields(collided, completed, sample);
    auto const expected = validation_reference<Descriptor>(system.storage(), sample);
    system.recordInternalForceSample(
        collided, completed, force_step(populationStep));
    auto batch = system.takeGlobalBoundaryForceBatch();
    require(
        batch.size() == 1 && batch[0].step.populationStep == populationStep &&
            batch[0].boundaryForces.size() == 1 &&
            batch[0].boundaryForces[0].boundaryId.value == 0,
        "force validation batch shape mismatch");

    ForceCaseResult result;
    result.force = batch[0].boundaryForces[0].forceLb;
    result.expected = expected;
    result.objectGridCount = static_cast<int>(std::count(
        objectGrids.begin(), objectGrids.end(), 1));
    result.objectOwnerRankCount = ownerRankCount;
    require_force_equal(
        result.force, result.expected,
        "force decomposition direct reference mismatch");
    return result;
}

template <class Descriptor>
void check_decomposition()
{
    auto const geom = force_geometry<Descriptor>(4);
    auto const single = run_force_case<Descriptor>(
        geom, amrex::IntVect(AMREX_D_DECL(64, 64, 4)),
        false, 0, 5);
    auto const multi = run_force_case<Descriptor>(
        geom, amrex::IntVect(AMREX_D_DECL(32, 32, 2)),
        true, 0, 5);
    require(
        single.objectGridCount == 1 && multi.objectGridCount > 1,
        "single-/multi-FAB force fixture coverage mismatch");
    if (amrex::ParallelDescriptor::NProcs() == 2) {
        require(
            multi.objectOwnerRankCount == 2,
            "force ObjectId did not span both MPI ranks");
    }
    require_force_equal(
        multi.force, single.force,
        "single-/multi-FAB force decomposition mismatch");
}

template <class Descriptor>
amrex::GpuArray<amrex::Real, 3> ordered_reference(
    boundary::InternalBoundaryMetadataStorage const& storage,
    int sample,
    bool swapped)
{
    amrex::GpuArray<amrex::Real, 3> expected{0.0, 0.0, 0.0};
    auto const view = storage.hostView();
    for (amrex::Long n = 0; n < view.count; ++n) {
        auto const& execution = view.execution[n];
        const int i = execution.anchorCell[0];
        const int j = execution.anchorCell[1];
        const int k = execution.anchorCell[2];
        const amrex::Real momentum = swapped
            ? completed_value(i, j, k, execution.direction, sample) +
                collided_value(
                    i, j, k, execution.oppositeDirection, sample)
            : collided_value(i, j, k, execution.direction, sample) +
                completed_value(
                    i, j, k, execution.oppositeDirection, sample);
        for (int d = 0; d < 3; ++d) {
            expected[d] += momentum * static_cast<amrex::Real>(
                Descriptor::c(execution.direction, d));
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(expected.data(), 3);
    return expected;
}

template <class Descriptor>
void check_population_order()
{
    auto const geom = force_geometry<Descriptor>(4);
    auto resolved = resolve<Descriptor>(
        geom, {cylinder("force_population_order", 8.0, 8.0)});
    require(resolved.ok(), resolved.error_message().c_str());
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(32, 32, 2)));
    auto dm = force_distribution(ba, geom, true);
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    std::string error;
    require(
        system.define(resolved.resolved, geom, ba, dm, error),
        error.c_str());
    LatticeField<Descriptor> collided("order_collided", ba, dm);
    LatticeField<Descriptor> completed("order_completed", ba, dm);
    fill_force_fields(collided, completed, 1);

    auto const expected = ordered_reference<Descriptor>(
        system.storage(), 1, false);
    system.recordInternalForceSample(collided, completed, force_step(20));
    auto correct = system.takeGlobalBoundaryForceBatch();
    require(
        correct.size() == 1 && correct[0].boundaryForces.size() == 1,
        "ordered force sample missing");
    require_force_equal(
        correct[0].boundaryForces[0].forceLb, expected,
        "old-collided/current-completed force order mismatch");

    auto const swappedExpected = ordered_reference<Descriptor>(
        system.storage(), 1, true);
    system.recordInternalForceSample(completed, collided, force_step(21));
    auto swapped = system.takeGlobalBoundaryForceBatch();
    require(
        swapped.size() == 1 && swapped[0].boundaryForces.size() == 1,
        "swapped force sample missing");
    require_force_equal(
        swapped[0].boundaryForces[0].forceLb, swappedExpected,
        "swapped population reference mismatch");
    bool differs = false;
    for (int d = 0; d < 3; ++d) {
        differs = differs || std::abs(expected[d] - swappedExpected[d]) >
            4096.0 * std::numeric_limits<amrex::Real>::epsilon();
    }
    require(
        differs,
        "population-order negative fixture cannot detect argument reversal");
}

template <class Descriptor>
void check_batch_delivery_timing()
{
    auto const geom = force_geometry<Descriptor>(4);
    auto resolved = resolve<Descriptor>(
        geom, {cylinder("force_batch_timing", 8.0, 8.0)});
    require(resolved.ok(), resolved.error_message().c_str());
    amrex::BoxArray ba(geom.Domain());
    ba.maxSize(amrex::IntVect(AMREX_D_DECL(32, 32, 2)));
    auto dm = force_distribution(ba, geom, true);
    boundary::InternalSolidBoundarySystem<Descriptor> system;
    std::string error;
    require(
        system.define(resolved.resolved, geom, ba, dm, error),
        error.c_str());
    LatticeField<Descriptor> collided("batch_collided", ba, dm);
    LatticeField<Descriptor> completed("batch_completed", ba, dm);

    std::vector<amrex::GpuArray<amrex::Real, 3>> immediate;
    for (int sample = 0; sample < 3; ++sample) {
        fill_validation_fields(collided, completed, sample);
        system.recordInternalForceSample(
            collided, completed, force_step(30 + sample));
        auto batch = system.takeGlobalBoundaryForceBatch();
        require(
            batch.size() == 1 && batch[0].boundaryForces.size() == 1,
            "immediate force delivery mismatch");
        immediate.push_back(batch[0].boundaryForces[0].forceLb);
    }

    for (int sample = 0; sample < 3; ++sample) {
        fill_validation_fields(collided, completed, sample);
        system.recordInternalForceSample(
            collided, completed, force_step(40 + sample));
    }
    auto batched = system.takeGlobalBoundaryForceBatch();
    require(batched.size() == 3, "batched force delivery size mismatch");
    for (int sample = 0; sample < 3; ++sample) {
        require(
            batched[static_cast<std::size_t>(sample)].step.populationStep ==
                    40 + sample &&
                batched[static_cast<std::size_t>(sample)].boundaryForces.size() == 1,
            "batched force stamp/order mismatch");
        require_force_equal(
            batched[static_cast<std::size_t>(sample)].boundaryForces[0].forceLb,
            immediate[static_cast<std::size_t>(sample)],
            "force delivery cadence changed a raw sample");
    }
    require(
        system.takeGlobalBoundaryForceBatch().empty(),
        "force batch delivery did not clear pending samples");
}

template <class Descriptor>
void check_periodic_z_span()
{
    if constexpr (Descriptor::d == 3) {
        auto const twoLayers = run_force_case<Descriptor>(
            force_geometry<Descriptor>(2),
            amrex::IntVect(AMREX_D_DECL(64, 64, 2)),
            false, 2, 60);
        auto const fourSingle = run_force_case<Descriptor>(
            force_geometry<Descriptor>(4),
            amrex::IntVect(AMREX_D_DECL(64, 64, 4)),
            false, 2, 60);
        for (int d = 0; d < 2; ++d) {
            require_force_close(
                fourSingle.force[d], 2.0 * twoLayers.force[d],
                "D3 periodic-z force does not scale with span");
        }
        require_force_close(
            twoLayers.force[2], 0.0,
            "D3 two-layer periodic-z Fz symmetry residual");
        require_force_close(
            fourSingle.force[2], 0.0,
            "D3 four-layer periodic-z Fz symmetry residual");
        require(
            std::abs(fourSingle.force[0]) + std::abs(fourSingle.force[1]) >
                4096.0 * std::numeric_limits<amrex::Real>::epsilon(),
            "D3 span fixture has no in-plane force signal");
    }
}

} // namespace internal_solid_force_validation_test

template <class Descriptor>
void run_internal_solid_force_validation_tests()
{
    using namespace internal_solid_force_validation_test;
    check_decomposition<Descriptor>();
    check_population_order<Descriptor>();
    check_batch_delivery_timing<Descriptor>();
    check_periodic_z_span<Descriptor>();
    amrex::Print() << Descriptor::name()
                   << " internal solid force validation tests passed.\n";
}

} // namespace lbm::tests::integration

#endif
