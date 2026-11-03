#ifndef LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCECOLLECTOR_HH
#define LBM_BOUNDARY_INTERNAL_FORCE_INTERNALSOLIDFORCECOLLECTOR_HH

#include <AMReX_BLassert.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Reduce.H>

#include <limits>
#include <utility>

namespace lbm::boundary {

namespace detail {

template <class Descriptor>
amrex::GpuArray<amrex::Real, 3> reduce_internal_boundary_force(
    InternalBoundaryMetadataStorage const& storage,
    LatticeField<Descriptor> const& collidedSource,
    LatticeField<Descriptor> const& completedPopulation,
    int boundaryId)
{
    amrex::ReduceOps<
        amrex::ReduceOpSum,
        amrex::ReduceOpSum,
        amrex::ReduceOpSum> ops;
    amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real> data(ops);
    using ForceTuple = typename decltype(data)::Type;
    const InternalBoundaryMetadataView host = storage.hostView();
    const InternalBoundaryMetadataView device = storage.deviceView();
    auto const& collidedMf = collidedSource.multifab();
    auto const& completedMf = completedPopulation.multifab();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(collidedMf); mfi.isValid(); ++mfi) {
        const int grid = mfi.index();
        const amrex::Long begin = host.begin(grid);
        const amrex::Long count = host.end(grid) - begin;
        if (count == 0) {
            continue;
        }
        auto const collided = collidedMf.const_array(mfi);
        auto const completed = completedMf.const_array(mfi);
        ops.eval(
            count,
            data,
            [=] AMREX_GPU_DEVICE(amrex::Long n) noexcept -> ForceTuple {
                auto const& execution = device.execution[begin + n];
                auto const& canonical =
                    device.canonical[execution.canonicalIndex];
                if (canonical.boundaryId != boundaryId) {
                    return {0.0, 0.0, 0.0};
                }
                const int i = execution.anchorCell[0];
                const int j = execution.anchorCell[1];
                const int k = execution.anchorCell[2];
                auto const force = static_internal_solid_link_force<Descriptor>(
                    collided(i, j, k, execution.direction),
                    completed(i, j, k, execution.oppositeDirection),
                    execution.direction);
                return {force[0], force[1], force[2]};
            });
    }
    ForceTuple const local = data.value(ops);
    return {
        amrex::get<0>(local), amrex::get<1>(local),
        amrex::get<2>(local)};
}

} // namespace detail

template <class Descriptor>
bool InternalSolidForceCollector<Descriptor>::define(
    InternalBoundaryMetadataStorage const& storage,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::vector<BoundaryId> const& ownedBoundaryIds,
    std::string& error)
{
    reset();
    error.clear();
    if (ownedBoundaryIds.empty() ||
        ownedBoundaryIds.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max() / 3) ||
        ba.size() != dm.size() || storage.hostView().gridCount != ba.size()) {
        error = "internal force collector setup contract failed";
        return false;
    }
    storage_ = &storage;
    boxArray_ = ba;
    distributionMap_ = dm;
    ownedBoundaryIds_ = ownedBoundaryIds;
    return true;
}

template <class Descriptor>
void InternalSolidForceCollector<Descriptor>::reset() noexcept
{
    storage_ = nullptr;
    boxArray_ = amrex::BoxArray{};
    distributionMap_ = amrex::DistributionMapping{};
    ownedBoundaryIds_.clear();
    pendingSteps_.clear();
    pendingForces_.clear();
}

template <class Descriptor>
void InternalSolidForceCollector<Descriptor>::recordInternalForceSample(
    LatticeField<Descriptor> const& collidedSource,
    LatticeField<Descriptor> const& completedPopulation,
    InternalForceStep const& step)
{
    auto const& collidedMf = collidedSource.multifab();
    auto const& completedMf = completedPopulation.multifab();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        storage_ != nullptr && !ownedBoundaryIds_.empty() && step.populationStep >= 0 &&
            &collidedSource != &completedPopulation &&
            collidedMf.boxArray() == boxArray_ &&
            completedMf.boxArray() == boxArray_ &&
            collidedMf.DistributionMap() == distributionMap_ &&
            completedMf.DistributionMap() == distributionMap_ &&
            (pendingSteps_.empty() || step.populationStep > pendingSteps_.back().populationStep),
        "InternalSolidForceCollector::record contract failed");

    const auto begin = pendingForces_.size();
    pendingForces_.resize(begin + 3 * ownedBoundaryIds_.size());
    for (std::size_t index = 0; index < ownedBoundaryIds_.size(); ++index) {
        auto const force = detail::reduce_internal_boundary_force(
            *storage_, collidedSource, completedPopulation, ownedBoundaryIds_[index].value);
        for (int d = 0; d < 3; ++d) pendingForces_[begin + 3 * index + d] = force[d];
    }
    pendingSteps_.push_back(step);
}

template <class Descriptor>
std::vector<InternalBoundaryForceValues>
InternalSolidForceCollector<Descriptor>::takeGlobalBoundaryForceBatch()
{
    if (pendingSteps_.empty()) {
        return {};
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        pendingForces_.size() ==
                pendingSteps_.size() * 3 * ownedBoundaryIds_.size() &&
            pendingForces_.size() <=
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "InternalSolidForceCollector pending batch contract failed");

    amrex::ParallelDescriptor::ReduceRealSum(
        pendingForces_.data(), static_cast<int>(pendingForces_.size()));
    std::vector<InternalBoundaryForceValues> batch;
    batch.reserve(pendingSteps_.size());
    for (std::size_t sampleIndex = 0;
         sampleIndex < pendingSteps_.size(); ++sampleIndex) {
        InternalBoundaryForceValues sample;
        sample.step = pendingSteps_[sampleIndex];
        sample.boundaryForces.reserve(ownedBoundaryIds_.size());
        for (std::size_t index = 0; index < ownedBoundaryIds_.size(); ++index) {
            const auto offset = (sampleIndex * ownedBoundaryIds_.size() + index) * 3;
            sample.boundaryForces.push_back({ownedBoundaryIds_[index],
                {pendingForces_[offset], pendingForces_[offset + 1], pendingForces_[offset + 2]}});
        }
        batch.push_back(std::move(sample));
    }
    pendingSteps_.clear();
    pendingForces_.clear();
    return batch;
}

} // namespace lbm::boundary

#endif
