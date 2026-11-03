#ifndef LBM_BOUNDARY_CORE_BOUNDARYSYSTEM_H
#define LBM_BOUNDARY_CORE_BOUNDARYSYSTEM_H

#include "lbm/boundary/core/BoundaryDiagnostics.H"
#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/domain/DomainBoundarySystem.H"
#include "lbm/boundary/internal/InternalSolidBoundarySystem.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_REAL.H>

#include <string>
#include <utility>
#include <vector>

namespace lbm::boundary {

/**
 * @brief Optional local timing breakdown for the pre-stream boundary stage.
 *
 * Values are accumulated on the calling rank. Supplying the sample introduces
 * GPU stream synchronization around measured substeps; MPI reduction and the
 * full-stage timing boundary remain solver responsibilities.
 */
struct PreStreamingTimingSample {
    amrex::Real fillBoundary = 0.0;
    amrex::Real domainSetup = 0.0;
    amrex::Real physicalGhosts = 0.0;
};

/**
 * Optional rank-local timing breakdown for post-stream reconstruction.
 * Supplying a sample synchronizes the active GPU stream after each sub-stage;
 * MPI reduction and the outer PostStream timer remain solver responsibilities.
 */
struct PostStreamingTimingSample {
    amrex::Real domain = 0.0;
    amrex::Real internalSolid = 0.0;
};

/**
 * @brief Application-facing owner of resolved and compiled LBM boundaries.
 *
 * Initialization is a host-side, once-per-geometry operation. Thereafter the
 * object owns immutable resolved metadata plus DomainBoundarySystem runtime
 * tables/plans. All ranks must call communication-capable stage methods in the
 * same order. The class does not advance time: each supplied time is forwarded
 * unchanged as the boundary-profile sampling time selected by the solver.
 */
template <class Descriptor>
class BoundarySystem {
public:
    /**
     * @brief Result-based production setup used by deep runtime preparation.
     *
     * For a fresh setup object, expected specification errors return before
     * runtime state is installed. Deep preparation discards that local owner
     * on failure. Domain and internal specifications are resolved together so
     * cross-source names and object ownership remain one setup contract.
     */
    bool tryInitialize(
        std::vector<DomainBoundarySpec> const& domainSpecs,
        std::vector<InternalBoundarySpec> const& internalSpecs,
        amrex::Geometry const& geom,
        std::string& error,
        int verbose = 0)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            state_ == AssemblyState::Empty,
            "BoundarySystem setup cannot be reused after successful initialization");
        BoundaryResolveResult resolved =
            resolve_boundary_specs<Descriptor>(domainSpecs, internalSpecs, geom);
        if (!resolved.ok()) {
            error = resolved.error_message();
            return false;
        }
        if (!domainSystem_.tryDefine(
                geom, resolved.resolved.domain, error)) {
            return false;
        }
        internalSystem_.reset();
        geom_ = geom;
        resolved_ = std::move(resolved.resolved);
        state_ = AssemblyState::Resolved;
        requestedVerbose_ = verbose;
        error.clear();
        return true;
    }

    /**
     * @brief Result-based layout binding for deep preparation.
     *
     * Expected metadata failure is reported to the caller, which discards the
     * local BoundarySystem instead of observing partially compiled metadata.
     */
    bool tryBindLayout(
        amrex::BoxArray const& ba,
        amrex::DistributionMapping const& dm,
        std::string& error,
        int verbose = -1)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            state_ == AssemblyState::Resolved,
            "BoundarySystem::tryBindLayout requires the Resolved state");
        std::string detail;
        if (!internalSystem_.define(resolved_, geom_, ba, dm, detail)) {
            error = "Internal solid metadata setup failed:\n  - " + detail;
            return false;
        }
        state_ = AssemblyState::MetadataReady;
        const int effectiveVerbose = verbose >= 0 ? verbose : requestedVerbose_;
        if (effectiveVerbose >= 0) {
            writeDiagnostics();
        }
        error.clear();
        return true;
    }

    /**
     * @brief Synchronize population ghosts and then fill physical HBB ghosts.
     *
     * The input field is the collided source layer for the pending pull. AMReX
     * FillBoundary first supplies same-level/periodic valid-to-ghost copies;
     * PhysBC then writes only physical ghosts outside the valid+periodic grown
     * domain. Post-stream valid-node boundaries are not modified here.
     *
     * @param physicalTime Source physical time of the pending step.
     */
    template <class PopulationField>
    void fillPreStreaming(PopulationField& populations, amrex::Real physicalTime) const
    {
        fillPreStreaming(populations, physicalTime, nullptr);
    }

    /**
     * @brief Timed overload of fillPreStreaming().
     *
     * The call remains rank-symmetric even though physical ghost work is local.
     * The optional timings are local measurements and are not MPI collectives.
     */
    template <class PopulationField>
    void fillPreStreaming(
        PopulationField& populations,
        amrex::Real physicalTime,
        PreStreamingTimingSample* timing) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            metadataReady(),
            "BoundarySystem::fillPreStreaming called before tryBindLayout");

        auto timed = [timing](amrex::Real PreStreamingTimingSample::* member, auto&& f) {
            if (timing == nullptr) {
                f();
                return;
            }
            const amrex::Real start = static_cast<amrex::Real>(
                amrex::ParallelDescriptor::second());
            f();
            amrex::Gpu::streamSynchronize();
            timing->*member += static_cast<amrex::Real>(
                amrex::ParallelDescriptor::second()) - start;
        };

        timed(&PreStreamingTimingSample::fillBoundary, [&] {
            populations.fillBoundary(geom_);
        });
        timed(&PreStreamingTimingSample::physicalGhosts, [&] {
            domainSystem_.fillPhysicalGhosts(populations, physicalTime);
        });
    }

    /**
     * @brief Initialize on-lattice valid boundary nodes at the initial time.
     *
     * HBB-only configurations are a no-op because their physical boundary lies
     * on a valid/ghost face. For current Guo configurations this entry uses the
     * compiled valid-node regions after bulk populations have been initialized,
     * completing boundary macros from prescribed data and inward donors before
     * writing equilibrium populations only.
     *
     * @param physicalTime Initial physical time for boundary-profile sampling.
     */
    template <class PopulationField>
    void initializeValidBoundaries(
        PopulationField& populations,
        amrex::Real physicalTime) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            metadataReady(),
            "BoundarySystem::initializeValidBoundaries called before tryBindLayout");
        domainSystem_.initializeValidBoundaries(populations, physicalTime);
    }

    /**
     * @brief Complete domain and internal post-stream valid-node reconstruction.
     *
     * collidedSource is the collision-complete source whose ghosts were filled
     * by fillPreStreaming(). streamedDestination already contains full-valid
     * pull output. Domain reconstruction runs first and may synchronize its dst
     * donors symmetrically; internal reconstruction then reads only src and
     * writes its disjoint destination components without communication.
     *
     * @param physicalTime Target physical time of the completed pull.
     */
    void applyPostStreamingValidNodes(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor>& streamedDestination,
        amrex::Real physicalTime) const
    {
        applyPostStreamingValidNodes(
            collidedSource, streamedDestination, physicalTime, nullptr);
    }

    /** Timed overload; sub-stage measurements remain local and optional. */
    void applyPostStreamingValidNodes(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor>& streamedDestination,
        amrex::Real physicalTime,
        PostStreamingTimingSample* timing) const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            metadataReady(),
            "BoundarySystem::applyPostStreamingValidNodes called before tryBindLayout");

        auto timed = [timing](
                         amrex::Real PostStreamingTimingSample::* member,
                         auto&& operation) {
            if (timing == nullptr) {
                operation();
                return;
            }
            const amrex::Real start = static_cast<amrex::Real>(
                amrex::ParallelDescriptor::second());
            operation();
            amrex::Gpu::streamSynchronize();
            timing->*member += static_cast<amrex::Real>(
                amrex::ParallelDescriptor::second()) - start;
        };

        timed(&PostStreamingTimingSample::domain, [&] {
            domainSystem_.applyPostStreamingValidNodes(
                streamedDestination, physicalTime);
        });
        timed(&PostStreamingTimingSample::internalSolid, [&] {
            internalSystem_.applyPostStreamingValidNodes(
                collidedSource, streamedDestination, physicalTime);
        });
    }

    /** @brief Record boundary force with the caller's complete physical step label. */
    void recordInternalForceSample(
        LatticeField<Descriptor> const& collidedSource,
        LatticeField<Descriptor> const& completedPopulation,
        InternalForceStep const& step)
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            metadataReady(),
            "BoundarySystem::record force called before tryBindLayout");
        internalSystem_.recordInternalForceSample(
            collidedSource, completedPopulation, step);
    }

    /** @brief Publish single-level Boundary/Object samples from one raw reduction. */
    [[nodiscard]] InternalForceBatch
    takeGlobalInternalForceBatch()
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            metadataReady(),
            "BoundarySystem::take force batch called before tryBindLayout");
        auto raw = internalSystem_.takeGlobalBoundaryForceBatch();
        InternalForceBatch batch;
        std::vector<ObjectId> objects;
        for (auto const& object : resolved_.internalObjects) objects.push_back(object.objectId);
        for (auto& sample : raw) {
            batch.objectSteps.push_back({sample.step,
                sum_internal_object_forces(sample.boundaryForces, resolved_, objects)});
            batch.boundarySteps.push_back({sample.step, std::move(sample.boundaryForces)});
        }
        return batch;
    }

    bool hasPostStreamingBoundaries() const noexcept
    {
        return domainSystem_.hasPostStreamingBoundaries() ||
            internalSystem_.hasPostStreamingBoundaries();
    }

    /** @brief Borrow the compiled plan for this one-shot setup lifetime. */
    PostStreamingBoundaryPlan const& postStreamingPlan() const noexcept
    {
        return domainSystem_.postStreamingPlan();
    }

    void writeDiagnostics() const
    {
        if (!metadataReady()) {
            return;
        }
        print_boundary_diagnostics(resolved_, geom_);
        amrex::Print() << internalSystem_.formatDiagnostics();
    }

    [[nodiscard]] bool metadataReady() const noexcept
    {
        return state_ == AssemblyState::MetadataReady;
    }

    /** @brief Borrow canonical setup metadata; no population state is exposed. */
    ResolvedBoundarySet const& resolved() const noexcept { return resolved_; }

private:
    enum class AssemblyState {
        Empty,
        Resolved,
        MetadataReady
    };

    AssemblyState state_ = AssemblyState::Empty;
    int requestedVerbose_ = -1;
    amrex::Geometry geom_;
    ResolvedBoundarySet resolved_;
    DomainBoundarySystem<Descriptor> domainSystem_;
    InternalSolidBoundarySystem<Descriptor> internalSystem_;
};

} // namespace lbm::boundary

#endif
