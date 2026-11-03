#ifndef LBM_AMR_AMRBOUNDARYSYSTEM_H
#define LBM_AMR_AMRBOUNDARYSYSTEM_H

#include "lbm/amr/PreparedAmrEvolution.H"
#include "lbm/boundary/core/BoundaryResolver.H"
#include "lbm/boundary/domain/DomainBoundarySystem.H"
#include "lbm/boundary/internal/InternalSolidBoundarySystem.H"

#include <memory>
#include <string>
#include <vector>

namespace lbm::amr {

struct AmrBoundarySpec {
    std::vector<boundary::DomainBoundarySpec> domain;
    std::vector<boundary::InternalBoundarySpec> internal;
    bool collectForces = false;
};

/** Layouts borrowed only while preparing the final static hierarchy. */
struct BoundaryLevelLayout {
    amrex::Geometry const& geometry;
    amrex::BoxArray const& boxes;
    amrex::DistributionMapping const& mapping;
    amrex::IntVect ratioToFiner;
};

/** Own global identities and independent per-level runtime boundary records. */
template <class Descriptor>
class AmrBoundarySystem {
public:
    using Field = LatticeField<Descriptor>;

    bool tryDefine(AmrBoundarySpec const& spec,
                   std::vector<BoundaryLevelLayout> const& levels,
                   std::string& error);
    [[nodiscard]] boundary::ResolvedBoundarySet const& resolved() const noexcept { return resolved_; }
    [[nodiscard]] int internalBoundaryLevel(boundary::BoundaryId id) const noexcept;
    void initializeValidBoundaries(Field& coarsest, amrex::Real physicalTime) const;
    void fillPhysicalGhosts(Field& coarsest, amrex::Real physicalTime) const;
    void applyPostStreamingValidNodes(int level, Field const& collidedSource,
                                     Field& destination, amrex::Real physicalTime) const;
    void recordInternalForceSample(boundary::InternalForceStep const& step,
                                   Field const& collidedSource, Field const& completedPopulation);
    template<class Collision>
    boundary::InternalForceBatch takeGlobalInternalForceBatch(
        PreparedAmrEvolution<Collision> const& evolution, int coarsestStep,
        amrex::Real beginPhysicalTime, amrex::Real endPhysicalTime);

private:
    boundary::ResolvedBoundarySet resolved_;
    std::vector<int> internalLevels_;
    boundary::DomainBoundarySystem<Descriptor> domain_;
    std::vector<std::unique_ptr<boundary::InternalSolidBoundarySystem<Descriptor>>> internal_;
    bool collectForces_ = false;
};

} // namespace lbm::amr

#include "lbm/amr/AmrBoundarySystem.HH"
#endif
