#ifndef LBM_AMR_PREPAREDAMREVOLUTION_H
#define LBM_AMR_PREPAREDAMREVOLUTION_H

#include "lbm/config/LbmConfig.H"
#include "lbm/amr/RefinementTagging.H"

#include <memory>
#include <utility>
#include <vector>

namespace lbm::amr {
template<class Descriptor, class Profile, class Tagging = StaticTagging> class LbmAmrSimulation;
}

namespace lbm::runtime {
template<class Descriptor, class Selection> class LbmAmrEvolutionPreparation;
template<class Descriptor, class Profile, class Tagging, class Selection>
LbmAmrEvolutionPreparation<Descriptor, Selection> prepare_amr_evolution(
    amr::LbmAmrSimulation<Descriptor, Profile, Tagging> const& simulation,
    config::LbmConfigIntent const& intent, Selection const& selection);
}

namespace lbm::amr {

template<class Collision>
struct LevelEvolution {
    Collision collision;
    config::ResolvedLbmConfig lbm; // This level's local lattice units.
    amrex::Real dtPhys;
    int dtLb;                    // Relative to the finest lattice units.
    int dxLb() const noexcept { return dtLb; } // Acoustic scaling.
};

/** Immutable run dependencies for every allowed level of one hierarchy.
 * Keep this object alive throughout the run; moving it preserves its identity.
 */
template<class Collision>
class PreparedAmrEvolution {
public:
    PreparedAmrEvolution(PreparedAmrEvolution const&) = delete;
    PreparedAmrEvolution& operator=(PreparedAmrEvolution const&) = delete;
    PreparedAmrEvolution(PreparedAmrEvolution&&) noexcept = default;
    PreparedAmrEvolution& operator=(PreparedAmrEvolution&&) = delete;

    int numLevels() const noexcept { return static_cast<int>(levels_.size()); }
    LevelEvolution<Collision> const& level(int lev) const { return levels_.at(lev); }

private:
    PreparedAmrEvolution(std::shared_ptr<void const> hierarchy,
                         std::vector<LevelEvolution<Collision>> levels)
        : hierarchy_(std::move(hierarchy)), identity_(std::make_shared<char const>(0)),
          levels_(std::move(levels))
    {}

    // Tokens retain identity only, never another copy of parameters or fields.
    std::shared_ptr<void const> hierarchy_, identity_;
    std::vector<LevelEvolution<Collision>> levels_;

    template<class, class, class> friend class LbmAmrSimulation;
    template<class D, class P, class T, class S>
    friend runtime::LbmAmrEvolutionPreparation<D, S> runtime::prepare_amr_evolution(
        ::lbm::amr::LbmAmrSimulation<D, P, T> const&, config::LbmConfigIntent const&, S const&);
};

} // namespace lbm::amr
#endif
