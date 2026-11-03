#ifndef LBM_RUNTIME_LBMAMRPREPARATION_HH
#define LBM_RUNTIME_LBMAMRPREPARATION_HH

#include "lbm/config/LbmConfigResolver.H"

#include <cmath>
#include <limits>

namespace lbm::runtime {

template<class Descriptor, class Profile, class Tagging, class Selection>
LbmAmrEvolutionPreparation<Descriptor, Selection> prepare_amr_evolution(
    amr::LbmAmrSimulation<Descriptor, Profile, Tagging> const& simulation,
    config::LbmConfigIntent const& intent, Selection const& selection)
{
    LbmAmrEvolutionPreparation<Descriptor, Selection> result;
    if (simulation.state_ != decltype(simulation.state_)::Ready) {
        result.errors.push_back("invalid_run_state: prepare requires a finalized hierarchy");
        return result;
    }
    int const levels = simulation.numAllowedLevels();
    if (levels < 1 || levels > std::numeric_limits<int>::digits) {
        result.errors.push_back("step_range: level scales exceed the integer range");
        return result;
    }
    // Successful hierarchy preparation already guarantees the descriptor's
    // isotropic active grid, 2:1 ratios, shared physical domain and nesting.
    auto const& geometry = simulation.geometry(levels - 1);
    amrex::Array<int, 3> cells;
    amrex::Array<amrex::Real, 3> lo, hi;
    for (int d = 0; d < 3; ++d) {
        cells[d] = geometry.Domain().length(d);
        lo[d] = geometry.ProbLo(d);
        hi[d] = geometry.ProbHi(d);
    }
    auto resolution = config::resolve_lbm_config<Descriptor>(intent, cells, lo, hi);
    result.errors = std::move(resolution.errors);
    if (!result.errors.empty()) return result;
    auto const& finest = resolution.resolved;
    if (finest.streamingBackend != "baseline") {
        result.errors.push_back("unsupported_streaming_backend: AMR uses baseline pull streaming");
        return result;
    }
    amrex::Real const fineDt = finest.uRefLb *
        (finest.lengthRefPhys / finest.lengthRefLb) / finest.uRefPhys;
    using Preparation = LbmAmrEvolutionPreparation<Descriptor, Selection>;
    using Collision = typename Preparation::Collision;
    std::vector<amr::LevelEvolution<Collision>> evolution;
    evolution.reserve(levels);
    for (int lev = 0; lev < levels; ++lev) {
        int const scale = 1 << (levels - 1 - lev);
        auto lbm = finest;
        if (scale != 1) {
            lbm.lengthRefLb /= scale;
            lbm.nuLb /= scale;
            lbm.tau = amrex::Real(0.5) + (finest.tau - amrex::Real(0.5)) / scale;
            lbm.omega = 1 / lbm.tau;
            lbm.sources.nuLbDerived = true;
            lbm.sources.tauDerived = true;
        }
        amrex::Real const dt = fineDt * scale;
        if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(lbm.tau) || lbm.tau <= 0.5 ||
            !std::isfinite(lbm.omega) || lbm.omega <= 0 || lbm.omega >= 2 ||
            lbm.nuLb <= 0 || lbm.lengthRefLb <= 0) {
            result.errors.push_back("invalid_level_parameters: level " + std::to_string(lev) +
                                    " has unrepresentable timestep or relaxation parameters");
            return result;
        }
        auto dynamics = selection.template prepare<Descriptor>(lbm);
        if (!dynamics.ok()) {
            for (auto const& error : dynamics.errors)
                result.errors.push_back("level " + std::to_string(lev) + ": " + error);
            return result;
        }
        evolution.push_back({Collision(std::move(*dynamics.dynamics)), std::move(lbm), dt, scale});
    }
    result.evolution_.emplace(typename Preparation::Evolution(
        simulation.hierarchyIdentity_, std::move(evolution)));
    return result;
}

} // namespace lbm::runtime
#endif
