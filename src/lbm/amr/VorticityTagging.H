#ifndef LBM_AMR_VORTICITYTAGGING_H
#define LBM_AMR_VORTICITYTAGGING_H

#include "lbm/amr/RefinementTagging.H"

#include <cmath>
#include <string>
#include <vector>

namespace lbm::amr {

struct VorticityThresholds {
    amrex::Real refine = 20;
    amrex::Real keep = 10;
};

struct VorticityTaggingRule {
    VorticityThresholds thresholds;
    amrex::Real normalization;

    template<class Descriptor>
    AMREX_GPU_HOST_DEVICE
    bool operator()(TaggingCell<Descriptor> const& cell) const noexcept
    {
        auto const velocity = cell.velocityLb;
        int const i = cell.populations.i, j = cell.populations.j, k = cell.populations.k;
        auto const inverseDx = cell.inverseCellSizePhys;
        amrex::Real const wz = amrex::Real(0.5) *
            ((velocity(i+1,j,k,1)-velocity(i-1,j,k,1))*inverseDx[0] -
             (velocity(i,j+1,k,0)-velocity(i,j-1,k,0))*inverseDx[1]);
        amrex::Real magnitude = std::abs(wz);
        if constexpr (Descriptor::d == 3) {
            amrex::Real const wx = amrex::Real(0.5) *
                ((velocity(i,j+1,k,2)-velocity(i,j-1,k,2))*inverseDx[1] -
                 (velocity(i,j,k+1,1)-velocity(i,j,k-1,1))*inverseDx[2]);
            amrex::Real const wy = amrex::Real(0.5) *
                ((velocity(i,j,k+1,0)-velocity(i,j,k-1,0))*inverseDx[2] -
                 (velocity(i+1,j,k,2)-velocity(i-1,j,k,2))*inverseDx[0]);
            magnitude = std::sqrt(wx*wx + wy*wy + wz*wz);
        }
        amrex::Real const indicator = normalization * magnitude;
        return indicator > thresholds.refine ||
            (cell.coveredByFine && indicator > thresholds.keep);
    }
};

/** Fixed physical references; defaults are engineering controls, not paper values. */
class VorticityTagging {
public:
    static constexpr bool needsVelocity = true;
    std::vector<VorticityThresholds> thresholds{{}};

    VorticityTagging(amrex::Real lengthRefPhys, amrex::Real velocityRefPhys,
                    amrex::Real velocityLbToPhys)
        : lengthRefPhys_(lengthRefPhys), velocityRefPhys_(velocityRefPhys),
          velocityLbToPhys_(velocityLbToPhys)
    {}

    std::vector<std::string> validate(int maxLevel) const
    {
        std::vector<std::string> errors;
        if (!std::isfinite(lengthRefPhys_) || lengthRefPhys_ <= 0 ||
            !std::isfinite(velocityRefPhys_) || velocityRefPhys_ <= 0 ||
            !std::isfinite(velocityLbToPhys_) || velocityLbToPhys_ <= 0 ||
            !std::isfinite(normalization()) || normalization() <= 0)
            errors.push_back("invalid_vorticity_reference: finite positive physical references and velocity conversion required");
        if (thresholds.size() != 1 && thresholds.size() != static_cast<std::size_t>(maxLevel))
            errors.push_back("invalid_vorticity_thresholds: supply one pair or one per taggable level");
        for (auto const& pair : thresholds) {
            if (!std::isfinite(pair.refine) || !std::isfinite(pair.keep) ||
                pair.keep < 0 || pair.keep >= pair.refine)
                errors.push_back("invalid_vorticity_thresholds: require finite 0 <= keep < refine");
        }
        return errors;
    }

    VorticityTaggingRule rule(int level) const
    {
        return {thresholds[thresholds.size() == 1 ? 0 : level], normalization()};
    }

private:
    amrex::Real normalization() const { return lengthRefPhys_ / velocityRefPhys_ * velocityLbToPhys_; }
    amrex::Real lengthRefPhys_, velocityRefPhys_, velocityLbToPhys_;
};

} // namespace lbm::amr
#endif
