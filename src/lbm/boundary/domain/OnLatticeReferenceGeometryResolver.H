#ifndef LBM_BOUNDARY_DOMAIN_ONLATTICEREFERENCEGEOMETRYRESOLVER_H
#define LBM_BOUNDARY_DOMAIN_ONLATTICEREFERENCEGEOMETRYRESOLVER_H

#include "lbm/boundary/domain/LbmPostStreamingBoundary.H"
#include "lbm/config/LbmConfig.H"

#include <string>
#include <vector>

namespace lbm::boundary {

struct OnLatticeReferenceGeometryResolveResult {
    int axis = -1;
    std::vector<std::string> errors;

    bool ok() const noexcept { return errors.empty(); }
};

inline OnLatticeReferenceGeometryResolveResult
resolve_lbm_reference_geometry(
    PostStreamingBoundaryPlan const& plan,
    config::ResolvedLbmConfig const& lbm)
{
    OnLatticeReferenceGeometryResolveResult result;
    if (!plan.hasPostStreamingFaces) {
        return result;
    }

    result.axis = matching_on_lattice_reference_axis(
        plan, lbm.lengthRefLb, lbm.lengthRefPhys);
    if (result.axis < 0) {
        result.errors.push_back(
            "post-stream valid-cell boundaries require "
            "lbm.length_ref_lb/phys to match one active on-lattice physical "
            "interval: length_ref_lb=N and length_ref_phys=N*dx, not the "
            "N+1-cell AMReX envelope");
    }
    return result;
}

} // namespace lbm::boundary

#endif
