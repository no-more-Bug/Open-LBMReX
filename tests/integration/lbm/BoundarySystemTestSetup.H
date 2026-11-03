#ifndef LBM_TESTS_INTEGRATION_LBM_BOUNDARYSYSTEMTESTSETUP_H
#define LBM_TESTS_INTEGRATION_LBM_BOUNDARYSYSTEMTESTSETUP_H

#include "lbm/boundary/core/BoundarySystem.H"

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>

#include <string>
#include <vector>

namespace lbm::tests::integration {

template <class Descriptor>
void define_boundary_system_for_test(
    boundary::BoundarySystem<Descriptor>& system,
    std::vector<boundary::DomainBoundarySpec> const& domainSpecs,
    amrex::Geometry const& geom,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm,
    std::vector<boundary::InternalBoundarySpec> const& internalSpecs = {})
{
    std::string error;
    if (!system.tryInitialize(domainSpecs, internalSpecs, geom, error, -1) ||
        !system.tryBindLayout(ba, dm, error, -1)) {
        amrex::Abort(error);
    }
}

} // namespace lbm::tests::integration

#endif
