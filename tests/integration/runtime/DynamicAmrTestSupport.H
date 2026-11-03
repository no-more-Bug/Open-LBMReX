#ifndef LBM_TESTS_DYNAMICAMRTESTSUPPORT_H
#define LBM_TESTS_DYNAMICAMRTESTSUPPORT_H

#include "lbm/amr/DynamicRefinement.H"
#include "lbm/initialization/MacroscopicState.H"

#include <AMReX.H>
#include <cmath>
#include <string>

namespace lbm::tests::dynamic_amr {
using Real = amrex::Real;

inline void require(bool ok, std::string const& message)
{
    if (!ok) amrex::Abort(message);
}

inline void close(Real actual, Real expected, std::string const& message)
{
    require(std::isfinite(actual) && std::abs(actual-expected) < 2.e-12, message);
}

template<class Descriptor>
amr::DynamicHierarchySpec base_spec(int maxLevel = 2)
{
    amr::DynamicHierarchySpec spec;
    int const n = Descriptor::d == 3 ? 16 : 32;
    spec.nCell = {n,n,Descriptor::d == 3 ? n : 1};
    spec.probHi = {1,1,Descriptor::d == 3 ? 1 : 0.125};
    spec.maxGridSize = {amrex::IntVect(8,8,Descriptor::d == 3 ? 8 : 1)};
    spec.refRatio = amrex::IntVect(2,2,Descriptor::d == 3 ? 2 : 1);
    spec.maxLevel = maxLevel;
    return spec;
}
} // namespace lbm::tests::dynamic_amr
#endif
