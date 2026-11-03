#ifndef LBM_TESTS_NUMERICAL_RE100CASE_H
#define LBM_TESTS_NUMERICAL_RE100CASE_H

#include "Re20Case.H"

namespace amr_re100 {
using amr_re20::Real;
using amr_re20::Initial;
using amr_re20::meanSpeed;
using amr_re20::boundaries;

inline lbm::amr::StaticHierarchySpec geometry(int dimensions, std::string const& layout, int maxBox)
{
    return amr_re20::geometry(dimensions, layout, maxBox, layout == "coarse" ? 40 : 20);
}
} // namespace amr_re100
#endif
