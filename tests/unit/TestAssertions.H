#ifndef LBM_TESTS_UNIT_TESTASSERTIONS_H
#define LBM_TESTS_UNIT_TESTASSERTIONS_H

#include <AMReX.H>

namespace lbm::tests {

inline void require(bool condition, char const* message)
{
    if (!condition) {
        amrex::Abort(message);
    }
}

} // namespace lbm::tests

#endif
