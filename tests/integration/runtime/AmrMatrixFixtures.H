#ifndef LBM_TESTS_AMRMATRIXFIXTURES_H
#define LBM_TESTS_AMRMATRIXFIXTURES_H

#include "integration/runtime/AmrEvolutionTestCase.H"

namespace lbm::tests::amr_evolution {
// E3/E5/E6 additions to the E1/E2/E4 cases already exercised by C4.
std::vector<Fixture> matrix_fixtures();
Fixture retained_hole_fixture();
Fixture through_fixture(int box);
}
#endif
