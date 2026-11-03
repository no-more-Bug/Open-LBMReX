#ifndef LBM_TESTS_DYNAMICCONSTRAINTLIFECYCLETESTS_H
#define LBM_TESTS_DYNAMICCONSTRAINTLIFECYCLETESTS_H

namespace lbm::tests::dynamic_amr::constraint_lifecycle {
template<class Descriptor> void run_tagging_checks();
template<class Descriptor> void run_initialization_checks();
template<class Descriptor> void run_regrid_checks();
}
#endif
