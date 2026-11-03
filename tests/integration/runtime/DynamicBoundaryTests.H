#ifndef LBM_TESTS_DYNAMICBOUNDARYTESTS_H
#define LBM_TESTS_DYNAMICBOUNDARYTESTS_H
namespace lbm::tests::dynamic_amr::dynamic_boundary {
template<class Descriptor> void run_force_checks();
template<class Descriptor> void run_relayout_checks();
template<class Descriptor> void run_domain_checks();
}
#endif
