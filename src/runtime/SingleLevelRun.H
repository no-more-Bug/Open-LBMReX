#ifndef LBM_RUNTIME_SINGLELEVELRUN_H
#define LBM_RUNTIME_SINGLELEVELRUN_H

#include "runtime/SingleLevelPreparation.H"
#include "runtime/SingleLevelRunReport.H"

namespace lbm::runtime {

/** Consume one prepared token and execute the fixed single-level lifecycle. */
template <class Descriptor, class Dynamics, class Initialization, class RunSession>
SingleLevelRunReport run_single_level(
    PreparedSingleLevelRun<Descriptor, Dynamics, Initialization, RunSession> prepared);

} // namespace lbm::runtime

#include "runtime/SingleLevelRun.HH"

#endif
