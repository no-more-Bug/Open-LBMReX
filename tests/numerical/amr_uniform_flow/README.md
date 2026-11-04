# Static AMR uniform flow

A three-level, cell-centered BGK composition uses Guo velocity inlet/outlet with periodic y/z and tests uniform-state preservation and conservation across coarse/fine interfaces. D2Q9 uses Nz=1; D3Q27 is a quasi-two-dimensional counterpart. Inputs are [d2q9](inputs/d2q9) and [d3q27](inputs/d3q27).

## Build and engineering entry

From an appropriately configured repository root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_numerical_amr_uniform_flow lbm_numerical_amr_uniform_flow_d3q27
LBM_TEST_MAX_STEP=8 LBM_TEST_ENGINEERING=1 LBM_TEST_AMR_MPI_RANKS=4 \
bash tests/numerical/amr_uniform_flow/run.sh \
  build/cpu/tests/numerical/lbm_numerical_amr_uniform_flow \
  build/cpu/tests/numerical/lbm_numerical_amr_uniform_flow_d3q27 \
  build/cpu/tests/numerical/work/manual-amr-uniform \
  "$(command -v mpirun)" d2
```

This is an explicit eight-step engineering check, with one OpenMP thread per rank. Use an appropriate allocated CPU/MPI environment. The runner obtains a strict source identity as described in the [common contract](../README.md). The selected CPU/CUDA eight-step routes and the CPU source-archive route have [standalone evidence](../../../docs/validation.md); full scientific runs are separate.

## Complete comparison and artifacts

The complete identity ends at physical t=100, corresponding to 32000 L0 steps. Its prescribed relative velocity-L2 and mass-drift budgets are 1e-8, with quasi-two-dimensional quantities bounded by 1e-10. The engineering mode checks only its actual bounded output, not completion of this horizon.

Artifacts use `amr-uniform-v1` and include `identity.txt`, layout/sampling records and native `sync-*` outputs. [check_uniform_flow.py](check_uniform_flow.py) validates identity, geometry, source and completeness before the applicable numerical comparisons. Inputs expose stable controls; topology/composition and independent acceptance stay in their code owners. Do not use short-run exit 0 as a long-time scientific pass.
