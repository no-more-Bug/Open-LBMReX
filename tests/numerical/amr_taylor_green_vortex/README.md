# Static AMR Taylor–Green vortex

This periodic BGK Re=10 case compares center-refined AMR against explicit matching uniform-fine results. L0 resolutions are 20/40/80 and the corresponding uniform grids are 40/80/160. Both D2Q9 and quasi-two-dimensional D3Q27 reach physical t=0.2 for the complete comparison.

Targets are `lbm_numerical_amr_tgv` and `lbm_numerical_amr_tgv_d3q27`. Inputs are [d2q9](inputs/d2q9) and [d3q27](inputs/d3q27); entries are `{d2,d3}-{20,40,80}-{uniform,amr}`.

## Bounded uniform-reference entry

Configure from the root using [building](../../../docs/building.md), then:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_numerical_amr_tgv lbm_numerical_amr_tgv_d3q27
LBM_TEST_MAX_STEP=8 LBM_TEST_ENGINEERING=1 LBM_TEST_AMR_MPI_RANKS=4 \
bash tests/numerical/amr_taylor_green_vortex/run.sh \
  build/cpu/tests/numerical/lbm_numerical_amr_tgv \
  build/cpu/tests/numerical/lbm_numerical_amr_tgv_d3q27 \
  build/cpu/tests/numerical/work/manual-tgv-short \
  "$(command -v mpirun)" d2-20-uniform
```

This produces engineering evidence only. A scientific AMR comparison needs a complete eligible uniform-reference work directory as the explicit final argument. Missing or incompatible references are not silently recomputed.

## Acceptance and output

The global velocity L2 error is divided by the current analytical velocity L2 norm; interface peak error is normalized by U0. The complete AMR comparison uses relative velocity L2 ≤5% (≤2% for the finest case), peak error in the actual interface band of width L/16 ≤5%, and conservation/quasi-two-dimensional quantities ≤1e-10. AMR convergence order is reported; it is not required to be second order. Reference eligibility and matching source/descriptor/time are part of the result.

Artifacts use `amr-tgv-v1`, with `identity.txt`, layout/sampling records and native `sync-*` frames. [check_tgv.py](check_tgv.py) and [tgv_metrics.py](tgv_metrics.py) own the comparison; [summarize_convergence.py](summarize_convergence.py) consumes explicit outputs. See the [common status contract](../README.md) and [pending snapshot validation](../../../docs/validation.md).
