# Cylinder, tandem and experimental dynamic-AMR tests

These BGK tests use DFG-derived channel/cylinder geometry, Guo physical boundaries and explicit YMS/Bouzidi internal-boundary compositions. The constant-density LBM outlet is not the original DFG do-nothing discretization. Scientific references and package/source identities remain explicit.

## Static Re20 and Re100

| Identity | Targets and inputs | Complete comparison |
| --- | --- | --- |
| Re20 | `lbm_numerical_amr_channel_cylinder{,_d3q27}`; `inputs/d2q9` / `inputs/d3q27`; entries `{d2,d3}-{yms,bouzidi}-{coarse,fine,amr}`. | Uniform coarse ND20, fine ND80, and AMR ND20/40/80; full physical t=70; `re20-v1` artifacts. |
| Re100 | `lbm_numerical_amr_re100{,_d3q27}`; `inputs/re100_d2q9` / `inputs/re100_d3q27`; corresponding descriptor/scheme/layout entries. | Full physical t=35; `amr-re100-v1`; only eligible references and sufficient cycles allow scientific interpretation. |

Configure and build from the [root](../../../docs/building.md), not this directory. [run.sh](run.sh) requests the full Re20 horizon and does not honor `LBM_TEST_MAX_STEP` as a generic smoke switch. Coarse/AMR Re20 entries consume the explicit fine work directory.

[run_re100.sh](run_re100.sh) supports bounded engineering mode. For example, after building both Re100 targets:

```bash
LBM_TEST_MAX_STEP=8 LBM_TEST_ENGINEERING=1 LBM_TEST_AMR_MPI_RANKS=4 \
bash tests/numerical/amr_channel_cylinder/run_re100.sh \
  build/cpu/tests/numerical/lbm_numerical_amr_re100 \
  build/cpu/tests/numerical/lbm_numerical_amr_re100_d3q27 \
  build/cpu/tests/numerical/work/manual-re100-short \
  "$(command -v mpirun)" d2-yms-fine
```

This is not a full Re100 scientific run. CPU quasi-D3 engineering availability must not be confused with a complete scientific registration or completed long-run evidence.

For Re20 fine-to-DFG comparison, Cd, Cl and normalized pressure-difference budgets are 1%, absolute 0.002 and 3%. AMR compares against an eligible fine reference using those budgets plus velocity-line RMS 1%/peak 3%; the last three flux differences must be ≤1e-3. Cylinder AMR reference qualification includes the fine run's external literature gates, unlike cavity's steady-reference qualification.

Re100 retains scientific failures and insufficient-reference/cycle evidence; some CUDA quasi-D3 fine runs remain deferred. Preserve reference eligibility, interval force semantics, per-cycle criteria and all invalid/incomplete distinctions in [check_re100.py](check_re100.py). Do not infer combined acceptance from one successful coefficient or Strouhal statistic.

## Tandem objects

Targets `lbm_tandem_cylinder_integration{,_d3q27}` and [run_tandem.sh](run_tandem.sh) perform fixed eight-L0-step structural checks of distinct/shared ObjectId grouping, with an explicit matching distinct run where required. Artifacts use `tandem-v1`.

[run_tandem_compare.sh](run_tandem_compare.sh) is a separate fixed 20000-step D2/YMS/shared-object comparison for `same|cross` layouts, with `tandem-compare-v1` artifacts. It does not provide an arbitrary short-run control or a steady/literature similarity gate.

## Experimental dynamic boundaries

Targets `lbm_dynamic_cylinder_d2` / `lbm_dynamic_cylinder_d3` use the relevant `inputs/dynamic_*` identities. [run_dynamic.sh](run_dynamic.sh) accepts entries `{d2,d3}-{single,tandem-a,tandem-b}` and `LBM_TEST_MAX_STEP` for its short-data mode.

The full Re100 dynamic identity allows Lmax=2 and uses the YMS/Guo composition, 200000 L0 steps, physical t=35 and 141 frames. Current refine thresholds are 2/4, keep thresholds 1/2, and grid efficiency 1.0. `dynamic-cylinder-v1` artifacts include every frame's `layout-STEP.csv`, `grid-history.csv` and native Boundary/Object force history with its population step/time. Retain all of them when checking regrid/force relationships.

Dynamic 0/1/2 outcomes qualify the documented data/engineering relations, not full scientific error or performance. The force-sawtooth root cause is unresolved, and cancelled/timed-out/deferred GPU long runs are not complete results. [Known limitations](../../../docs/known-limitations.md), [references](../../../docs/references.md) and [pending snapshot validation](../../../docs/validation.md) apply.
