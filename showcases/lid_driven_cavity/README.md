# Lid-driven cavity

This BGK cavity starts from rest, uses a moving top lid along +x, and supports HBB midpoint or Guo valid-node wall composition. `lbm_lid_driven_cavity` selects D2Q9 and `lbm_lid_driven_cavity_d3q27` selects D3Q27. The supplied D3 cases are quasi-two-dimensional.

A physical interval count is not always a valid-cell/node count: a 129-node Guo wall layout has 128 intervals, whereas a 129-center HBB layout has 129 intervals. Do not compare them as an identical wall discretization.

## Build and a short case

Configure the repository root using [building.md](../../docs/building.md), then run from the root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_lid_driven_cavity lbm_lid_driven_cavity_d3q27
OMP_NUM_THREADS=1 ./build/cpu/lbm_lid_driven_cavity \
  showcases/lid_driven_cavity/inputs/inputs.d2q9_hbb_smoke
```

The input is Re=100, 50×50 intervals, **four z cells even with D2Q9**, and 50 steps. Its HBB mode disables valid-node wall metrics with `diagnostics.write_metrics=0`; those metrics do not apply to midpoint walls. This is a startup/boundary example, not a Ghia Re=1000 comparison.

## Controls, artifacts and references

The [inputs](inputs/) expose wall scheme, intervals/depth, physical bounds, density/Re/lid velocity, run/steady controls, diagnostics and output cadence. Re=1000 HBB and Guo inputs are distinct longer identities.

The case can write diagnostics, `cavity_centerline_x0p5_ux.csv`, `cavity_centerline_y0p5_uy.csv`, metadata, metrics and plotfiles according to enabled controls. Generated files belong to [outputs/](outputs/README.md), logs to [log/](log/README.md).

[Reference provenance](docs/references.md) documents the two Ghia numerical transcriptions. The local [comparison tool](scripts/compare_ghia_cavity.py) retains a **6%** default centerline policy and its own steady requirement. Existing Guo scientific failure remains a limitation; changing documentation does not change that outcome.

The [formal AMR cavity](../../tests/numerical/amr_lid_driven_cavity/README.md) is a different owner with a 6.5% current budget. Do not replace this showcase's threshold with that number. Both use a quasi-two-dimensional comparison rather than a full 3D validation. See [known limitations](../../docs/known-limitations.md) and [snapshot validation](../../docs/validation.md).
