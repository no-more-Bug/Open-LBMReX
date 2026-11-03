# Taylor–Green vortex

This periodic BGK example uses the analytical decaying vortex on a unit square with a thin periodic z direction. `lbm_taylor_green_vortex` selects D2Q9 and `lbm_taylor_green_vortex_d3q27` selects D3Q27; the supplied D3 identity is quasi-two-dimensional.

The showcase's analytical reference and production Taylor–Green profile/math share implementation. This is useful for demonstration and convergence observation, but the showcase reference alone is not an independent oracle for that shared implementation.

## Build and a bounded run

Configure the repository root using [building.md](../../docs/building.md), then run from the root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_taylor_green_vortex lbm_taylor_green_vortex_d3q27
OMP_NUM_THREADS=1 ./build/cpu/lbm_taylor_green_vortex \
  showcases/taylor_green_vortex/inputs/inputs.d2q9_periodic_2d \
  run.max_step=8
```

The periodic D2 input uses 64×64×1 cells and a default 100-step cap. The override above is only a bounded execution example. The separate `inputs.d2q9_tau_convergence` and `inputs.d3q27_tau_convergence` identities implement the fixed-tau refinement route; do not infer convergence order from this eight-step run.

## Controls and output

The [inputs](inputs/) define the active resolution, reference scales, viscosity/tau parameterization, run horizon, diagnostics and output cadence. Descriptor/dynamics and the periodic composition are selected by C++.

Generated artifacts include `diagnostics.csv` and optional AMReX plotfiles. The convergence path additionally writes `convergence.csv` with its actual resolution, parameters, time and error measures. Files remain in [outputs/](outputs/README.md); logs belong to [log/](log/README.md). Use the local [runner](scripts/run_taylor_green_vortex.sh) and [convergence checker](scripts/check_tgv_convergence.py) only with artifacts matching their identity and complete time horizon.

The [formal TGV tests](../../tests/numerical/README.md) and [AMR TGV tests](../../tests/numerical/amr_taylor_green_vortex/README.md) have their own compositions and acceptance. Static AMR reported convergence order is not automatically a second-order pass condition. See [validation status](../../docs/validation.md) and [limitations](../../docs/known-limitations.md).
