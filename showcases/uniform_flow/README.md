# Uniform flow

This showcase owns three BGK uniform-flow compositions: HBB midpoint velocity inlet/outlet, Guo valid-node velocity inlet/outlet, and Guo velocity inlet with full-q exact-copy free outflow. The y/z directions are periodic. The targets `lbm_uniform_flow` and `lbm_uniform_flow_d3q27` choose D2Q9/D3Q27 at compile time.

HBB uses 10 x-cell centers for 10 physical intervals. Guo uses 11 valid x nodes for the same intervals, with physical boundaries at x=0 and x=1 and a cell-centered AMReX envelope extending beyond them. Representative D3 runs retain the quasi-two-dimensional identity; changing the descriptor alone is not a new 3D benchmark.

## Build and run

First configure the repository root using [building.md](../../docs/building.md). These commands assume `build/cpu` and are run from the root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_uniform_flow lbm_uniform_flow_d3q27
OMP_NUM_THREADS=1 ./build/cpu/lbm_uniform_flow \
  showcases/uniform_flow/inputs/inputs.d2q9_guo_velocity_io
```

This tracked input is a 20-step, 10×10-interval, Nz=1 short case. Other tracked inputs select `hbb_velocity_io` and `guo_free_outflow`. A D3 run must explicitly select a suitable depth and output identity. A case-local build still uses the root source directory, for example `cmake -S . -B showcases/uniform_flow/build ...`.

## Controls, artifacts and interpretation

[inputs/](inputs/) exposes the variant, intervals/depth, density/velocity/viscosity scales, maximum grid size, run limits, diagnostics and output controls. Descriptor/dynamics are C++ composition choices. Do not treat one case's controls as a global parser contract.

`uniform_flow_metrics.csv` records exact-state errors, transverse velocity and raw/physical-weighted mass checks. The input's `output.directory` and `output.log_directory` keep generated data in [outputs/](outputs/README.md) and terminal/runtime records in [log/](log/README.md). Optional plotfiles are observations, not additional acceptance evidence.

The local [checker](scripts/check_uniform_flow.py) and [runner](scripts/run_uniform_flow.sh) are manual showcase tools. Short uniform-state preservation does not replace a sufficiently long flow-through comparison. Formal numerical tests independently own their compositions and exact reference in [tests/numerical](../../tests/numerical/README.md); they do not consume showcase outputs implicitly.

The D2Q9 CPU/CUDA 20-step route passed the [selected standalone checks](../../docs/validation.md). D3Q27 evidence here is limited to compilation.
