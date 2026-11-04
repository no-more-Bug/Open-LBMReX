# Poiseuille flow

This BGK channel-flow showcase provides HBB velocity-profile, Guo velocity, Guo pressure and Guo prescribed-state compositions. The physical channel is `[0,4]×[0,1]` with periodic z; the cell-centered AMReX envelope and valid-node/midpoint wall locations depend on the boundary scheme.

Targets `lbm_poiseuille_flow` and `lbm_poiseuille_flow_d3q27` select D2Q9/D3Q27 at compile time. For the Guo pressure input, 40×10 physical intervals correspond to 41×11 valid x/y nodes; this distinction matters in the pressure-drop formula.

## Build and bounded execution

Configure the repository root using [building.md](../../docs/building.md). From that root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_poiseuille_flow lbm_poiseuille_flow_d3q27
OMP_NUM_THREADS=1 ./build/cpu/lbm_poiseuille_flow \
  showcases/poiseuille_flow/inputs/inputs.d2q9_guo_pressure \
  run.max_step=8 diagnostics.stop_on_steady=0 \
  poiseuille_validation.assert=0 \
  output.plot_interval=0 output.write_final_plotfile=0
```

The tracked input normally allows 20000 steps and enables its profile assertions. The command deliberately disables those assertions only for startup/schema inspection at eight steps; it is **not a scientific acceptance run**. Use the unchanged input and complete convergence/steady-state protocol for a scientific comparison. Keep this distinction in any saved result description.

## Numerical meaning and controls

[inputs/](inputs/) exposes the chosen variant, physical intervals/depth, density and speed scales, relaxation/pressure controls, mesh size, diagnostics and output. The default Guo pressure identity has `rho0=1`, `Umax=0.1`, `tau=0.75` and inlet/outlet densities `1.04/0.96`. Pressure-drop scaling uses lattice intervals rather than the number of valid nodes.

`fixed_re_fixed_tau` is the formal refinement route. Fixed-Mach experiments characterize a different scaling and are not interchangeable convergence evidence. HBB and Guo physical plane conventions must remain separate.

## Artifacts and checks

The case writes `poiseuille_profile.csv`, `poiseuille_profile_metrics.csv` and `poiseuille_density_pressure.csv` under [outputs/](outputs/README.md), with logs under [log/](log/README.md). Profile error, transverse velocity, wall/pressure consistency and applicable mass/steady checks belong to the selected case policy.

The local [Guo runner](scripts/run_guo_poiseuille.sh), [velocity runner](scripts/run_poiseuille_velocity.sh) and [convergence checker](scripts/check_poiseuille_convergence.py) consume explicit case artifacts. [Formal numerical tests](../../tests/numerical/README.md) own separate references and paths. The direct D2Q9 CPU/CUDA eight-step route passed the [selected standalone checks](../../docs/validation.md). Full convergence runs and D3Q27 execution were not part of those checks.
