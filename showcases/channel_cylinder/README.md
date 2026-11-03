# Channel flow around a cylinder

This DFG-derived BGK case uses a physical domain `[0,2.2]×[0,0.41]`, a cylinder centered at `(0.2,0.2)` with radius `0.05`, and periodic z. Guo inlet/outlet/walls are combined with YMS or Bouzidi internal-cylinder treatment. The Guo constant-density outlet is a deliberate LBM choice and is not the original DFG do-nothing discretization.

Targets are `lbm_channel_cylinder` and `lbm_channel_cylinder_d3q27`. The showcase's `re100_yms` composition is **D2Q9-only**; the D3 target does not make every input valid for D3. Its Re20 quasi-D3 literature comparison is limited to `N_D=20`, `Nz=4`. Formal AMR/Re100 tests have separate compositions and evidence.

## Build and a structural short run

After configuring the repository root as in [building.md](../../docs/building.md), run from the root:

```bash
cmake --build build/cpu --parallel 4 \
  --target lbm_channel_cylinder lbm_channel_cylinder_d3q27
OMP_NUM_THREADS=1 ./build/cpu/lbm_channel_cylinder \
  showcases/channel_cylinder/inputs/inputs.single_yms_d2_short
```

This tracked input uses `N_D=20`, Re_D=20 represented by Re_H=82, and 12 steps. It checks startup, boundary/object bookkeeping, force cadence and artifact production. It cannot establish steady drag/lift accuracy. Other [inputs](inputs/) include Bouzidi and D3 short cases, Re20 refinements, D2 Re100 identities and tandem distinct/shared ObjectId compositions.

## Controls and output

Inputs expose the case variant, diameter intervals/depth, density/speed/Re scales, grid size, run/steady limits, force/output cadence and enabled observations. Scientific composition is C++-owned.

`channel_cylinder_force_history.csv` and `channel_cylinder_metrics.csv` identify boundaries/objects, population step and physical time, force/coefficient samples and pressure probes. Keep the force publication/final-drain timing and cadence when interpreting them. Outputs and logs have separate [case-owned directories](outputs/README.md); [log/](log/README.md) is not reference data.

[Scientific references](docs/references.md) specify DFG numerical provenance, normalization, complete-horizon/cycle requirements and current acceptance. Re100 combined scientific failures remain; a successful solver exit or passed Strouhal sub-gate does not make the combined comparison pass.

Formal [AMR cylinder tests](../../tests/numerical/amr_channel_cylinder/README.md), experimental dynamic AMR and this single-level showcase are distinct owners. The force-sawtooth and long-run limitations remain visible in [known limitations](../../docs/known-limitations.md). Standalone-snapshot builds/runs are still [pending](../../docs/validation.md).
