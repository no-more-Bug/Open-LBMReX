# Static AMR lid-driven cavity

This BGK/HBB Re=1000 test uses a 128² coarse grid, 256² uniform-fine grid and center-refined AMR. D2Q9 uses Nz=1; D3Q27 is quasi-two-dimensional. Targets are `lbm_numerical_amr_lid_driven_cavity` and `lbm_numerical_amr_lid_driven_cavity_d3q27`, with [d2q9](inputs/d2q9)/[d3q27](inputs/d3q27) inputs.

## Execution

Configure the root as in [building](../../../docs/building.md) and explicitly build the two targets. The [runner](run.sh) takes:

```text
<d2-exe> <d3-exe> <work-dir> <MPI-launcher>
<d2|d3>-<coarse|fine|amr> [fine-work-dir]
```

Run the fine reference first; coarse and AMR entries consume its explicit work directory. This runner requests the **complete** t=200 identity and does not honor `LBM_TEST_MAX_STEP` as a general short-run override. For bounded engineering coverage use the documented integration tests or a deliberately bounded direct executable run; do not label an incomplete run as scientific acceptance.

## Scientific contract

The last three steady residuals must be below 5e-8. Current Ghia centerline budgets are 6.5%; AMR-to-fine full-composite-domain volume-weighted velocity RMS (normalized by lid speed) and interface-peak budgets are 1% and 3%, mass drift is bounded by 1e-8, and quasi-two-dimensional quantities by 1e-10. The eligible fine reference for this AMR comparison is qualified by its required steady evidence; this differs from cylinder rules that additionally require external literature gates.

Earlier results outside the original 6% policy are retained as such. The later 6.5% policy does not mean those results were recomputed. The separate [single-level showcase](../../../showcases/lid_driven_cavity/README.md) and legacy single-level numerical checker retain their own 6% criteria.

Artifacts use `cavity-v1` and include identity, layout/sampling records, native frames and centerline comparisons. [check_cavity.py](check_cavity.py) owns acceptance. Ghia values are test-owned transcriptions with [provenance](../../../docs/references.md) and [distribution qualifications](../../../docs/third-party.md). The snapshot's fresh package/build/run checks remain [pending](../../../docs/validation.md).
