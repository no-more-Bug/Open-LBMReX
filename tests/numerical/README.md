# Numerical tests

Numerical tests own their executable compositions, inputs, runners, reference data and consumers independently of showcases. The runnable truth is local: `src/` selects descriptor/dynamics and composition, `inputs/` exposes supported controls, `run*.sh` defines entry/launch/horizon, and `check*.py` plus reference CSVs define acceptance.

Use the [common build and test instructions](../README.md). The current [CMake registration](CMakeLists.txt) determines which identities exist for the chosen backend. Do not run all numerical tests as a quick smoke check.

## Execution and result contract

```text
test-owned executable + input
→ attempt-local artifacts and logs
→ explicit checker + eligible reference
→ result
```

For scientific runners, `0` means complete valid evidence meeting the prescribed scientific criteria; `1` means complete valid evidence outside a criterion; `2` means invalid or insufficient evidence, including launch, schema, source identity, horizon or reference problems. Engineering modes and dynamic-data checkers have narrower contracts, described in their case pages. A successful short mode does not change a scientific failure.

Artifacts normally live under `<build>/tests/numerical/work/<test-name>/attempt-<timestamp>-<pid>/`. Each test owns its writable run/log/tmp paths. An entry requiring an external reference consumes an explicit matching reference work directory and does not silently compute it. The periodic-shear group runner explicitly includes its own uniform-reference matrix; see its page.

Strict AMR runners use `LBM_EXPECTED_GIT_COMMIT` or the actual containing clone HEAD. See [source identity](../README.md#source-identity). Reference eligibility follows each case: static scientific comparisons require their prescribed source/descriptor/physical identity and horizon. Dynamic DSL observational comparison may retain a different reference source/backend while checking descriptor, domain, time and resolution; both identities remain recorded. Invalid reference evidence is not made eligible merely because a solver finished.

`LBM_TEST_MAX_STEP` is not a universal short-run switch. Uniform/TGV/Re100 support it with their engineering mode; dynamic DSL/cylinder use it with short-data checks. Static DSL, periodic shear, AMR cavity and Re20 runners explicitly request the complete horizon. Tandem short and comparison runners have their own fixed lengths.

## Single-level scientific families

The entries below are independent from the similarly named showcases. D3 examples in these families are quasi-two-dimensional.

| CTest / locality | Configuration and complete scope | Scientific criteria |
| --- | --- | --- |
| `numerical_tgv_convergence` / `taylor_green_vortex/` | D2Q9/D3Q27 BGK, Re=10, tau=0.75, N=50/75/100, periodic domain, target physical time 0.2. | Each descriptor's three-point relative velocity-L2 slope in [1.8,2.2]; D3 max transverse velocity ≤1e-12. Identity/time mismatch is invalid evidence. |
| `numerical_uniform_flow` / `uniform_flow/` | HBB velocity I/O, Guo velocity I/O and Guo free outflow; D2/D3; rho=1, lattice speed 0.1, viscosity 1/12; 10000 steps/100 flow-throughs. | Relative velocity L2 and relative mass error each ≤1e-8; variant/descriptor/horizon eligibility retained. |
| `numerical_guo_pressure_poiseuille` / `poiseuille_flow/` | N=10/20/50, fixed Re/fixed tau=0.75; Umax=1/N, inlet/outlet density 1±4/N²; up to 200N² steps with required steady completion. | Each descriptor's profile-L2 slope in [1.8,2.2]. Other diagnostics are not silently added as hard criteria. |
| `numerical_hbb_cavity_ghia` / `lid_driven_cavity/` | Re=1000 HBB, 129×129 centers, D2 Nz=1/D3 Nz=4; up to 200000 steps, steady checks every 5000 steps. | Both Ghia centerline maximum metrics ≤0.06; interpolate at reference coordinates, use relative error for nonzero references and absolute error at zeros. This remains separate from AMR cavity's 6.5% policy. |
| `numerical_channel_cylinder_baseline` / `channel_cylinder/` | DFG-derived BGK/YMS; Re20 D2 ND20/40/80 and quasi-D3 ND20, t=52.5; Re100 D2 ND40/80, t=35. | Re20 drag/lift/pressure/refinement criteria; Re100 complete-cycle/drift and reference-envelope criteria. Re100 combined scientific failures remain. |

The legacy channel group may use mixed MPI/OpenMP layouts by default. Inspect its registration and runner before launching; `LBM_TEST_R5_MPI_RANKS` selects its documented pure-MPI override. Specify an appropriate launcher and resources rather than assuming all groups use the same ranks.

For Re20, ND80 reference tolerances are Cd 1%, Cl absolute 0.002 and normalized pressure difference 3%; ND40-to-ND80 uses Cd 1% and Cl absolute 0.002. D3-to-D2 coarse comparison also requires normalized transverse force ≤1e-10. Eligible runs retain the steady tolerance 5e-7.

Re100 evaluates the last three complete lift-minimum cycles after t*=250, with at least 200 samples/cycle and ≤1% drift of period/Cd extrema/Cl extrema. Reference envelopes are St 2%, Cd extrema/midrange 2%, Cl extrema 5%, and Cd/Cl peak-to-peak 10%. A passed Strouhal sub-gate is not a combined pass. The existing ND40/80 combined result remains a scientific failure; an incomplete history is invalid/insufficient evidence.

## AMR and dynamic localities

| Locality | Scope |
| --- | --- |
| [AMR uniform flow](amr_uniform_flow/README.md) | Static three-level uniform-state and conservation comparison. |
| [AMR TGV](amr_taylor_green_vortex/README.md) | Static center refinement and explicit uniform-fine reference. |
| [AMR cavity](amr_lid_driven_cavity/README.md) | HBB cavity, Ghia and eligible fine-grid comparison. |
| [Periodic shear](amr_periodic_shear/README.md) | Decay, interfaces and complex static layouts; genuinely 3D varying shear mode as well as D2. |
| [Double shear layer](amr_double_shear_layer/README.md) | Static literature/uniform comparisons and distinct experimental dynamic/fixed-band contracts. |
| [AMR cylinder](amr_channel_cylinder/README.md) | Re20/Re100, tandem and experimental dynamic boundaries/force artifacts. |

These tests use cell-centered AMReX data. AMR D2 has Nz=1 and refinement ratio (2,2,1); AMR D3 uses (2,2,2). This does not impose Nz=1 on every separate single-level showcase.

## Selecting a run

Build the required test executable targets first. Inspect the selection before executing an exact test name:

```bash
ctest --test-dir build/cpu -N -L numerical
ctest --test-dir build/cpu -R '^numerical_uniform_flow$' \
  --output-on-failure -V
```

The second command is a complete scientific family, not a startup check. For bounded engineering runs use the explicit case examples or registered short integration entries and their actual resource requirements. One CTest driver owns a build tree at a time.

Reference files and small numerical transcriptions retain [attribution and distribution limitations](../../docs/third-party.md). Do not substitute a reference or revise a threshold to turn a failure into a pass. [Known limitations](../../docs/known-limitations.md) and [pending standalone validation](../../docs/validation.md) apply to the whole suite.
