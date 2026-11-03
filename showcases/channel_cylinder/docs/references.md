# DFG-derived references and scope

The geometry follows the Featflow [DFG 2D-1 Re20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) and [DFG 2D-2 Re100](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html) definitions: physical domain `[0,2.2]×[0,0.41]`, cylinder center `(0.2,0.2)`, radius `0.05`. The original continuum outlet is do-nothing; the project's Guo constant-density outlet is a deliberate different composition.

The benchmark Reynolds number uses diameter D=0.1. The LBM parameterization uses channel height H=0.41, giving Re_H=82 for Re_D=20 and Re_H=410 for Re_D=100. Physical boundary positions and the cell-centered AMReX envelope remain distinct.

## Re20

[reference-data/dfg_re20_reference.csv](reference-data/dfg_re20_reference.csv) has SHA256 `0a4cb839f008506e7541f76f0db23c7eaa5c44672f2347d09870118b46448043`.

The official source values are Cd=5.57953523384, Cl=0.010618948146 and pressure difference 0.11752016697. The project normalization gives C_delta_p=2.93800417425. Tolerance columns, zero-baseline rows and resolution-difference/steady policies in the CSV are project acceptance parameters, not additional published reference measurements.

The complete showcase scope uses ND80-to-reference Cd 1%, Cl absolute 0.002 and C_delta_p 3%; ND40-to-ND80 Cd 1% and Cl absolute 0.002; and at least one eligible steady residual ≤5e-7. The quasi-D3 ND20 comparison uses the corresponding D2 coarse result and also bounds normalized transverse force by 1e-10. A 12-step structural artifact is not eligible to claim these long-time results.

## Re100

[reference-data/dfg_re100_reference.csv](reference-data/dfg_re100_reference.csv) has SHA256 `f1ff94190e21e78f8c313018e3376da20ffeb5c261b399bb5c1f2663bddf2d2f`.

The source is the official new reference dataset dated 2014-11-24, Q2/P1disc, level 6, dt4=1/1600. Its force/point-value traces include `bdforces_q2_lv6_dt4` and `pointvalues_q2_lv6_dt4`. Adjacent discrete lift minima at physical times 9.6609375 and 9.9921875 define the reference period.

The showcase's scientific scope is D2Q9/YMS, ND40/80 and this geometry/outlet. After t*≥250, the analyzer uses the last three complete lift-minimum cycles with at least 200 samples per cycle and ≤1% period/Cd-extrema/Cl-extrema drift. Relative envelopes are St 2%, Cd min/max/midrange 2%, Cl min/max 5%, and Cd/Cl peak-to-peak 10%.

Existing combined outcomes remain scientific failures: Cd peak-to-peak relative errors were about 31.78% and 27.02%, outside 10%, even though the scoped Strouhal check passed. A bounded short run must report insufficient cycle evidence rather than a long-time pass. This evidence does not extend to Bouzidi, showcase D3 Re100 or another outlet discretization.

## Ownership and distribution

The showcase owns these copies; numerical/regression tests own separate copies with the same documented values/hashes. Changes require explicit scientific review across the relevant owners, not a silent runtime fallback or configure-time copy.

These are limited numerical transcriptions with project-derived normalization and acceptance fields. No separate open redistribution licence has been obtained from the original sources. The website's [copyright notice](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/impressum.html) is not replaced by the project software licence. See [third-party material](../../../docs/third-party.md).
