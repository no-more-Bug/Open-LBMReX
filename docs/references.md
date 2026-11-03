# Scientific references and reference-file identity

Citation supports reproducibility and attribution; it is not a claim that the project owns the original publications or has obtained a separate open licence for them. Read [third-party material](third-party.md).

## AMReX

The fixed dependency's [CITATION](../patches/amrex/CITATION) preserves the upstream citation requests, including:

- W. Zhang et al., “AMReX: A Framework for Block-Structured Adaptive Mesh Refinement,” *Journal of Open Source Software* 4(37), 1370 (2019), [doi:10.21105/joss.01370](https://doi.org/10.21105/joss.01370).
- W. Zhang, A. Myers, K. Gott, A. Almgren and J. Bell, “AMReX: Block-structured adaptive mesh refinement for multiphysics applications,” *International Journal of High Performance Computing Applications* 35(6), 508–526 (2021), [doi:10.1177/10943420211022811](https://doi.org/10.1177/10943420211022811).

Use the actual pinned dependency identity with results. Do not substitute a generic AMReX version or an unrelated DOI for this project's unreleased software.

## LBM grid refinement and non-uniform grids

These works provide methodological background for learning about conservative grid refinement, non-uniform-grid implementation and interface artifacts. They are not evidence that every cited method or published result has been implemented or reproduced by Open-LBMReX.

- M. Rohde, D. Kandhai, J. J. Derksen and H. E. A. van den Akker, “A generic, mass conservative local grid refinement technique for lattice-Boltzmann schemes,” *International Journal for Numerical Methods in Fluids* 51(4), 439–468 (2006), [doi:10.1002/fld.1140](https://doi.org/10.1002/fld.1140).
- H. Chen, O. Filippova, J. Hoch, K. Molvig, R. Shock, C. Teixeira and R. Zhang, “Grid refinement in lattice Boltzmann methods based on volumetric formulation,” *Physica A: Statistical Mechanics and its Applications* 362(1), 158–167 (2006), [doi:10.1016/j.physa.2005.09.036](https://doi.org/10.1016/j.physa.2005.09.036).
- Christophe Coreixas and Jonas Latt, “GPU-based compressible lattice Boltzmann simulations on non-uniform grids using standard C++ parallelism: From best practices to aerodynamics, aeroacoustics and supersonic flow simulations,” *Computer Physics Communications* 317, 109833 (2025), [doi:10.1016/j.cpc.2025.109833](https://doi.org/10.1016/j.cpc.2025.109833); [arXiv:2504.04465](https://arxiv.org/abs/2504.04465).
- Alexander Schukmann, Viktor Haas and Andreas Schneider, “Spurious Aeroacoustic Emissions in Lattice Boltzmann Simulations on Non-Uniform Grids,” *Fluids* 10(2), 31 (2025), [doi:10.3390/fluids10020031](https://doi.org/10.3390/fluids10020031).

## Flow benchmarks

- **DFG/Featflow cylinder:** official [2D-1 Re20](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html) and [2D-2 Re100](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html) definitions and reference values. The LBM constant-density outlet differs from the original do-nothing discretization; results are DFG-derived.
- **Cavity:** U. Ghia, K. N. Ghia and C. T. Shin, “High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method,” *Journal of Computational Physics* 48(3), 387–411 (1982), [doi:10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4).
- **Double shear layer:** Michael L. Minion and David L. Brown, “Performance of Under-resolved Two-Dimensional Incompressible Flow Simulations, II,” *Journal of Computational Physics* 138, 734–765 (1997), [doi:10.1006/jcph.1997.5843](https://doi.org/10.1006/jcph.1997.5843). The primary digitized reference is the open-circle series in Figures 5/6, not the PSpect line.

Analytic uniform flow, Taylor–Green, Poiseuille and periodic-shear expectations are described by their [test contracts](../tests/numerical/README.md). Project tolerances are acceptance policies, not accuracy guarantees made by reference authors.

## CSV ownership and SHA256

Reference ownership remains local. Matching values do not justify merging test/showcase assets or silently updating one from another.

| Reference | Owners | SHA256 |
| --- | --- | --- |
| DFG Re20 | [showcase](../showcases/channel_cylinder/docs/reference-data/dfg_re20_reference.csv), [numerical](../tests/numerical/channel_cylinder/reference-data/dfg_re20_reference.csv), [regression](../tests/regression/reference-data/channel_cylinder/dfg_re20_reference.csv) | `0a4cb839f008506e7541f76f0db23c7eaa5c44672f2347d09870118b46448043` |
| DFG Re100 | [showcase](../showcases/channel_cylinder/docs/reference-data/dfg_re100_reference.csv), [numerical](../tests/numerical/channel_cylinder/reference-data/dfg_re100_reference.csv) | `f1ff94190e21e78f8c313018e3376da20ffeb5c261b399bb5c1f2663bddf2d2f` |
| Ghia Ux | [showcase](../showcases/lid_driven_cavity/docs/reference-data/Re1000_x0.5_Ux_Ghia.csv) | `1b0081be1d63be075cdcc609a28cef242440803cc39b7aec00b115fdf3549d8a` |
| Ghia Uy | [showcase](../showcases/lid_driven_cavity/docs/reference-data/Re1000_y0.5_Uy_Ghia.csv) | `791c823f3d574b9dadbcb6da4581be22cc719fbde237c65c02d201ae9164dc0c` |
| Ghia Ux | [numerical](../tests/numerical/lid_driven_cavity/reference-data/Re1000_x0.5_Ux_Ghia.csv) | `82edf9b0451c8432c7652dd589bd008465b399dd87243eaf6a4a7ccca610b9a0` |
| Ghia Uy | [numerical](../tests/numerical/lid_driven_cavity/reference-data/Re1000_y0.5_Uy_Ghia.csv) | `20c5b65b8e07d80362f48b17de26041099cb19bfb337f32e703189a0ca39a9cc` |
| Minion open circles | [numerical](../tests/numerical/amr_double_shear_layer/reference-data/fig5-6-open-circle-reference.csv) | `0cee6f1346f68ad830082d6a19ff94dbbf4bdbb2ab69c5247c4e0e558a20aef5` |
| Minion PSpect auxiliary | [numerical](../tests/numerical/amr_double_shear_layer/reference-data/fig5-6-pspect-auxiliary.csv) | `386fca9675992be3c649e6a8071094e98c5e9481aa15358b17df1f7da2156534` |

The Ghia numerical/showcase numbers agree, but line endings/trailing whitespace differ; the files are not byte-identical. DFG files also contain project acceptance/normalization fields, not only source-author values. The Minion reference consists of 11 primary and 21 auxiliary samples, with the [documented readout uncertainty](../tests/numerical/amr_double_shear_layer/reference-data/README.md). Numerical reference files are preserved byte-for-byte in this preparation.
