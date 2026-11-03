# Ghia reference provenance

U. Ghia, K. N. Ghia and C. T. Shin, “High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method,” *Journal of Computational Physics* 48(3), 387–411 (1982), [doi:10.1016/0021-9991(82)90058-4](https://doi.org/10.1016/0021-9991(82)90058-4).

The two files are numerical transcriptions of the Re=1000 vertical Ux(y) centerline at x/L=0.5 and horizontal Uy(x) centerline at y/L=0.5, normalized by lid speed. They are not project-generated reference solutions.

| File | SHA256 |
| --- | --- |
| [Re1000_x0.5_Ux_Ghia.csv](reference-data/Re1000_x0.5_Ux_Ghia.csv) | `1b0081be1d63be075cdcc609a28cef242440803cc39b7aec00b115fdf3549d8a` |
| [Re1000_y0.5_Uy_Ghia.csv](reference-data/Re1000_y0.5_Uy_Ghia.csv) | `791c823f3d574b9dadbcb6da4581be22cc719fbde237c65c02d201ae9164dc0c` |

This showcase reads its own reference files. Test-owned copies retain separate ownership and distinct byte hashes despite agreeing numerical values. No automatic copy, merge or fallback between owners is implied.

The showcase's 6% centerline policy and adjacent-step steady criterion are project decisions, not claims made by Ghia et al. Existing Guo scientific failure remains a [known limitation](../../../docs/known-limitations.md). The AMR numerical cavity has a different current policy; it does not overwrite this one.

No publisher PDF is distributed. No separate open redistribution licence has been obtained for the original source; the project licence does not relicense the paper or its protected expression. See [third-party material](../../../docs/third-party.md).
