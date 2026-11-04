# Validation scope

## v0.1.0 evidence and source revisions

Selected standalone checks completed on 2026-11-04 using fresh CPU and CUDA AMReX installation packages. These are bounded engineering checks, not a release-wide scientific pass claim.

- [`e90c00432586bac8fc5ad693813e7f61df85f387`](https://github.com/no-more-Bug/Open-LBMReX/commit/e90c00432586bac8fc5ad693813e7f61df85f387): the CPU and CUDA test matrices, physical tests/showcases isolation, and the CPU example/performance checks.
- [`fe75e2bbc7fbc13dbf48ac97d5eecf45842f2ceb`](https://github.com/no-more-Bug/Open-LBMReX/commit/fe75e2bbc7fbc13dbf48ac97d5eecf45842f2ceb): the CUDA example/performance checks and an ordinary source archive after fixing CUDA compilation settings in four showcase CMake files.

The second revision changes only conditional CUDA target setup for uniform flow, TGV, Poiseuille and lid-driven cavity. Core implementation, tests, shared CMake and the root CMake file are unchanged; the CPU branch and the `LBM_BUILD_SHOWCASES=OFF` test configuration do not execute those additions. The earlier matrices remain applicable on that basis; they were not rerun at the second SHA. The v0.1.0 release adds documentation, citation metadata and illustrative animations to that tested code revision. The animations reuse earlier development runs and are not additional release-validation evidence.

The initial CUDA showcase build failed because these targets were compiled as ordinary C++. The corrected D2/D3 targets then compiled successfully, and the selected D2 runs passed. That failure and its correction are separate evidence; the result is not described as a clean first attempt.

## Tested environments

Both installations used official AMReX `4ceb814db7d79c5858b13dab153fb04ecc3b129d` with the [projected-tags patch](../patches/amrex/README.md), SHA256 `32474c1a9f61ac7bbae0aa34b046bca3ad21e610d2587c6d99b0f1d5113ab4b4`. Both packages are 3D, double precision, EB-enabled, MPI-enabled, Release builds using C++20. The clean submodule and the separately patched preparation source are distinct.

| Route | Recorded toolchain and execution scope |
| --- | --- |
| Linux CPU/MPI | GCC 12.3.0, Open MPI 5.0.10, CMake 3.29.4, Python 3.11.7 and NumPy 1.26.4; Intel Xeon Platinum 8358. Selected tests use 1/2/4 ranks, examples use 2, performance uses 16. Each rank uses one OpenMP thread with explicit core binding. |
| NVIDIA CUDA | GCC 11.2.0, CUDA/NVCC 12.2.140, Open MPI 4.1.1, CMake 3.26.5, Python 3.9.7 and NumPy 1.26.4; driver 545.23.08, SM80, A100-SXM4-40GB `1g.5gb` MIG. Selected tests use 1/2/4 ranks with one distinct MIG and five host CPUs per rank; example/performance smoke runs use one MIG and eight host CPUs. |

The CPU package enables OpenMP, but all recorded executions use `OMP_NUM_THREADS=1`; this does not validate OpenMP/hybrid execution. The CUDA package disables OpenMP and uses pinned staging, `amrex.use_gpu_aware_mpi=0`. Its MPI installation was built against CUDA 11.5; the tested application toolkit is 12.2.140. These versions describe the tested combination, not a universal compatibility guarantee.

Due to limited computing resources, GPU multi-device evidence is limited to MIG configurations. This snapshot's multi-rank tests used four distinct MIG instances on one physical A100, including an independent four-rank device-mapping check. Execution across multiple full physical GPUs has not been validated and is not guaranteed to run successfully or pass tests.

## Checks completed

| Area | Actual evidence |
| --- | --- |
| New packages | Fresh CPU/CUDA builds and installations; consuming EB header/link probes passed. CUDA tests initialized real devices. A base SHA alone does not identify the patch or configuration. |
| Selected tests | CPU: 90 of 392 registered tests passed. CUDA: 80 of 399 registered tests passed. Neither selected set had failures or skips. The CPU set includes 11 host-only checks, counted once; the CUDA set includes a public-header link test that is not kernel evidence. |
| Runtime coverage | D2/D3 application/core/boundary units, single-level and static-hierarchy runtime, coarse/fine exchange and advance, dynamic preparation/clustering/regridding/constraints, boundary preparation/evolution/force and relayout/domain checks. Selection is bounded; plain `ctest` includes additional and long-running cases. |
| Numerical runner paths | Per backend: D2/D3 static uniform flow and AMR TGV at 8 steps, dynamic DSL at 16 steps and dynamic single cylinder at 32 steps. Source identity, arguments, artifacts and checkers were checked. These use engineering/observation scope. |
| Physical isolation | Dedicated source copies physically lacked `showcases/`, used `LBM_BUILD_SHOWCASES=OFF`, and built/ran the selected CPU/CUDA tests. Filesystem, CMake, include, link and execution dependencies were checked; merely switching showcases off was not treated as isolation proof. |
| Five showcases | All ten D2/D3 targets compiled on each backend. D2 runs completed uniform flow 20 steps, TGV 8, Poiseuille 8, cavity 50 and channel cylinder 12, with finite diagnostics and expected basic artifacts. D3 showcase execution was not part of this validation. |
| Performance tools | Eight explicit targets compiled per backend. CPU ran two legacy 20-step StageBreakdown records and four AMR 8-step records across TGV/DFG and Throughput/StageBreakdown. CUDA ran four legacy 20-step and four AMR 8-step records. All runs were D2, used independently supplied source/package/resource expectations, and passed their consumers. These 14 processes are smoke checks, not formal timing baselines or speedup measurements. |
| Ordinary source archive | A tree without `.git` or bundled AMReX was built against the new external CPU package. A uniform-flow example and an eight-step strict AMR uniform runner passed with the corresponding public SHA. Missing/invalid identities returned `2` before solver execution. CUDA source-archive execution was not tested. |

The preparation and portability tools also have host-side checks for dependency preparation, source-identity resolution, scratch ownership and CPU/GPU/AMR artifact consumers, including intentionally invalid records. Those synthetic tests are not CFD solver or GPU kernel evidence.

The [build commands](building.md) describe the exercised routes. For an archive downloaded from the release tag, use the full public commit corresponding to that archive as `LBM_EXPECTED_GIT_COMMIT`, not one of the older test-source SHAs above. HIP, SYCL, native Windows builds, arbitrary compiler/MPI/CUDA combinations, OpenMP/hybrid execution, Compute Sanitizer and optional ParaView/GUI checks are outside this release's validation.

## Engineering and scientific outcomes

Tests own their observable expectations independently of production code and showcases. Short engineering tests cover initialization, layouts, transfers, runtime/boundary behavior and artifacts. The [numerical suite](../tests/numerical/README.md) owns longer analytic/literature comparisons and explicit reference eligibility.

For scientific numerical runners, `0` means valid complete evidence meeting the selected criteria, `1` means valid complete evidence outside at least one scientific criterion, and `2` means invalid or insufficient evidence. A documented engineering/short mode has its own bounded meaning and cannot turn an earlier scientific `1` or `2` into success. Performance tools retain their own documented status contracts.

No complete scientific long-run or formal performance matrix was repeated for this release. Dynamic-grid granularity, Re100 scientific failures/insufficient evidence, unresolved force sawtooth behavior and incomplete GPU long runs remain [known limitations](known-limitations.md). The version number is not a measure of scientific completion.
