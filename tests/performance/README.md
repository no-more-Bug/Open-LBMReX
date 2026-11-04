# Performance tools

This tests-owned harness measures fixed engineering workloads. It is not a numerical acceptance suite, a general hardware benchmark or evidence of equal-accuracy speedup. Explicit target builds and selected real CPU/CUDA smoke artifacts passed the [standalone checks](../../docs/validation.md), with D3 execution outside that scope. Historical platform results are not distributed as current results.

## Build and workload scope

Configure from the repository root using [the installation-package route](../../docs/building.md), `BUILD_TESTING=ON` and `CMAKE_BUILD_TYPE=Release`. Performance executables are excluded from the default build and are not registered as solver runs in CTest.

| Family | Explicit targets | Inputs and accepted resources |
| --- | --- | --- |
| Single-level CPU | `lbm_performance_{tgv,dfg}_{d2q9,d3q27}` | [inputs/](inputs/); 16 MPI ranks, one OpenMP thread per rank, fixed 16-box workloads. |
| Single-level CUDA | `lbm_gpu_performance_{tgv,dfg}_{d2q9,d3q27}` | [inputs/gpu/](inputs/gpu/); one or four `1g.5gb` MIG instances on one parent GPU, one rank per instance. |
| AMR versus uniform fine | `lbm_amr_performance_{tgv,dfg}_{d2q9,d3q27}` | [inputs/amr/](inputs/amr/); CPU 16 ranks or CUDA one/four MIG instances; matched physical windows and resource contracts. |

Braces abbreviate separate names. After configuring `build/cpu`:

```bash
cmake --build build/cpu --target lbm_performance_tgv_d2q9
cmake --build build/cpu --target lbm_amr_performance_tgv_d2q9
```

Executables are under `build/cpu/tests/performance/`. A CUDA build supplies the GPU single-level names instead of the CPU names; AMR target names are shared across backend builds.

| Single-level identity | Global grid | max_grid_size | Boxes |
| --- | --- | --- | --- |
| TGV D2Q9 | 1024 × 1024 × 1 | 256 | 16 |
| TGV D3Q27 | 512 × 512 × 4 | 128 | 16 |
| DFG D2Q9 | 1761 × 329 × 1 | 224 | 16 |
| DFG D3Q27 | 881 × 165 × 4 | 112 | 16 |

Single-level DFG uses Re20/YMS with Guo inlet/outlet/walls; AMR DFG reuses the Re100 numerical composition. AMR TGV reuses numerical TGV. [AmrPerformanceCases.H](src/AmrPerformanceCases.H) depends on those numerical sources. D3Q27 here is quasi-2D. Force sampling, plots and scientific acceptance are disabled for timing.

## CPU example and metadata

Use an allocation with the requested physical cores. Record CPU topology, allowed affinity and the MPI/compiler/package identity first. Non-hybrid runs use `OMP_NUM_THREADS=1` and explicit binding.

The CPU CSV writer accepts `LBM_PERFORMANCE_AMREX_SOURCE_COMMIT`, `LBM_PERFORMANCE_AMREX_PREFIX`, `LBM_PERFORMANCE_MPI`, `LBM_PERFORMANCE_UCX_VERSION`, `LBM_PERFORMANCE_UCX_TLS`, `LBM_PERFORMANCE_CPUSET`, `LBM_PERFORMANCE_BINDING_POLICY` and `LBM_PERFORMANCE_QOS`. Populate them with actual installation/allocation records. Compiler identity comes from CMake; node/partition come from the environment, with `local` as the non-Slurm partition/QoS fallback. Source normally comes from configuration; `LBM_PERFORMANCE_GIT_COMMIT`, if used, must describe the actual source.

After setting that metadata, this Open MPI command is a 20-step runtime/schema smoke, not a baseline:

```bash
mkdir -p build/cpu/performance-smoke
OMP_NUM_THREADS=1 mpirun -np 16 \
  --map-by core --bind-to core --rank-by slot --report-bindings \
  ./build/cpu/tests/performance/lbm_performance_tgv_d2q9 \
  tests/performance/inputs/tgv_d2q9.inputs \
  run.max_step=20 profiling.warmup_steps=0 profiling.measurement_steps=20 \
  performance.mode=smoke performance.repeat=1 \
  performance.output_csv="$PWD/build/cpu/performance-smoke/window.csv"
```

The current CPU solver targets require 16 ranks, with one box per rank. The raw CPU consumer also accepts historical 8-rank records; that compatibility does not make 8 ranks a supported launch for the current fixed targets.

CPU smoke acceptance requires repeat 1 and the full sample window 1:20. Formal baselines require three process repeats and an actual synchronized window of at least 600 seconds per record. Calibrate step counts on the intended resources; default input step counts do not guarantee duration.

## Independent consumer expectations

[scripts/performance_expectations.py](scripts/performance_expectations.py) defines required explicit expectations. Obtain them from independent installation/allocation records, not from the CSV being checked.

| Consumers | Required `--expected-` suffixes |
| --- | --- |
| Legacy CPU raw and summary | `amrex-source-commit`, `amrex-prefix`, `compiler`, `mpi`, `ucx-version`, `ucx-tls`, `partition`, `qos`, `binding-policy` |
| Legacy GPU consumers using the shared interface | `amrex-source-commit`, `amrex-prefix`, `amrex-config-sha256`, `compiler`, `nvcc`, `cuda-toolkit`, `mpi`, `mpi-cuda-build-version`, `partition`, `qos`, `mig-profile` |
| AMR raw and summary | `--expected-amrex-source-commit` and one or more `--expected-amrex-prefix` options, each a complete permitted prefix. |

After assigning the variables from that independent CPU context:

```bash
python3 tests/performance/scripts/check_performance_csv.py \
  build/cpu/performance-smoke/window.csv \
  --expected-amrex-source-commit="$AMREX_BASE_SHA" \
  --expected-amrex-prefix="$AMREX_INSTALL" \
  --expected-compiler="$COMPILER_ID" --expected-mpi="$MPI_ID" \
  --expected-ucx-version="$UCX_VERSION" --expected-ucx-tls="$UCX_TLS" \
  --expected-partition="$PARTITION" --expected-qos="$QOS" \
  --expected-binding-policy="$BINDING_POLICY"
```

`AMREX_BASE_SHA` is `4ceb814db7d79c5858b13dab153fb04ecc3b129d` for this snapshot. The actual package must also contain the [projected-tags patch](../../patches/amrex/README.md); a base SHA does not prove that. Retain patch/configuration digests with results. Passing `unknown` as an expectation does not fix missing metadata.

Raw checkers are [check_performance_csv.py](scripts/check_performance_csv.py) and [check_gpu_performance_csv.py](scripts/check_gpu_performance_csv.py). Their raw-record exit codes are `0` accepted, `1` invalid record, `2` unreadable/input usage failure. Summary/comparison consumers take explicit paths; inspect `--help` for positional arguments. Formal GPU summaries require the complete fixed workload/resource/repeat matrix, not arbitrary CSVs.

## CUDA/MIG launch scope

[gpu-rank-launch.sh](scripts/hpc-slurm/gpu-rank-launch.sh) imports [check_gpu_mapping_log.py](scripts/check_gpu_mapping_log.py). It is a Linux Slurm task-local MIG launcher. Required variables are `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS`, `SLURM_CPUS_PER_TASK`, task-local `CUDA_VISIBLE_DEVICES`, `LBM_GPU_ALLOCATION_CUDA_VISIBLE_DEVICES` and `LBM_GPU_ALLOCATION_CPUSET`. CPU slices must be disjoint and exactly cover the declared allocation, including noncontiguous IDs; each task must be allowed to bind to its slice.

The single-level runtime also requires the context represented by `LBM_GPU_PARENT_UUIDS`, `LBM_GPU_PARENT_GPU_COUNT`, `LBM_GPU_AMREX_CONFIG_SHA256`, `LBM_GPU_NVCC_VERSION`, `LBM_GPU_CUDA_TOOLKIT_VERSION`, `LBM_GPU_MPI`, `LBM_GPU_MPI_CUDA_BUILD_VERSION`, `LBM_GPU_MPI_CUDA_SUPPORT_REPORTED`, `LBM_GPU_QOS`, `LBM_GPU_BINDING_POLICY` and `LBM_GPU_MIG_PROFILE`, with actual `SLURM_JOB_ID`, `HOSTNAME`, `SLURM_JOB_PARTITION` and `SLURM_JOB_GPUS`. Base/prefix/source overrides are `LBM_GPU_AMREX_SOURCE_COMMIT`, `LBM_GPU_AMREX_PREFIX` and `LBM_GPU_GIT_COMMIT`. The launcher sets `LBM_GPU_DEVICE_ASSIGNMENT_SOURCE=slurm-task-local-cvd`; [GpuPerformanceRunSession.H](src/GpuPerformanceRunSession.H) owns the checked contract.

Each rank sees one distinct MIG UUID and uses local CUDA device 0. Preserve one/four-MIG contracts and pinned staging (`amrex.use_gpu_aware_mpi=0`). A positive MPIX capability report does not establish a working device-buffer path. A site supplies compatible allocation/submission commands and records the mapping; platform submission adapters are not in the public snapshot.

`throughput` uses one synchronized complete window and supplies formal MLUPS. `stage_breakdown` is a separate synchronized diagnostic process. Formal Throughput windows must reach 600 seconds; legacy GPU StageBreakdown baseline windows have their own 60–120 second contract. Final finite values and communication fingerprints remain artifact checks. MIG results are not whole-GPU performance. The CSV `driver` field is the CUDA Driver API compatibility version, not the NVIDIA driver release string.

## AMR artifacts

AMR and uniform-fine runs use the same simulation and compare matched physical intervals. Uniform-fine TGV uses twice the AMR level-0 step count; DFG uses four times. Throughput and StageBreakdown run separately. CPU uses 16 ranks/one thread each; CUDA uses the retained one/four-MIG contracts.

`window.csv` owns `window-levels.csv`, `window-boxes.csv`, optional `window-stages.csv` and `window-rank-N.csv`. They record physical windows, valid updates including covered coarse/solid-interior cells, grid owners, stage counts and resources. Device memory is sampled, not a measured peak; process RSS is not a node aggregate.

```bash
python3 tests/performance/scripts/check_amr_performance_csv.py \
  /absolute/path/to/window.csv --source "$PROJECT_SOURCE_SHA" \
  --expected-amrex-source-commit="$AMREX_BASE_SHA" \
  --expected-amrex-prefix="$CPU_AMREX_INSTALL" \
  --expected-amrex-prefix="$CUDA_AMREX_INSTALL"
```

Only allow prefixes from the independent context; a single-backend check normally needs one. `--source` is the LBMReX SHA, distinct from the AMReX base SHA. The AMR checker returns `0` valid, `1` non-finite final state, `2` insufficient/invalid evidence. There is no speedup/scaling-efficiency pass threshold.

[summarize_amr_performance.py](scripts/summarize_amr_performance.py) takes explicit raw paths, `--output`, the same dependency expectations and optional `--calibrate`. Formal summaries require complete groups/repeats and compatible intervals/layouts/contexts. Calibration estimates future windows, not guaranteed durations. Stage shares use a common critical rank, not a sum of unrelated MPI maxima. Scientific accuracy requires independent assessment.

## Host consumers

`cpu_performance_consumers`, `gpu_performance_consumers` and `amr_performance_consumers` are registered synthetic CTest consumers, including in CPU builds. They exercise schemas, expectations and launcher fixtures without performance solvers. Success does not establish actual MPI/GPU mapping, kernels, long-window timing or science; see [validation status](../../docs/validation.md).
