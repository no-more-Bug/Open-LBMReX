# Known limitations

This is a research codebase. Numerical validity depends on the composition, resolution, boundary placement, time horizon, reference eligibility and execution environment. A build, finite result or short engineering check does not establish all of these.

| Area | Current limitation |
| --- | --- |
| Dynamic AMR | Experimental. Box granularity, thin blocks and coverage/cost tradeoffs need improvement. Regridding can enlarge fine coverage; improved clustering is not a general speedup claim. |
| Re100 cylinder comparisons | Existing evidence includes scientific failures on uniform meshes and AMR results with insufficient reference/cycle evidence. Some CUDA quasi-D3 fine runs remain deferred. Do not report the whole Re100 matrix as passing. |
| AMR force histories | Sawtooth behavior remains unresolved. Its root cause has not been established and is not attributed simply to insufficient run time. |
| Dynamic-boundary long runs | CPU and GPU evidence is uneven. Some single-cylinder GPU runs were cancelled or timed out; some tandem single-GPU/quasi-D3 long runs were deferred. Short or budget runs do not fill those gaps. |
| Three-dimensional claims | D3Q27 engineering tests and quasi-two-dimensional benchmarks are distinct from fully three-dimensional scientific validation. |
| Cavity comparisons | The showcase checker retains its own 6% centerline criterion and known Guo failures. The AMR numerical checker uses a separate 6.5% centerline budget; earlier failures under 6% are not erased or described as rerun successes. |
| Parallel validation scope | Independent CPU MPI and CUDA engineering checks are recorded for this snapshot. Limited computing resources have restricted GPU multi-device testing to NVIDIA MIG slice configurations; this snapshot used one physical A100. Parallel execution across multiple full physical GPUs remains unvalidated and is not guaranteed to run successfully or pass tests. See the exact [tested scope](validation.md). |
| GPU communication | The intended route is pinned staging with `amrex.use_gpu_aware_mpi=0`. Device-buffer/GPU-aware MPI is not generally supported by the available evidence. |
| Visualization | Thin AMR slabs can expose covered-coarse masking limitations in ParaView. A visually plausible image or GUI inspection does not override native coverage checks or establish a valid composite-domain integral. |
| Release packaging | Selected standalone builds/runs and a CPU ordinary-source-archive route have passed. Other toolchains, OpenMP/hybrid execution, CUDA source archives and full scientific matrices are outside this release's evidence. See [validation](validation.md). |

Scientific checkers keep the distinction between valid results outside the criterion and invalid/incomplete evidence. Do not relax a threshold, substitute a reference, relabel a cancelled run as complete, or interpret a numerical exit status without the case's documented contract.

The geometry and boundary contracts remain case-specific. In the AMR tests, D2 uses one z cell with refinement ratio `(2,2,1)`; D3 uses `(2,2,2)`. The single-level cavity showcase can use multiple z cells even for its D2 descriptor. Each complete internal boundary is constrained to an appropriate level; distinct boundaries belonging to one force object can have different levels. The code does not make arbitrary geometry/AMR combinations automatically valid.
