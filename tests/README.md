# Formal tests

This directory owns the formal test compositions, fixtures, inputs, references and artifact consumers. Production code does not depend on the testing tree. Scientific showcases own separate application assets. Performance AMR compositions intentionally reuse numerical test compositions; that dependency is retained.

Configure tests from the repository root with `BUILD_TESTING=ON` using [the build instructions](../docs/building.md). `LBM_BUILD_SHOWCASES=OFF` disables showcase targets. Physical source isolation is a separate [validation requirement](../docs/validation.md), not something established by that switch alone.

## Layers and selection

| Layer | Scope |
| --- | --- |
| `unit/` | Small deterministic module/interface checks, including public-header linkage. |
| [integration/](integration/README.md) | Real AMReX objects, runtime, transfer, AMR, boundary and force observables; host synthetic consumers where explicitly labelled. |
| `regression/` | Short user-visible executable/artifact behavior, failure contracts and parallel comparisons. |
| [numerical/](numerical/README.md) | Explicit longer analytic/literature comparisons and engineering counterparts. |
| [performance/](performance/README.md) | Explicit performance executables plus registered host consumer tests. Executables are not part of the default build. |

The actual test list depends on backend and parallel options. Discover it instead of relying on a historical count:

```bash
ctest --test-dir build/cpu -N
ctest --test-dir build/cpu -L unit --output-on-failure -j1
ctest --test-dir build/cpu -N -L integration
```

Plain `ctest` runs all registered tests, including long numerical work. Select named tests or labels after checking their rank/thread requirements and prerequisites. `fast`/`short` includes more than pure C++ unit tests; some consumers require NumPy or Linux affinity. Do not run multiple independent CTest drivers against one build tree because tests own and may clean their own work paths.

## Backend and resource controls

`LBM_ENABLE_CUDA=OFF/ON` selects CPU/CUDA compilation. `LBM_TEST_OPENMP` controls optional OpenMP/hybrid identities; use `OFF` for the documented pure-MPI route. `LBM_TEST_AMR_MPI_RANKS` and `LBM_TEST_R5_MPI_RANKS` affect their documented groups, not every test. Some identities have their own fixed rank counts.

CTest sets build-local work and temporary directories and explicit OpenMP settings. Non-hybrid CPU MPI runs use one thread per rank and core binding. `LBM_TEST_MPI_LAUNCHER` accepts an explicit tests-owned/environment-compatible launcher; if unset, CMake's MPI executable is used. Site-specific submission scripts are not prerequisites of the public build.

CUDA tests use pinned staging with `amrex.use_gpu_aware_mpi=0`. Device visibility and rank mapping must be correct in the execution environment. The host `gpu_performance_consumers` fixture also runs on a CPU build and is not GPU kernel evidence.

## Source identity

CMake and the AMR numerical runners share [source_identity.py](scripts/source_identity.py). A clone uses the containing source repository's committed HEAD; an ordinary archive accepts an explicit public SHA through `LBM_EXPECTED_GIT_COMMIT`. The optional CMake cache input has the same name. Cache/environment values must agree, and an expected value in a clone must equal HEAD. An unrelated ancestor repository is not a source identity.

Ordinary builds may record `unknown`; strict numerical runners return `2` when identity is missing, malformed or inconsistent. Existing integration writers consume the environment value. CTest receives it automatically after configuration; manually launched integration executables need the environment explicitly.

`source_identity` is an integration/synthetic tool test requiring Git, Python, NumPy, Bash and CMake. It exercises small Git fixtures, CMake script-mode output and actual runner arguments with a non-computing launcher. It does not run a solver.

## Artifacts and interpretation

CTest writes under each build tree's test-specific `work/` directory. Explicit `TMPDIR` takes priority; affected standalone synthetic checkers fall back to their own `build/consumer-tests`. Cleanup is limited to the test's temporary descendants and never owns showcase research outputs.

The [numerical contract](numerical/README.md) distinguishes scientific `0/1/2` from bounded engineering modes. Host fixtures, unit checks and short boundary/AMR runs do not establish full scientific acceptance or performance advantage. Current [limitations](../docs/known-limitations.md) and [standalone snapshot validation status](../docs/validation.md) apply.

Reference CSVs retain their independent test/showcase owners and [source attribution](../docs/references.md); do not merge or regenerate them as an incidental cleanup.
