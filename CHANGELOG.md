# Changelog

## 0.1.0 — 2026-11-04

Initial curated source snapshot with independent repository history.

### Included

- The LBMReX codebase with D2Q9 and D3Q27 case compositions, cell-centered block-structured AMR, boundary handling and force observations.
- Five single-level showcases, separate test-owned multilevel cases, numerical checkers and performance tools.
- An official AMReX submodule pinned to `4ceb814db7d79c5858b13dab153fb04ecc3b129d`, with a projected-tag deduplication patch applied to a separate preparation copy.
- An installation-package build route, English documentation and a Chinese README, reference attribution, and AI-generated architecture and class diagrams.
- Source identity support for clones and ordinary archives, portable artifact consumers, and CUDA compilation settings for all showcase executables.

### Validation and limitations

Selected standalone checks passed with fresh CPU/CUDA AMReX packages. The CPU 90-test and CUDA 80-test matrices ran at `e90c004`; CUDA showcase/performance checks and an ordinary CPU source archive ran at `fe75e2b` after the four-showcase CMake correction. The version commit adds documentation and citation metadata to the tested code. See [validation](docs/validation.md) for exact source identities, toolchains, test selection and the basis for retaining the earlier matrix evidence.

Five D2 showcases ran on CPU and CUDA. Fourteen D2 performance smoke processes exercised real artifacts and consumers; they are not formal benchmarks or speedup results. D3 showcase/performance targets were compiled, but their runtime behavior was not tested in this release validation.

Dynamic AMR remains experimental. Grid granularity and coverage costs need improvement; some Re100 comparisons fail scientific criteria or lack sufficient evidence; the root cause of sawtooth behavior in AMR force histories is unresolved. GPU long-run evidence is incomplete.

Limited computing resources restricted GPU multi-device validation to MIG instances on one physical A100. Full physical multi-GPU, OpenMP/hybrid, complete scientific long-run and formal performance matrices remain outside this release's evidence.

See [known limitations](docs/known-limitations.md). This hobby project provides no guarantee of correctness or continued updates, and no project DOI is claimed.
