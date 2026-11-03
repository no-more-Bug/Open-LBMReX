# Changelog

## Unreleased — preparing v0.1.0

This is the initial public-repository source candidate. No released version or project DOI is claimed.

### Included

- The LBMReX codebase with D2Q9 and D3Q27 case compositions, cell-centered block-structured AMR, boundary handling and force observations.
- Five single-level showcases, separate test-owned multilevel cases, numerical checkers and performance tools.
- An official AMReX submodule pinned to `4ceb814db7d79c5858b13dab153fb04ecc3b129d`, with a projected-tag deduplication patch applied to a separate preparation copy.
- An installation-package build route, English documentation and a Chinese README, reference attribution, and AI-generated architecture and class diagrams.

### Validation and limitations

Independent build/run validation of this standalone candidate and its new AMReX installation packages is still pending. Historical development results do not validate the new release packages, and no release-wide pass count is claimed.

Dynamic AMR remains experimental. Grid granularity and coverage costs need improvement; some Re100 comparisons fail scientific criteria or lack sufficient evidence; the root cause of sawtooth behavior in AMR force histories is unresolved. GPU long-run evidence is incomplete.

CPU MPI execution has been tested during development. Limited computing resources restricted GPU multi-device testing to NVIDIA MIG slice configurations; execution across multiple full physical GPUs has not been validated and is not guaranteed to pass.

See [validation](docs/validation.md) and [known limitations](docs/known-limitations.md) for the applicable scope. This hobby project provides no guarantee of correctness or continued updates.
