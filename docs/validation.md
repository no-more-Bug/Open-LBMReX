# Validation scope

## What the present evidence establishes

Previous development runs have tested CPU-side MPI parallel execution. Due to limited computing resources, GPU-side multi-device parallel tests have only used NVIDIA MIG slice configurations. Execution across multiple full physical GPUs has not been validated and is not guaranteed to run successfully or pass tests. These results apply to their tested revisions and environments; they do not establish validation of the new standalone snapshot or its AMReX packages.

The preparation and portability tools have host-side checks for dependency preparation, source-identity resolution, test scratch ownership, and CPU/GPU/AMR artifact consumers. Those checks include synthetic inputs and intentionally invalid records. They do not execute a CFD solver or establish a new package's GPU behavior.

This standalone snapshot is still awaiting independent build/run validation with the pinned official AMReX source plus its patch. No release-wide pass count is claimed. Exact platform, compiler, MPI/CUDA versions, source revision, dependency configuration, selected tests and limitations must accompany any future published result.

## Pending standalone-snapshot checks

| Route | Required evidence before claiming it validated |
| --- | --- |
| Linux CPU/MPI | Fresh compatible AMReX package and LBMReX build; selected unit, integration, boundary and AMR checks with explicit ranks/threads/affinity. |
| NVIDIA CUDA | Separate AMReX and LBMReX builds, package/EB/device checks and selected real GPU/MPI execution. Host Python consumers alone are insufficient. |
| Examples | Build all five showcases, run bounded examples and check their basic artifacts. This does not replace full literature horizons. |
| Tests without showcases | A dedicated source copy physically lacking showcases, configured with `LBM_BUILD_SHOWCASES=OFF`, with the selected tests built and run. Disabling compilation alone is not proof of asset independence. |
| Performance tools | Explicit target builds and short real records using independently supplied package/platform expectations. No full benchmarking or speedup claim is implied. |
| Ordinary source archive | A source tree without `.git`, a compatible external AMReX package, and explicit public SHA propagation for strict runners. |

The [build commands](building.md) have been checked against the current interfaces. The pending checks above have not been run on this standalone snapshot. The current documentation is not a claim that all compiler, MPI or CUDA combinations are supported. HIP, SYCL and native Windows builds are outside the stated scope.

## Engineering and scientific outcomes

Tests own their observable expectations independently of production code and showcases. Short engineering tests cover initialization, layouts, transfers, runtime/boundary behavior and artifacts. The [numerical suite](../tests/numerical/README.md) owns longer analytic/literature comparisons and explicit reference eligibility.

For scientific numerical runners, `0` means valid complete evidence meeting the selected criteria, `1` means valid complete evidence outside at least one scientific criterion, and `2` means invalid or insufficient evidence. A documented engineering/short mode has its own bounded meaning and cannot turn an earlier scientific `1` or `2` into success. Performance tools retain their own documented status contracts.

Keep original errors, thresholds, incomplete runs and the [known limitations](known-limitations.md) visible when reporting results. The software's prerelease version number is not a measure of scientific completion.
