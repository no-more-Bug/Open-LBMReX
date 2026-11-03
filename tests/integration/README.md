# Integration tests

Integration tests exercise observable behavior through production interfaces using real AMReX geometry, layouts, distributed fields, runtime and boundary objects. Configure this directory through the repository-root CMake project; it is not a separate CMake project.

| Location | Responsibilities |
| --- | --- |
| `lbm/` | Geometry/BoxArray/DistributionMapping, MultiFab/iMultiFab, field operators, domain boundaries, EB metadata, internal reconstruction and collective force behavior. |
| `runtime/` | Preparation, run lifecycle, state publication, static/dynamic AMR, transfers, boundary lifecycle and force sampling. |
| `app/` | Application observables, initialization and descriptor-independent profiling support. |
| `inputs/` | Test-owned input identities and saved candidate geometry fixtures. |

Use [tests/README.md](../README.md) for common build, source-identity and resource rules. Exact targets and registrations are in [CMakeLists.txt](CMakeLists.txt). For example, after the appropriate targets have been built:

```bash
ctest --test-dir build/cpu -N -R '^static_hierarchy_'
ctest --test-dir build/cpu -R '^static_hierarchy_.*serial$' \
  --output-on-failure -j1
```

Inspect the selected names and prerequisites first. A target name containing `single_level` does not imply that every consumer exercises only a single AMR level.

## Static and evolving AMR

Static checks cover normalized hierarchy geometry, periodic projections, fixed-thickness domains, output/readback and one-level degeneration. Coarse/fine exchange and advance tests compare actual populations with tests-owned forward-push/donor expectations, rather than calling the production operation to manufacture its expected result.

Evolution artifacts describe actual global BoxArrays, DistributionMappings, populations, steps and physical times. Composite mass and momentum integrate valid cells not covered by the adjacent finer level; do not double-count covered coarse cells or MPI-reduce data that has already been globally gathered. D2 area is multiplied by the common physical thickness once; D3 uses volume.

Donor/witness records distinguish the logical dependency origin from communication implementation. A remote donor is evidence of a cross-rank dependency, not a count of MPI messages. Counts alone do not prove the population-transfer result.

## Dynamic grids and boundaries

Dynamic checks cover candidate preparation/clustering, population remap, synchronized regrid, refinement constraints, invalid-domain handling, boundary relayout and force publication. [Saved initial candidates](inputs/dynamic_regrid_clustering/README.md) are test inputs, not a final-grid oracle.

Fixed geometry constraints and dynamic tags contribute to candidate finalization. Complete internal boundaries must satisfy their level/coverage constraints; boundaries in one force object can reside at different levels. Force records, population step/time and the point at which samples are extracted relative to regrid are part of the observable contract.

These tests establish bounded engineering properties. They do not close the dynamic granularity, force-sawtooth or long-run scientific limitations in [known limitations](../../docs/known-limitations.md).

## Output and visualization

Tests use their own build-local work directories. Consumers validate identities, field/schema completeness, geometry and comparisons before interpreting results. Native AMR readback and ParaView are different consumers; a plausible rendered image is not a substitute for native coverage or composite-integral checks.

`runtime/check_static_hierarchy_paraview.py` is an optional `pvpython` consumer with explicit root/report/source-id parameters and optional rendering. Thin-slab covered-coarse masking can remain incorrect in the external reader. GUI inspection does not change an independently reported masking failure. Standalone snapshot execution of these routes is [pending](../../docs/validation.md).
