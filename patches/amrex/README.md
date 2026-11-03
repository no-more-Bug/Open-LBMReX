# AMReX projected-tag deduplication patch

LBMReX uses the official AMReX commit `4ceb814db7d79c5858b13dab153fb04ecc3b129d` and the patch [0001-amrmesh-deduplicate-projected-tags.patch](0001-amrmesh-deduplicate-projected-tags.patch).

Patch SHA256: `32474c1a9f61ac7bbae0aa34b046bca3ad21e610d2587c6d99b0f1d5113ab4b4`.

The patch adds sorting and duplicate removal to the combined ordinary/projected tag coordinates on the IOProcessor path in `AmrMesh::MakeNewGrids`, before `ClusterList` construction. It changes two source lines and does not add diagnostic printing or alter the application algorithm.

## Prepare a separate source tree

From the project root, with the submodule initialized:

```bash
git submodule update --init -- external/amrex
git -C external/amrex rev-parse HEAD
sha256sum patches/amrex/0001-amrmesh-deduplicate-projected-tags.patch
mkdir -p build/dependencies
bash scripts/prepare-amrex.sh "$PWD/build/dependencies/amrex-source"
```

Check both identities against the values above. The [preparation script](../../scripts/prepare-amrex.sh) requires a clean pinned submodule, refuses existing destinations and paths within the submodule, clones into a separate tree and applies the patch there. It leaves any failed preparation directory available for diagnosis instead of deleting it automatically.

The preparation tree's `.lbmrex-dependency` records the base SHA, patch digest and filename. CPU/CUDA build and install directories remain separate; their full package configurations and toolchains also belong in provenance. Follow [building](../../docs/building.md). The application continues to use `find_package(AMReX)`; it does not implicitly patch the submodule during configure or execution.

Successful patch application is not proof of a validated CPU/CUDA installation. The standalone snapshot's package/build/run checks remain [pending](../../docs/validation.md). Do not apply this patch blindly to a different AMReX revision.

## Upstream notices

The patch contains AMReX source context. Complete notices from the fixed upstream revision are included here so that an ordinary source archive retains them even without an expanded submodule: [LICENSE](LICENSE), [NOTICE](NOTICE), and [CITATION](CITATION). These files are preserved verbatim; upstream authors and copyright are not replaced with the project's maintainer identity.

The official dependency retains its full source and notices. See also [third-party material](../../docs/third-party.md).
