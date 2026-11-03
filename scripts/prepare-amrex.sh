#!/usr/bin/env bash
# Prepare a separate patched source tree; never build or modify the submodule.
set -euo pipefail

if (( $# != 1 )); then
    printf 'Usage: %s <new-source-directory>\n' "$0" >&2
    exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_tree="$project_root/external/amrex"
patch_file="$project_root/patches/amrex/0001-amrmesh-deduplicate-projected-tags.patch"
base_sha=4ceb814db7d79c5858b13dab153fb04ecc3b129d
patch_sha=32474c1a9f61ac7bbae0aa34b046bca3ad21e610d2587c6d99b0f1d5113ab4b4
destination="$1"

[[ -e "$source_tree/.git" ]] || {
    printf 'Initialize the pinned external/amrex submodule first.\n' >&2
    exit 2
}
[[ "$(git -C "$source_tree" rev-parse HEAD)" == "$base_sha" ]] || {
    printf 'AMReX submodule is not at the pinned commit %s.\n' "$base_sha" >&2
    exit 2
}
[[ -z "$(git -C "$source_tree" status --porcelain)" ]] || {
    printf 'AMReX submodule has local changes; leave them intact.\n' >&2
    exit 2
}
[[ "$(sha256sum "$patch_file" | cut -d ' ' -f 1)" == "$patch_sha" ]] || {
    printf 'AMReX patch checksum differs from the pinned patch.\n' >&2
    exit 2
}
[[ ! -e "$destination" && ! -L "$destination" ]] || {
    printf 'Destination already exists; use a new source directory: %s\n' "$destination" >&2
    exit 2
}

destination="$(realpath -m -- "$destination")"
source_tree="$(realpath -- "$source_tree")"
case "$destination/" in
    "$source_tree/"*)
        printf 'Destination must be outside the AMReX submodule.\n' >&2
        exit 2
        ;;
esac

git clone --no-hardlinks --no-checkout "$source_tree" "$destination"
git -C "$destination" checkout --detach "$base_sha"
git -C "$destination" apply --check "$patch_file"
git -C "$destination" apply "$patch_file"
printf 'base_sha=%s\npatch_sha256=%s\npatch_file=%s\n' \
    "$base_sha" "$patch_sha" "$(basename "$patch_file")" \
    > "$destination/.lbmrex-dependency"
printf 'Prepared %s; use separate CPU/CUDA build and install directories.\n' "$destination"
