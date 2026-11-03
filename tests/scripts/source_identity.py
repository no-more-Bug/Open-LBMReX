#!/usr/bin/env python3
"""Resolve the source identity shared by CMake and numerical runners."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


def source_identity(root: Path, expected: str = "") -> str:
    environment = os.environ.get("LBM_EXPECTED_GIT_COMMIT", "")
    for value in (expected, environment):
        if value and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("LBM_EXPECTED_GIT_COMMIT must be a full lowercase 40-digit SHA")
    if expected and environment and expected != environment:
        raise ValueError("CMake/CLI and environment LBM_EXPECTED_GIT_COMMIT disagree")
    expected = expected or environment
    root = root.resolve()
    actual = ""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            top, commit = result.stdout.strip().splitlines()
            owner = Path(top).resolve()
            if owner == root or (root.name == "LBMReX" and owner == root.parent):
                relative = (root / "CMakeLists.txt").relative_to(owner).as_posix()
                tracked = subprocess.run(
                    ["git", "-C", str(root), "cat-file", "-e", f"HEAD:{relative}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if tracked.returncode == 0:
                    actual = commit
    except FileNotFoundError:
        pass  # Source archives may be used without Git installed.
    if actual and expected and actual != expected:
        raise ValueError("LBM_EXPECTED_GIT_COMMIT differs from the source repository HEAD")
    return actual or expected or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        identity = source_identity(Path(__file__).resolve().parents[2], args.expected)
        if args.strict and identity == "unknown":
            raise ValueError(
                "source identity is unavailable; set LBM_EXPECTED_GIT_COMMIT "
                "to the public commit corresponding to this source archive"
            )
    except ValueError as exc:
        print(f"source identity: {exc}", file=sys.stderr)
        return 2
    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
