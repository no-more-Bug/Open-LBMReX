#!/usr/bin/env python3
"""Clean explicit test-local outputs, then require one process to succeed."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def clean_test_local_path(path: Path) -> None:
    working_directory = Path.cwd().resolve()
    requested = path if path.is_absolute() else working_directory / path
    resolved = requested.resolve(strict=False)
    if resolved == working_directory or not resolved.is_relative_to(working_directory):
        raise SystemExit(f"refusing to clean path outside test work directory: {path}")
    if requested.is_symlink() or requested.is_file():
        requested.unlink()
    elif requested.is_dir():
        shutil.rmtree(requested)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="append", default=[], type=Path)
    parser.add_argument("--contains")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing subprocess command")
    for path in args.clean:
        clean_test_local_path(path)
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(f"regression process failed with status {completed.returncode}")
    if args.contains is not None and args.contains not in completed.stdout:
        raise SystemExit(f"expected diagnostic {args.contains!r} was not published")
    print(f"regression_process passed status={completed.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
