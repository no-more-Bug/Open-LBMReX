#!/usr/bin/env python3
"""Require one subprocess to fail and publish an expected diagnostic."""

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contains", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("missing subprocess command")
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode == 0:
        raise SystemExit("expected subprocess failure, got exit status 0")
    if args.contains not in completed.stdout:
        raise SystemExit(
            f"expected diagnostic {args.contains!r} was not published"
        )
    print(
        f"expected_process_failure passed status={completed.returncode} "
        f"diagnostic={args.contains!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
