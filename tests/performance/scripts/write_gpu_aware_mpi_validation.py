#!/usr/bin/env python3
"""Write the explicit OFF/ON communication-smoke decision artifact."""

import argparse
import csv
from pathlib import Path
import re


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--off-job-id", required=True)
    parser.add_argument("--off-status", required=True, choices=("pass", "fail"))
    parser.add_argument("--off-evidence", required=True)
    parser.add_argument("--on-job-id", required=True)
    parser.add_argument("--on-status", required=True, choices=("pass", "fail"))
    parser.add_argument("--on-evidence", required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        if re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is None:
            raise ValueError("git commit must be a full lowercase SHA")
        rows = []
        for flag, mode, job_id, status, evidence in (
            (0, "pinned_staging", args.off_job_id, args.off_status, args.off_evidence),
            (1, "device_buffer", args.on_job_id, args.on_status, args.on_evidence),
        ):
            if not job_id or not evidence:
                raise ValueError("job id and evidence path must be non-empty")
            rows.append({
                "schema_version": 1,
                "git_commit": args.git_commit,
                "workload": "tgv_d2q9",
                "layout": "4mig",
                "amrex_use_gpu_aware_mpi": flag,
                "communication_mode": mode,
                "job_id": job_id,
                "status": status,
                "formal_candidate": "true" if status == "pass" else "false",
                "evidence": evidence,
            })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 2
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"gpu_aware_mpi_validation passed rows=2 output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
