#!/usr/bin/env python3
"""Check CPU consumers with independently specified synthetic records."""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from check_performance_csv import EXPECTED_COLUMNS, TIMING_FIELDS


EXPECTED_CONTEXT = {
    "amrex_source_commit": "4ceb814db7d79c5858b13dab153fb04ecc3b129d",
    "amrex_prefix": "/opt/test-amrex/cpu",
    "compiler": "GNU-13.2.0",
    "mpi": "test-openmpi",
    "ucx_version": "1.20.0",
    "ucx_tls": "^xpmem",
    "partition": "test-cpu",
    "qos": "test-qos",
    "binding_policy": "map-by-core_bind-to-core_rank-by-slot",
}
EXPECTED_ARGS = [
    "--expected-" + field.replace("_", "-") + "=" + value
    for field, value in EXPECTED_CONTEXT.items()
]


def record(mode="smoke", repeat=1):
    nsteps = 20 if mode == "smoke" else 1000
    start = 1 if mode == "smoke" else 101
    window = 2.0 if mode == "smoke" else 600.0 + repeat
    row = {
        **{field: "0" for field in TIMING_FIELDS},
        **EXPECTED_CONTEXT,
        "schema_version": "2", "workload": "tgv_d2q9", "mode": mode,
        "descriptor": "D2Q9", "repeat": str(repeat), "git_commit": "a" * 40,
        "amrex_build_type": "Release", "build_type": "Release",
        "node": "test-node", "grid_x": "1024", "grid_y": "1024", "grid_z": "1",
        "global_valid_cells": "1048576", "max_grid_size": "256", "box_count": "16",
        "mpi_ranks": "8", "omp_threads": "1", "cpuset": "0-7",
        "sample_start": str(start), "sample_end": str(start + nsteps - 1),
        "nsteps": str(nsteps),
        "timing_reduction": "independent_rank_max_with_synchronized_window",
        "rank_max_local_total_window_s": str(window),
        "synchronized_window_s": str(window), "total_s_per_step": str(window / nsteps),
        "mlups": str(1048576 * nsteps / window / 1e6),
    }
    return row


def write(path, row):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def run(code, *args):
    result = subprocess.run([sys.executable, "-B", *map(str, args)], text=True, capture_output=True)
    assert result.returncode == code, (args, result.returncode, result.stdout, result.stderr)


def main():
    scripts = Path(__file__).resolve().parent
    scratch = Path(os.environ.get("TMPDIR", scripts / "build"))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpu-consumers-", dir=scratch) as temporary:
        root = Path(temporary)
        checker = scripts / "check_performance_csv.py"
        raw = root / "raw.csv"
        write(raw, record())
        run(0, checker, *EXPECTED_ARGS, raw)
        run(2, checker, raw)
        run(2, checker, *EXPECTED_ARGS, root / "missing.csv")
        for field in EXPECTED_CONTEXT:
            row = record()
            row[field] = "d" * 40 if field == "amrex_source_commit" else "/different" if field == "amrex_prefix" else "different"
            write(raw, row)
            run(1, checker, *EXPECTED_ARGS, raw)
        row = record()
        row["nsteps"] = "21"
        write(raw, row)
        run(1, checker, *EXPECTED_ARGS, raw)
        row = record()
        row["mlups"] = "1"
        write(raw, row)
        run(1, checker, *EXPECTED_ARGS, raw)
        row = record()
        row["compiler"] = "unset"
        write(raw, row)
        unknown_args = [value if not value.startswith("--expected-compiler=") else "--expected-compiler=unset"
                        for value in EXPECTED_ARGS]
        run(1, checker, *unknown_args, raw)
        records = []
        for repeat in (1, 2, 3):
            path = root / ("baseline-" + str(repeat) + ".csv")
            write(path, record("baseline", repeat))
            records.append(path)
        summary = root / "summary.csv"
        consumer = scripts / "summarize_cpu_mpi_baseline.py"
        run(0, consumer, *EXPECTED_ARGS, "--output", summary, *records)
        with summary.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]["amrex_source_commit"] == EXPECTED_CONTEXT["amrex_source_commit"]
        wrong = [value if not value.startswith("--expected-compiler=") else "--expected-compiler=different"
                 for value in EXPECTED_ARGS]
        run(1, consumer, *wrong, "--output", root / "bad-summary.csv", *records)
    print("CPU consumers PASS: explicit new-context positive, all nine context mismatches, missing arguments/file, window/MLUPS negatives, summary propagation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
