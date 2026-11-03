#!/usr/bin/env python3
"""Extract and evaluate DFG Re100 metrics from tests-owned force history."""

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

CYCLES_USED = 3
MIN_SAMPLES = 200
MAX_DRIFT = 0.01
REFERENCE_SHA256 = "f1ff94190e21e78f8c313018e3376da20ffeb5c261b399bb5c1f2663bddf2d2f"
REQUIRED_COLUMNS = {
    "population_step", "force_time_lb", "t_star", "reference_label",
    "object_name", "object_id", "Cd", "Cl",
}
OUTPUT_COLUMNS = (
    "analysis_status", "message", "reference_label", "object_name", "object_id",
    "history_path", "warmup_t_star", "total_samples", "completed_cycles",
    "cycles_used", "min_samples_in_used_cycles", "cycle_start_population_step",
    "cycle_end_population_step", "cycle_start_t_star", "cycle_end_t_star",
    "samples_in_cycle", "Cd_min", "Cd_max", "Cd_midrange", "Cd_peak_to_peak",
    "Cl_min", "Cl_max", "Cl_midrange", "Cl_peak_to_peak", "period_t_star", "St",
    "max_period_relative_drift", "max_Cd_extrema_relative_drift",
    "max_Cl_extrema_relative_drift", "St_relative_error", "st_status",
    "combined_status", "failed_gates",
)
METRICS_COLUMNS = [
    "reference_label", "descriptor", "reynolds", "diameter_intervals",
    "depth_cells", "sample_count", "final_population_step", "Cd_final",
    "Cl_final", "C_delta_p", "max_abs_Cz", "steady_reached",
    "steady_tolerance", "final_steady_residual",
]


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    step: int
    force_time: float
    t_star: float
    cd: float
    cl: float


@dataclass(frozen=True)
class Extremum:
    index: int
    time: float
    value: float


@dataclass(frozen=True)
class Cycle:
    start: Extremum
    end: Extremum
    samples: int
    cd_min: float
    cd_max: float
    cl_min: float
    cl_max: float

    @property
    def period(self) -> float:
        return self.end.time - self.start.time


def read_history(path: Path) -> tuple[tuple[str, str, int], list[Sample]]:
    if not path.is_file():
        raise EvidenceError(f"missing history: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or REQUIRED_COLUMNS.difference(fields):
            raise EvidenceError("history schema is missing or has duplicate columns")
        identity = None
        samples = []
        for line, row in enumerate(reader, start=2):
            try:
                row_identity = (
                    row["reference_label"], row["object_name"], int(row["object_id"])
                )
                sample = Sample(
                    int(row["population_step"]), float(row["force_time_lb"]),
                    float(row["t_star"]), float(row["Cd"]), float(row["Cl"]),
                )
            except (TypeError, ValueError) as exception:
                raise EvidenceError(f"line {line}: invalid history value") from exception
            if not all(math.isfinite(value) for value in (
                sample.force_time, sample.t_star, sample.cd, sample.cl
            )):
                raise EvidenceError(f"line {line}: non-finite history value")
            if identity is None:
                identity = row_identity
            elif identity != row_identity:
                raise EvidenceError(f"line {line}: object identity changed")
            if samples and (
                sample.step <= samples[-1].step
                or sample.force_time <= samples[-1].force_time
                or sample.t_star <= samples[-1].t_star
            ):
                raise EvidenceError(f"line {line}: stamps must increase")
            samples.append(sample)
    if identity != ("dfg_2d_2_re100", "cylinder", 0) or not samples:
        raise EvidenceError("history identity is not DFG Re100 cylinder:0")
    return identity, samples


def quadratic(samples: list[Sample], index: int, field: str, minimum: bool) -> Extremum:
    center = samples[index]
    center_value = getattr(center, field)
    if index == 0 or index + 1 == len(samples):
        return Extremum(index, center.t_star, center_value)
    left, right = samples[index - 1], samples[index + 1]
    x0, x2 = left.t_star - center.t_star, right.t_star - center.t_star
    y0 = getattr(left, field) - center_value
    y2 = getattr(right, field) - center_value
    curvature = (y0 / x0 - y2 / x2) / (x0 - x2)
    slope = y0 / x0 - curvature * x0
    if curvature == 0.0 or (minimum and curvature < 0.0) or (not minimum and curvature > 0.0):
        return Extremum(index, center.t_star, center_value)
    offset = -slope / (2.0 * curvature)
    if offset < x0 or offset > x2:
        return Extremum(index, center.t_star, center_value)
    return Extremum(index, center.t_star + offset,
                    center_value + slope * offset + curvature * offset * offset)


def range_extremum(samples: list[Sample], begin: int, end: int,
                   field: str, minimum: bool) -> float:
    chooser = min if minimum else max
    index = chooser(range(begin, end + 1), key=lambda item: getattr(samples[item], field))
    return quadratic(samples, index, field, minimum).value


def make_cycle(samples: list[Sample], start: Extremum, end: Extremum) -> Cycle:
    return Cycle(
        start, end, end.index - start.index + 1,
        range_extremum(samples, start.index, end.index, "cd", True),
        range_extremum(samples, start.index, end.index, "cd", False),
        range_extremum(samples, start.index, end.index, "cl", True),
        range_extremum(samples, start.index, end.index, "cl", False),
    )


def drift(values: list[float]) -> float:
    return max(
        abs(current - previous) / max(abs(current), abs(previous), 1.0e-300)
        for previous, current in zip(values, values[1:])
    )


def analyze(identity: tuple[str, str, int], samples: list[Sample],
            history: Path, warmup: float) -> dict[str, object]:
    result = {column: "" for column in OUTPUT_COLUMNS}
    result.update(
        reference_label=identity[0], object_name=identity[1], object_id=identity[2],
        history_path=str(history), warmup_t_star=warmup, total_samples=len(samples),
        st_status="not_evaluated", combined_status="not_evaluated",
    )
    minima = [
        quadratic(samples, index, "cl", True)
        for index in range(1, len(samples) - 1)
        if samples[index].cl < samples[index - 1].cl
        and samples[index].cl < samples[index + 1].cl
    ]
    cycles = [
        make_cycle(samples, start, end)
        for start, end in zip(minima, minima[1:])
        if start.time >= warmup
    ]
    result["completed_cycles"] = len(cycles)
    if len(cycles) < CYCLES_USED:
        result.update(
            analysis_status="insufficient_cycles",
            message=f"need {CYCLES_USED} completed post-warmup cycles; found {len(cycles)}",
            cycles_used=0,
        )
        return result
    used = cycles[-CYCLES_USED:]
    minimum_samples = min(cycle.samples for cycle in used)
    result.update(cycles_used=CYCLES_USED, min_samples_in_used_cycles=minimum_samples)
    if minimum_samples < MIN_SAMPLES:
        result.update(
            analysis_status="insufficient_samples",
            message=f"each selected cycle needs {MIN_SAMPLES} samples; minimum is {minimum_samples}",
        )
        return result
    selected = used[-1]
    result.update(
        analysis_status="ok",
        message="periodic metrics extracted from the last three completed cycles",
        cycle_start_population_step=samples[selected.start.index].step,
        cycle_end_population_step=samples[selected.end.index].step,
        cycle_start_t_star=selected.start.time,
        cycle_end_t_star=selected.end.time,
        samples_in_cycle=selected.samples,
        Cd_min=selected.cd_min, Cd_max=selected.cd_max,
        Cd_midrange=0.5 * (selected.cd_min + selected.cd_max),
        Cd_peak_to_peak=selected.cd_max - selected.cd_min,
        Cl_min=selected.cl_min, Cl_max=selected.cl_max,
        Cl_midrange=0.5 * (selected.cl_min + selected.cl_max),
        Cl_peak_to_peak=selected.cl_max - selected.cl_min,
        period_t_star=selected.period, St=1.0 / selected.period,
        max_period_relative_drift=drift([cycle.period for cycle in used]),
        max_Cd_extrema_relative_drift=max(
            drift([cycle.cd_min for cycle in used]),
            drift([cycle.cd_max for cycle in used]),
        ),
        max_Cl_extrema_relative_drift=max(
            drift([cycle.cl_min for cycle in used]),
            drift([cycle.cl_max for cycle in used]),
        ),
    )
    return result


def read_reference(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise EvidenceError("DFG Re100 reference is missing or its checksum changed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        "St", "Cd_min", "Cd_max", "Cd_midrange", "Cd_peak_to_peak",
        "Cl_min", "Cl_max", "Cl_peak_to_peak",
    }
    if {row["metric"] for row in rows} != expected:
        raise EvidenceError("DFG Re100 reference rows changed")
    return {
        row["metric"]: (float(row["value"]), float(row["relative_tolerance"]))
        for row in rows
    }


def validate_completion(path: Path, diameter: int, steps: int) -> None:
    if not path.is_file():
        raise EvidenceError(f"missing metrics: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if reader.fieldnames != METRICS_COLUMNS or len(rows) != 1:
        raise EvidenceError("Re100 metrics schema or row count changed")
    row = rows[0]
    expected = {
        "reference_label": "dfg_2d_2_re100", "descriptor": "D2Q9",
        "reynolds": "100", "diameter_intervals": str(diameter),
        "depth_cells": "1", "sample_count": str(steps),
        "final_population_step": str(steps), "steady_reached": "false",
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise EvidenceError("Re100 scientific identity or completion changed")
    for key in (
        "Cd_final", "Cl_final", "C_delta_p", "max_abs_Cz",
        "steady_tolerance", "final_steady_residual",
    ):
        if not math.isfinite(float(row[key])):
            raise EvidenceError(f"Re100 metrics {key} is not finite")


def evaluate(result: dict[str, object], reference: dict[str, tuple[float, float]]) -> None:
    if result["analysis_status"] != "ok":
        return
    failures = []
    errors = {}
    for metric, (expected, tolerance) in reference.items():
        actual = float(result[metric])
        relative_error = abs(actual - expected) / abs(expected)
        errors[metric] = relative_error
        if relative_error > tolerance:
            failures.append(f"{metric}:{relative_error:.12g}>{tolerance:.12g}")
    period_ok = float(result["max_period_relative_drift"]) <= MAX_DRIFT
    cd_ok = float(result["max_Cd_extrema_relative_drift"]) <= MAX_DRIFT
    cl_ok = float(result["max_Cl_extrema_relative_drift"]) <= MAX_DRIFT
    if not period_ok:
        failures.append("period_drift")
    if not cd_ok:
        failures.append("Cd_extrema_drift")
    if not cl_ok:
        failures.append("Cl_extrema_drift")
    st_pass = period_ok and errors["St"] <= reference["St"][1]
    combined_pass = st_pass and cd_ok and cl_ok and not failures
    result.update(
        St_relative_error=errors["St"], st_status="pass" if st_pass else "fail",
        combined_status="pass" if combined_pass else "fail",
        failed_gates=";".join(failures),
    )


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(result)


def main() -> int:
    if len(sys.argv) != 7:
        return 2
    history, reference, output = map(Path, sys.argv[1:4])
    try:
        warmup = float(sys.argv[4])
        expected_steps = int(sys.argv[5])
        expected_diameter = int(sys.argv[6])
        if not math.isfinite(warmup) or warmup < 0.0:
            raise EvidenceError("warmup must be finite and nonnegative")
        identity, samples = read_history(history)
        if (
            len(samples) != expected_steps
            or samples[0].step != 1
            or samples[-1].step != expected_steps
        ):
            raise EvidenceError("Re100 history is incomplete or has the wrong horizon")
        validate_completion(history.with_name("channel_cylinder_metrics.csv"),
                            expected_diameter, expected_steps)
        result = analyze(identity, samples, history, warmup)
        evaluate(result, read_reference(reference))
        write_result(output, result)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as exception:
        print(f"invalid Re100 result: {exception}", file=sys.stderr)
        return 2
    print(
        f"Re100 analysis: status={result['analysis_status']} "
        f"St={result.get('St', '')} st_status={result['st_status']} "
        f"combined_status={result['combined_status']} failed={result['failed_gates']}"
    )
    return 0 if result["analysis_status"] == "ok" and result["combined_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
