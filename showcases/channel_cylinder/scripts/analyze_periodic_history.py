#!/usr/bin/env python3
"""Extract and evaluate DFG Re100 periodic metrics from public force history."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


CYCLES_USED = 3
MIN_SAMPLES_PER_CYCLE = 200
MAX_CYCLE_DRIFT = 0.01
EXPECTED_REFERENCE_LABEL = "dfg_2d_2_re100"
REFERENCE_HASH = "f1ff94190e21e78f8c313018e3376da20ffeb5c261b399bb5c1f2663bddf2d2f"
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


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    population_step: int
    force_time_lb: float
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
    sample_count: int
    cd_min: float
    cd_max: float
    cl_min: float
    cl_max: float

    @property
    def period(self) -> float:
        return self.end.time - self.start.time


def read_history(path: Path) -> tuple[tuple[str, str, int], list[Sample]]:
    if not path.is_file():
        raise AnalysisError(f"missing history: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or REQUIRED_COLUMNS.difference(fields):
            raise AnalysisError("history schema is missing or has duplicate columns")
        identity: tuple[str, str, int] | None = None
        samples = []
        for line_number, row in enumerate(reader, start=2):
            try:
                row_identity = (
                    row["reference_label"], row["object_name"], int(row["object_id"]),
                )
                sample = Sample(
                    int(row["population_step"]), float(row["force_time_lb"]),
                    float(row["t_star"]), float(row["Cd"]), float(row["Cl"]),
                )
            except (TypeError, ValueError) as error:
                raise AnalysisError(f"line {line_number}: invalid history value") from error
            if not all(
                math.isfinite(value)
                for value in (sample.force_time_lb, sample.t_star, sample.cd, sample.cl)
            ):
                raise AnalysisError(f"line {line_number}: non-finite history value")
            if identity is None:
                identity = row_identity
            elif identity != row_identity:
                raise AnalysisError(f"line {line_number}: object identity changed")
            if samples and (
                sample.population_step <= samples[-1].population_step
                or sample.force_time_lb <= samples[-1].force_time_lb
                or sample.t_star <= samples[-1].t_star
            ):
                raise AnalysisError(f"line {line_number}: stamps must increase")
            samples.append(sample)
    if identity is None or not samples:
        raise AnalysisError("history has no rows")
    if identity[0] != EXPECTED_REFERENCE_LABEL:
        raise AnalysisError(
            f"expected reference_label={EXPECTED_REFERENCE_LABEL}; got {identity[0]}"
        )
    return identity, samples


def quadratic_extremum(
    samples: list[Sample], index: int, attribute: str, minimum: bool
) -> Extremum:
    center = samples[index]
    center_value = getattr(center, attribute)
    if index == 0 or index + 1 == len(samples):
        return Extremum(index, center.t_star, center_value)
    left = samples[index - 1]
    right = samples[index + 1]
    x0 = left.t_star - center.t_star
    x2 = right.t_star - center.t_star
    y0 = getattr(left, attribute) - center_value
    y2 = getattr(right, attribute) - center_value
    curvature = (y0 / x0 - y2 / x2) / (x0 - x2)
    slope = y0 / x0 - curvature * x0
    if curvature == 0.0 or (minimum and curvature < 0.0) or (
        not minimum and curvature > 0.0
    ):
        return Extremum(index, center.t_star, center_value)
    offset = -slope / (2.0 * curvature)
    if offset < x0 or offset > x2:
        return Extremum(index, center.t_star, center_value)
    return Extremum(
        index,
        center.t_star + offset,
        center_value + slope * offset + curvature * offset * offset,
    )


def lift_minima(samples: list[Sample]) -> list[Extremum]:
    return [
        quadratic_extremum(samples, index, "cl", True)
        for index in range(1, len(samples) - 1)
        if samples[index].cl < samples[index - 1].cl
        and samples[index].cl < samples[index + 1].cl
    ]


def range_extremum(
    samples: list[Sample], begin: int, end: int, attribute: str, minimum: bool
) -> float:
    chooser = min if minimum else max
    index = chooser(
        range(begin, end + 1),
        key=lambda candidate: getattr(samples[candidate], attribute),
    )
    return quadratic_extremum(samples, index, attribute, minimum).value


def make_cycle(samples: list[Sample], start: Extremum, end: Extremum) -> Cycle:
    return Cycle(
        start, end, end.index - start.index + 1,
        range_extremum(samples, start.index, end.index, "cd", True),
        range_extremum(samples, start.index, end.index, "cd", False),
        range_extremum(samples, start.index, end.index, "cl", True),
        range_extremum(samples, start.index, end.index, "cl", False),
    )


def max_relative_drift(values: list[float]) -> float:
    return max(
        abs(current - previous) / max(abs(current), abs(previous), 1.0e-300)
        for previous, current in zip(values, values[1:])
    )


def empty_result(
    identity: tuple[str, str, int], history: Path, warmup: float, samples: int
) -> dict[str, object]:
    result: dict[str, object] = {column: "" for column in OUTPUT_COLUMNS}
    result.update(
        reference_label=identity[0], object_name=identity[1], object_id=identity[2],
        history_path=str(history), warmup_t_star=warmup, total_samples=samples,
        st_status="not_evaluated", combined_status="not_evaluated",
    )
    return result


def analyze_cycles(
    identity: tuple[str, str, int], samples: list[Sample], history: Path, warmup: float
) -> dict[str, object]:
    result = empty_result(identity, history, warmup, len(samples))
    minima = lift_minima(samples)
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
    minimum_samples = min(cycle.sample_count for cycle in used)
    result.update(cycles_used=CYCLES_USED, min_samples_in_used_cycles=minimum_samples)
    if minimum_samples < MIN_SAMPLES_PER_CYCLE:
        result.update(
            analysis_status="insufficient_samples",
            message=(
                f"each selected cycle needs {MIN_SAMPLES_PER_CYCLE} samples; "
                f"minimum is {minimum_samples}"
            ),
        )
        return result
    selected = used[-1]
    result.update(
        analysis_status="ok",
        message="periodic metrics extracted from the last three completed cycles",
        cycle_start_population_step=samples[selected.start.index].population_step,
        cycle_end_population_step=samples[selected.end.index].population_step,
        cycle_start_t_star=selected.start.time,
        cycle_end_t_star=selected.end.time,
        samples_in_cycle=selected.sample_count,
        Cd_min=selected.cd_min,
        Cd_max=selected.cd_max,
        Cd_midrange=0.5 * (selected.cd_min + selected.cd_max),
        Cd_peak_to_peak=selected.cd_max - selected.cd_min,
        Cl_min=selected.cl_min,
        Cl_max=selected.cl_max,
        Cl_midrange=0.5 * (selected.cl_min + selected.cl_max),
        Cl_peak_to_peak=selected.cl_max - selected.cl_min,
        period_t_star=selected.period,
        St=1.0 / selected.period,
        max_period_relative_drift=max_relative_drift([cycle.period for cycle in used]),
        max_Cd_extrema_relative_drift=max(
            max_relative_drift([cycle.cd_min for cycle in used]),
            max_relative_drift([cycle.cd_max for cycle in used]),
        ),
        max_Cl_extrema_relative_drift=max(
            max_relative_drift([cycle.cl_min for cycle in used]),
            max_relative_drift([cycle.cl_max for cycle in used]),
        ),
    )
    return result


def read_reference(path: Path) -> dict[str, tuple[float, float, str]]:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_HASH:
        raise AnalysisError("DFG Re100 reference checksum changed")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    reference = {}
    for row in rows:
        reference[row["metric"]] = (
            float(row["value"]), float(row["relative_tolerance"]), row["status_group"]
        )
    return reference


def evaluate_reference(
    result: dict[str, object], reference: dict[str, tuple[float, float, str]]
) -> None:
    if result["analysis_status"] != "ok":
        return
    failures = []
    errors = {}
    for metric, (expected, tolerance, _) in reference.items():
        actual = float(result[metric])
        error = abs(actual - expected) / abs(expected)
        errors[metric] = error
        if error > tolerance:
            failures.append(f"{metric}:{error:.12g}>{tolerance:.12g}")
    period_stable = float(result["max_period_relative_drift"]) <= MAX_CYCLE_DRIFT
    cd_stable = float(result["max_Cd_extrema_relative_drift"]) <= MAX_CYCLE_DRIFT
    cl_stable = float(result["max_Cl_extrema_relative_drift"]) <= MAX_CYCLE_DRIFT
    if not period_stable:
        failures.append("period_drift")
    if not cd_stable:
        failures.append("Cd_extrema_drift")
    if not cl_stable:
        failures.append("Cl_extrema_drift")
    st_pass = period_stable and errors["St"] <= reference["St"][1]
    combined_pass = st_pass and cd_stable and cl_stable and not any(
        errors[metric] > tolerance
        for metric, (_, tolerance, _) in reference.items()
    )
    result.update(
        St_relative_error=errors["St"],
        st_status="pass" if st_pass else "fail",
        combined_status="pass" if combined_pass else "fail",
        failed_gates=";".join(failures),
    )


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerow(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--warmup-t-star", type=float, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            Path(__file__).parents[1]
            / "docs"
            / "reference-data"
            / "dfg_re100_reference.csv"
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.warmup_t_star) or args.warmup_t_star < 0.0:
        parser.error("--warmup-t-star must be finite and nonnegative")
    output = args.output or args.history.with_name("channel_cylinder_periodic_metrics.csv")
    if output.resolve() == args.history.resolve():
        parser.error("--output must not overwrite history")
    try:
        identity, samples = read_history(args.history)
        result = analyze_cycles(
            identity, samples, args.history, args.warmup_t_star
        )
        evaluate_reference(result, read_reference(args.reference))
        write_result(output, result)
    except (AnalysisError, OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 2
    print(
        f"analysis_status={result['analysis_status']} st_status={result['st_status']} "
        f"combined_status={result['combined_status']} output={output}"
    )
    if result["analysis_status"] != "ok":
        return 1
    return 0 if result["combined_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
