#!/usr/bin/env python3
"""Compare completed Re100 exports without smoothing, resampling, or scientific gates."""
import argparse
import csv
from dataclasses import dataclass
from itertools import combinations, zip_longest
import math
from pathlib import Path
import sys
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from amr_plotfile import ArtifactError, identity, require, rows
from amr_validation import close, common_identity, time_matches, write_csv

TIMES = (0, 7, 14, 21, 28, 35)
PHYSICS = ("reynolds", "viscosity", "u0_lb", "u0_phys", "rho_reference",
           "velocity_reference", "length_z", "diameter_lb", "force_denominator")
SOURCE_KEYS = ("schema", "source_id", "descriptor", "backend", "ranks", "scheme", "layout",
               "variant", "full_steps", "reynolds", "viscosity", "u0_lb", "u0_phys",
               "rho_reference", "velocity_reference", "length_z", "diameter_lb",
               "force_denominator", "span_lb", "grid_efficiency",
               "regrid_interval", "theta_refine_0", "theta_keep_0", "theta_refine_1", "theta_keep_1")
FIELD_KEYS = ("phi_in", "phi_out", "flux_imbalance", "uz_max", "rho_z_max", "velocity_z_max")


@dataclass
class Observation:
    label: str
    group: str
    directory: Path
    meta: dict
    checker_exit: int
    checker_scope: str
    family: str
    cylinders: tuple
    force_dt: float
    sample_count: int

    @property
    def analysis(self):
        return self.directory / "analysis"


class ForceSample(NamedTuple):
    cylinder: tuple
    time: float
    Cd: float
    Cl: float


def load_observation(label, group, directory):
    directory = Path(directory)
    meta = identity(directory / "analysis")
    require(meta == identity(directory), "run/analysis identity mismatch")
    schema = meta["schema"]
    require(schema in ("dynamic-cylinder-v1", "amr-re100-v1"), "unsupported cylinder schema")
    dimensions = common_identity(meta, schema)
    exits = identity(directory, "exit.txt")
    require(exits.get("solver_exit") == "0" and exits.get("tee_exit") == "0", "incomplete solver/output")
    result = dict(line.split("=", 1) for line in (directory / "analysis/result.txt").read_text().splitlines())
    status, scope = int(result["exit"]), result["scope"]
    if schema == "dynamic-cylinder-v1":
        require(status == 0 and scope == "engineering-and-data-only", "dynamic data check did not pass")
        require(meta["variant"] in ("single", "tandem-a", "tandem-b"), "unknown cylinder variant")
        family = "single" if meta["variant"] == "single" else "tandem"
        cylinders = tuple(sorted((int(row["boundary_id"]), int(row["object_id"]))
                                 for row in rows(directory / "boundaries.csv")))
        owner, full = int(meta["max_level"]), 200000
        require(meta["force_units"] == "allowed-finest-lattice", "wrong dynamic force units")
    else:
        require(status == 0 or (scope == "scientific" and status in (1, 2)),
                "legacy data check did not finish")
        family = "single"
        cylinders = ((int(meta["boundary_id"]), int(meta["object_id"])),)
        owner = int(meta["owner_level"])
        full = {"coarse": 400000, "fine": 800000, "amr": 200000}[meta["layout"]]
    require(int(meta["full_steps"]) == int(meta["requested_steps"]) == full, "incomplete full horizon")
    require(cylinders and len(set(cylinders)) == len(cylinders), "invalid cylinder inventory")
    close(float(meta["dt_0"]) * full, 35, "wrong physical endpoint")
    span = float(meta["span_lb"]) if dimensions == 3 else 1
    denominator = .5 * float(meta["rho_reference"]) * float(meta["velocity_reference"])**2
    denominator *= float(meta["diameter_lb"]) * span
    require(denominator > 0, "invalid force scale")
    close(meta["force_denominator"], denominator, "wrong force denominator")
    return Observation(label, group, directory, meta, status, scope, family, cylinders,
                       float(meta[f"dt_{owner}"]), full * 2**owner)


def compatible(group):
    first = group[0]
    for run in group[1:]:
        require(run.family == first.family and run.cylinders == first.cylinders,
                "different cylinder geometry/inventory")
        for key in ("descriptor", "dimensions", "backend", "scheme", "collision",
                    "periodic", "initialization", "real_digits"):
            require(run.meta[key] == first.meta[key], "incompatible " + key)
        for key in PHYSICS:
            close(run.meta[key], float(first.meta[key]), "incompatible " + key)
        close(run.force_dt, first.force_dt, "different native force cadence")
        require(run.sample_count == first.sample_count, "different native force sample count")
        if int(first.meta["dimensions"]) == 3:
            close(run.meta["span_lb"], float(first.meta["span_lb"]), "incompatible D3 span")


def iter_force_records(run):
    """Lightweight native-sequence reader, also usable by pvpython; no reduction or writes."""
    dynamic = run.meta["schema"] == "dynamic-cylinder-v1"
    with (run.analysis / "force-history.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["boundary_id"]), int(row["object_id"])) if dynamic else run.cylinders[0]
            values = tuple(float(row[name]) for name in ("time", "Cd", "Cl"))
            require(key in run.cylinders and all(math.isfinite(v) for v in values), "invalid force record")
            yield ForceSample(key, *values)


def force_batches(run):
    batch = {}
    for record in iter_force_records(run):
        require(record.cylinder not in batch, "duplicate/missing cylinder in force step")
        batch[record.cylinder] = record
        if len(batch) == len(run.cylinders):
            yield batch
            batch = {}
    require(not batch, "incomplete force step")


@dataclass
class Moments:
    count: int = 0
    mean: float = 0
    mean_square: float = 0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value):
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.mean_square += (value * value - self.mean_square) / self.count
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)


def compare_forces(group):
    """One streaming pass per run; compare matching native midpoints, including startup."""
    compatible(group)
    pairs = list(combinations(range(len(group)), 2))
    keys = group[0].cylinders
    stats = {(i, key, component): Moments() for i in range(len(group))
             for key in keys for component in ("Cd", "Cl")}
    differences = {(a, b, key, component): Moments() for a, b in pairs
                   for key in keys for component in ("Cd", "Cl")}
    count = 0
    for count, batches in enumerate(zip_longest(*(force_batches(run) for run in group)), 1):
        require(all(batch is not None for batch in batches), "different force sequence lengths")
        for key in keys:
            for i, run in enumerate(group):
                record = batches[i][key]
                require(time_matches(record.time, (count - .5) * run.force_dt, count, run.force_dt),
                        "force midpoint is shifted or missing")
                for component in ("Cd", "Cl"):
                    stats[i, key, component].add(getattr(record, component))
            for a, b in pairs:
                for component in ("Cd", "Cl"):
                    differences[a, b, key, component].add(
                        getattr(batches[b][key], component) - getattr(batches[a][key], component))
    require(count == group[0].sample_count, "incomplete exported force sequence")
    summaries, deltas = [], []
    for (i, key, component), value in stats.items():
        run = group[i]
        summaries.append(dict(label=run.label, group=run.group, boundary_id=key[0], object_id=key[1],
                              component=component, samples=value.count,
                              first_midpoint=run.force_dt / 2, last_midpoint=(count - .5) * run.force_dt,
                              minimum=value.minimum, maximum=value.maximum, mean=value.mean))
    for (a, b, key, component), value in differences.items():
        deltas.append(dict(group=group[a].group, label_a=group[a].label, label_b=group[b].label,
                           boundary_id=key[0], object_id=key[1], component=component, samples=value.count,
                           absolute_rms=math.sqrt(value.mean_square),
                           maximum_absolute_difference=max(abs(value.minimum), abs(value.maximum))))
    return summaries, deltas


def field_observations(run):
    selected = {}
    for row in rows(run.analysis / "fields.csv"):
        for target in TIMES:
            if math.isclose(float(row["time"]), target, rel_tol=1e-10, abs_tol=1e-12):
                require(target not in selected, "duplicate common field time")
                selected[target] = row
    require(set(selected) == set(TIMES), "missing common field time; no interpolation is allowed")
    initial = float(selected[0]["fluid_center_mass"])
    require(math.isfinite(initial) and initial > 0, "invalid initial fluid-center mass")
    output = []
    for time in TIMES:
        row = selected[time]
        mass = float(row["fluid_center_mass"])
        values = {key: float(row[key]) for key in FIELD_KEYS}
        require(math.isfinite(mass) and all(math.isfinite(v) for v in values.values()),
                "nonfinite field diagnostic")
        output.append(dict(label=run.label, group=run.group, time=time, fluid_center_mass=mass,
                           initial_fluid_center_mass=initial, relative_mass_drift=abs(mass / initial - 1),
                           **values))
    # flux_imbalance is the existing absolute normalized flux difference, not a signed residual.
    return output


def compare(manifest, output):
    entries = rows(manifest)
    require(entries and all(set(row) == {"label", "group", "run"} for row in entries),
            "manifest requires label,group,run")
    require(all(row["label"] and row["group"] for row in entries) and
            len({row["label"] for row in entries}) == len(entries), "empty/duplicate observation label")
    runs = [load_observation(row["label"], row["group"], row["run"]) for row in entries]
    output = Path(output)
    require(all(output.resolve() not in (run.directory.resolve(), run.analysis.resolve()) for run in runs),
            "comparison output must differ from original run/analysis")
    groups = {}
    for run in runs:
        groups.setdefault(run.group, []).append(run)
    force, differences, fields = [], [], []
    for group in groups.values():
        summaries, deltas = compare_forces(group)
        force.extend(summaries)
        differences.extend(deltas)
        for run in group:
            fields.extend(field_observations(run))
    sources = [dict(label=run.label, group=run.group, run=str(run.directory.resolve()),
                    checker_exit=run.checker_exit, checker_scope=run.checker_scope,
                    force_dt=run.force_dt, **{key: run.meta.get(key, "") for key in SOURCE_KEYS}) for run in runs]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "sources.csv", list(sources[0]), sources)
    write_csv(output / "force-statistics.csv", list(force[0]), force)
    write_csv(output / "force-differences.csv",
              ("group", "label_a", "label_b", "boundary_id", "object_id", "component",
               "samples", "absolute_rms", "maximum_absolute_difference"), differences)
    write_csv(output / "field-observations.csv", list(fields[0]), fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare(args.manifest, args.output)
    except (ArtifactError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"cylinder observations unavailable: {error}", file=sys.stderr)
        return 2
    (args.output / "result.txt").write_text("exit=0\nscope=observation-only\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
