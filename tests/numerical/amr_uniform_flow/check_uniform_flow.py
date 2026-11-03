#!/usr/bin/env python3
"""Uniform Guo flow: complete science by default; --engineering reads short artifacts."""
import argparse
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import ArtifactError, require, identity, rows, composite_masks
from amr_validation import common_identity, close, time_matches, read_frames, integrals, quasi_metrics, write_csv
from native_planes import dump_planes


def field_metrics(plot, dimensions):
    masks = composite_masks(plot)
    mass, _ = integrals(plot, masks, dimensions)
    error2 = measure = 0.
    for level, mask in zip(plot.levels, masks):
        difference = level.values[1:, mask]-np.array([1., 0., 0.])[:, None]
        weight = math.prod(level.spacing[:dimensions])
        error2 += float(np.sum(difference**2))*weight
        measure += int(mask.sum())*weight
    return dict(mass=mass, relative_l2=math.sqrt(error2/measure), **quasi_metrics(plot, masks))


def analyze(directory, engineering=False, output=None):
    directory = Path(directory)
    meta = identity(directory)
    dimensions = common_identity(meta, "amr-uniform-v1")
    require(meta["periodic"] == "0,1,1" and meta["boundary"] == "guo_velocity_io" and
            meta["initialization"] == "constant-equilibrium" and int(meta["levels"]) == 3,
            "wrong uniform-flow composition")
    for key, expected in (("u0_lb", .1), ("u0_phys", 1), ("nu_fine_lb", 1/12),
                          ("rho_reference", 1), ("velocity_reference", .1), ("length_z", .125)):
        close(meta[key], expected, f"wrong {key}")
    for lev in range(3):
        close(meta[f"dt_{lev}"], 1/(320*2**lev), "wrong level time scale")
        close(meta[f"tau_{lev}"], .5+.25/2**(2-lev), "wrong level tau")
    last = int(meta["requested_steps"])
    require(int(meta["full_steps"]) == 32000 and 0 <= last <= 32000, "wrong horizon")
    require(engineering or last == 32000, "short run is incomplete scientific evidence")
    steps = sorted({0, last} | ({16000} if last >= 16000 else set()))
    shapes = [((4*2**lev if dimensions == 3 else 1), 32*2**lev, 33*2**lev) for lev in range(3)]
    plots = read_frames(directory, meta, steps, shapes, (-1/64, 0, 0), (1+1/64, 1, .125), 1/320)
    for plot in plots.values():
        for lev, bounds in ((1, (16, 48, 16, 48)), (2, (48, 80, 48, 80))):
            xlo, xhi, ylo, yhi = bounds
            valid = plot.levels[lev].valid
            require(valid[:, ylo:yhi, xlo:xhi].all() and not valid[:, :, 0].any() and not valid[:, :, -1].any(),
                    "refinement misses requested region or reaches Guo boundary")
    if output is not None:
        for step, plot in plots.items():
            for lev, (level, mask) in enumerate(zip(plot.levels, composite_masks(plot))):
                difference = level.values[1:]-np.array([1., 0., 0.])[:, None, None, None]
                dump_planes(Path(output)/"planes", f"s{step}-L{lev}",
                            dict(ux=level.values[1], rho_error=level.values[0]-1,
                                 velocity_error=np.linalg.norm(difference, axis=0)),
                            mask, level.spacing, dimensions, origin=np.array(plot.low),
                            level=lev, step=level.step, coarsest_step=step, time=plot.time, phase="Sync",
                            source_id=meta["source_id"], descriptor=meta["descriptor"], backend=meta["backend"])
        (Path(output)/"identity.txt").write_text((directory/"identity.txt").read_text())
    expected_steps = sorted({0, last} | set(range(320, last+1, 320)))
    records = rows(directory/"observations.csv")
    require(len(records) == len(expected_steps), "missing scalar observations")
    initial_mass = field_metrics(plots[0], dimensions)["mass"]
    result = []
    for step, record in zip(expected_steps, records):
        row = {key: float(record[key]) for key in ("step", "time", "mass", "relative_l2", "uz_max", "rho_z_max", "velocity_z_max")}
        require(all(math.isfinite(v) for v in row.values()) and row["step"] == step and
                time_matches(row["time"], step/320, step*4, 1/1280) and row["mass"] > 0 and
                all(row[key] >= 0 for key in ("relative_l2", "uz_max", "rho_z_max", "velocity_z_max")),
                "invalid uniform scalar value/clock")
        if step in plots:
            for key, value in field_metrics(plots[step], dimensions).items():
                close(row[key], value, "native/scalar mismatch: "+key)
        row["mass_relative_change"] = abs(row["mass"]-initial_mass)/initial_mass
        row["passed"] = int(row["relative_l2"] <= 1e-8 and row["mass_relative_change"] <= 1e-8 and
                            all(row[key] <= 1e-10 for key in ("uz_max", "rho_z_max", "velocity_z_max")))
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status, error = 2, ""
    try:
        records = analyze(args.run, args.engineering, args.output)
        write_csv(args.output/"metrics.csv", list(records[0]), records)
        status = 0 if all(row["passed"] for row in records) else 1
    except (ArtifactError, OSError, ValueError, KeyError, IndexError) as exception:
        error = str(exception)
    (args.output/"result.txt").write_text(f"exit={status}\nscope={'engineering' if args.engineering else 'scientific'}\nerror={error}\n")
    print(f"uniform_checker exit={status} {error}")
    return status


if __name__ == "__main__":
    sys.exit(main())
