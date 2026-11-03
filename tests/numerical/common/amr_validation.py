"""Small consumers shared by the Stage-6-05 native-field checks."""
import csv
import math
from pathlib import Path
import re
import numpy as np
from amr_plotfile import require, rows, read_plot, composite_masks, totals


def common_identity(meta, schema):
    dimensions = int(meta["dimensions"])
    require(meta["schema"] == schema and meta["backend"] in ("cpu", "cuda") and
            (meta["descriptor"], dimensions) in (("D2Q9", 2), ("D3Q27", 3)) and
            meta["collision"] == "BGK" and int(meta["real_digits"]) == 53 and
            re.fullmatch(r"[0-9a-f]{40}", meta["source_id"]) and int(meta["ranks"]) > 0,
            "invalid scientific/source/backend identity")
    return dimensions


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-14),
            message)


def time_matches(actual, expected, step, dt):
    unit = np.finfo(float).eps / 2
    gamma = step * unit / (1-step*unit)
    return math.isfinite(actual) and abs(actual-expected) <= min(dt/4, 4*gamma*abs(expected)+16*unit*max(1, abs(expected)))


def read_frames(directory, meta, steps, shapes, low, high, dt):
    """Validate the writer's synchronized index against the actual native files."""
    directory = Path(directory)
    require((directory/"exit.txt").read_text().splitlines() == ["solver_exit=0", "tee_exit=0"],
            "solver/tee did not complete")
    boxes = [[] for _ in shapes]
    for row in rows(directory/"layout.csv"):
        lev, index, rank = (int(row[key]) for key in ("level", "box", "rank"))
        require(0 <= lev < len(shapes) and index == len(boxes[lev]) and
                0 <= rank < int(meta["ranks"]), "invalid layout level/box/rank")
        boxes[lev].append((tuple(int(row[k]) for k in ("ilo", "jlo", "klo")),
                           tuple(int(row[k]) for k in ("ihi", "jhi", "khi"))))
    index = {}
    for row in rows(directory/"samples.csv"):
        step, lev = int(row["coarsest_step"]), int(row["level"])
        require((step, lev) not in index and step in steps and 0 <= lev < len(shapes) and
                row["kind"] == "sync" and row["phase"] == "Sync" and row["path"] == f"sync-{step}" and
                int(row["step"]) == step*2**lev and
                time_matches(float(row["time"]), step*dt, step*2**lev, dt/2**lev),
                "invalid or duplicate synchronized sample")
        index[step, lev] = row
    require(set(index) == {(step, lev) for step in steps for lev in range(len(shapes))},
            "missing synchronized frame/level")
    result = {}
    for step in steps:
        plot = read_plot(directory/f"sync-{step}")
        require(len(plot.levels) == len(shapes) and
                np.allclose(plot.low, low, rtol=0, atol=1e-14) and
                np.allclose(plot.high, high, rtol=0, atol=1e-14) and
                time_matches(plot.time, step*dt, step*2**(len(shapes)-1), dt/2**(len(shapes)-1)),
                "wrong physical domain/time/level count")
        for lev, level in enumerate(plot.levels):
            require(level.valid.shape == shapes[lev] and level.step == step*2**lev and
                    sorted(level.boxes) == sorted(boxes[lev]) and
                    np.array_equal(level.valid, np.broadcast_to(level.valid[0], level.valid.shape)),
                    "wrong native shape/layout/clock or non-extruded coverage")
            close(float(index[step, lev]["time"]), plot.time, "native/index clock mismatch")
        require(plot.levels[0].valid.all(), "incomplete L0 domain")
        totals(plot, composite_masks(plot))
        result[step] = plot
    return result


def integrals(plot, masks, dimensions):
    mass, momentum = totals(plot, masks)
    scale = 1/(plot.high[2]-plot.low[2]) if dimensions == 2 else 1
    return mass*scale, momentum*scale


def quasi_metrics(plot, masks):
    uz = rho_z = velocity_z = 0.
    for level, mask in zip(plot.levels, masks):
        if not mask.any():
            continue
        require(np.array_equal(mask, np.broadcast_to(mask[0], mask.shape)), "partial composite z column")
        columns = level.values[:, :, mask[0]]
        mean = columns.mean(axis=1, keepdims=True)
        difference = columns-mean
        uz = max(uz, float(np.max(np.abs(columns[3]))))
        rho_z = max(rho_z, float(np.max(np.abs(difference[0]/mean[0]))))
        velocity_z = max(velocity_z, float(np.max(np.linalg.norm(difference[1:], axis=0))))
    return dict(uz_max=uz, rho_z_max=rho_z, velocity_z_max=velocity_z)


def write_csv(path, fields, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(records)
