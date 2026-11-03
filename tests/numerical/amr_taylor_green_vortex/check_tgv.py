#!/usr/bin/env python3
"""Native TGV science/checker; short artifact readback is explicitly engineering."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import ArtifactError, identity, require, composite_masks, conservative_average
from amr_validation import common_identity, close, read_frames, integrals, write_csv
from tgv_metrics import exact, statistics, errors, passes, interface_neighborhood
from native_planes import dump_planes


def export_planes(run, output, reference=None):
    for step, plot in run.plots.items():
        masks = composite_masks(plot)
        near = interface_neighborhood(plot, masks)
        for lev, (level, mask) in enumerate(zip(plot.levels, masks)):
            analytic = exact(level, plot.time, run.speed)
            fields = dict(ux=level.values[1], analytic_ux=analytic[1],
                          error_analytic=np.linalg.norm(level.values[1:]-analytic[1:], axis=0),
                          interface_band=near[lev].astype(float))
            if reference is not None:
                mapped = conservative_average(reference.plots[2*step].levels[0].values, mask.shape)
                fields["error_reference"] = np.linalg.norm(level.values[1:]-mapped[1:], axis=0)
            dump_planes(output/"planes", f"s{step}-L{lev}", fields, mask, level.spacing, run.dimensions,
                        origin=np.array(plot.low), level=lev, step=level.step, coarsest_step=step,
                        time=plot.time, phase="Sync", source_id=run.meta["source_id"],
                        descriptor=run.meta["descriptor"], backend=run.meta["backend"])
    (output/"identity.txt").write_text((run.directory/"identity.txt").read_text())


class TgvRun:
    def __init__(self, directory, engineering=False):
        self.directory = Path(directory)
        self.meta = identity(directory)
        meta = self.meta
        self.dimensions = common_identity(meta, "amr-tgv-v1")
        self.coarse_cells = int(meta["coarse_cells"])
        self.amr = meta["layout"] == "amr"
        levels = 2 if self.amr else 1
        require(self.coarse_cells in (20, 40, 80) and meta["layout"] in ("uniform", "amr") and
                int(meta["levels"]) == levels and meta["periodic"] == "1,1,1" and
                meta["initialization"] == "tgv-equilibrium", "wrong TGV composition")
        fine = 2*self.coarse_cells
        self.speed = 5/(6*fine)
        self.dt = (2 if self.amr else 1)*5/(6*fine*fine)
        self.full_steps = (3 if self.amr else 6)*fine*fine//25
        self.last = int(meta["requested_steps"])
        require(int(meta["full_steps"]) == self.full_steps and 0 <= self.last <= self.full_steps,
                "wrong TGV horizon")
        require(engineering or self.last == self.full_steps, "short run is incomplete scientific evidence")
        for key, value in (("reynolds", 10), ("viscosity", .1), ("u0_lb", self.speed), ("u0_phys", 1),
                           ("rho_reference", 1), ("velocity_reference", self.speed), ("length_z", .2)):
            close(meta[key], value, f"wrong {key}")
        for lev in range(levels):
            close(meta[f"dt_{lev}"], self.dt/2**lev, "wrong TGV dt")
            close(meta[f"tau_{lev}"], .5+.25/2**(levels-1-lev), "wrong TGV tau")
        steps = sorted({0, self.last} | {s for s in (self.full_steps//4, self.full_steps//2) if s <= self.last})
        base = self.coarse_cells if self.amr else fine
        shapes = [((base*2**lev//5 if self.dimensions == 3 else 1), base*2**lev, base*2**lev)
                  for lev in range(levels)]
        self.plots = read_frames(directory, meta, steps, shapes, (0, 0, 0), (1, 1, .2), self.dt)
        for plot in self.plots.values():
            if self.amr:
                valid = plot.levels[1].valid
                require(valid[:, fine//4:3*fine//4, fine//4:3*fine//4].all() and
                        not (valid[:, 0, :].any() or valid[:, -1, :].any() or
                             valid[:, :, 0].any() or valid[:, :, -1].any()),
                        "refinement misses center request or reaches the periodic domain seam")
        initial = self.plots[0]
        for level in initial.levels:
            require(np.max(np.abs(level.values[:, level.valid]-exact(level, 0, self.speed)[:, level.valid])) <= 1e-12,
                    "native initial field differs from TGV equilibrium macros")
        mass, momentum = integrals(initial, composite_masks(initial), self.dimensions)
        self.records = []
        for step, plot in self.plots.items():
            row = dict(step=step, **statistics(plot, self.speed, self.dimensions, mass, momentum))
            row["passed"] = int(passes(row, self.coarse_cells, self.amr))
            self.records.append(row)

    def compare(self, reference):
        require(not reference.amr and self.amr and self.coarse_cells == reference.coarse_cells,
                "reference must be the matched uniform fine grid")
        for key in ("source_id", "backend", "descriptor", "dimensions", "u0_lb", "length_z"):
            require(self.meta[key] == reference.meta[key], "reference identity differs: "+key)
        require(all(row["passed"] for row in reference.records), "uniform reference fails its qualification budget")
        for row in self.records:
            step = row["step"]
            e2, peak = errors(self.plots[step], self.speed, self.dimensions, reference.plots[2*step])
            row["reference_e2"], row["reference_interface_peak"] = e2, peak
            # Error budgets apply at the three nonzero observation times.
            if step:
                row["passed"] &= int(e2 <= (.02 if self.coarse_cells == 80 else .05) and peak <= .05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status, error = 2, ""
    try:
        run = TgvRun(args.run, args.engineering)
        export_planes(run, args.output)
        write_csv(args.output/"metrics.csv", list(run.records[0]), run.records)
        if run.amr and not args.engineering:
            require(args.reference is not None, "matching uniform reference is required")
            reference = TgvRun(args.reference)
            run.compare(reference)
            export_planes(run, args.output, reference)
        write_csv(args.output/"metrics.csv", list(run.records[0]), run.records)
        status = 0 if args.engineering or all(row["passed"] for row in run.records) else 1
    except (ArtifactError, OSError, ValueError, KeyError, IndexError) as exception:
        error = str(exception)
    (args.output/"result.txt").write_text(f"exit={status}\nscope={'engineering' if args.engineering else 'scientific'}\nerror={error}\n")
    print(f"tgv_checker exit={status} {error}")
    return status


if __name__ == "__main__":
    sys.exit(main())
