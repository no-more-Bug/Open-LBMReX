"""Re100 native fields and unsmoothed cylinder force records."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import identity, require
from amr_validation import common_identity, close, read_frames, quasi_metrics, write_csv
from check_re20 import fluid_masks, flux, coordinates, force_steps
from re100_cycles import Sample, cycle_records, select_window, summarize
from native_planes import dump_planes


class Re100Run:
    def __init__(self, directory, engineering=False):
        self.directory = Path(directory)
        self.id = identity(directory)
        meta = self.id
        self.dimensions = common_identity(meta, "amr-re100-v1")
        self.layout = meta["layout"]
        require(meta["scheme"] in ("YMS", "Bouzidi") and self.layout in ("coarse", "fine", "amr") and
                meta["periodic"] == "0,0,1" and meta["initialization"] == "uniform-equilibrium-then-Guo",
                "wrong Re100 composition")
        self.levels = 3 if self.layout == "amr" else 1
        self.owner_level = self.levels-1
        nd = {"coarse": 40, "fine": 80, "amr": 20}[self.layout]
        self.h = .1/nd
        self.dt = .035*self.h
        self.full_steps = round(35/self.dt)
        self.last = int(meta["requested_steps"])
        require(int(meta["levels"]) == self.levels and int(meta["owner_level"]) == self.owner_level and
                int(meta["full_steps"]) == self.full_steps and 0 < self.last <= self.full_steps,
                "wrong Re100 owner/horizon")
        require(engineering or self.last == self.full_steps, "short run is incomplete scientific evidence")
        for key, expected in (("reynolds", 100), ("viscosity", .001), ("u0_lb", .035),
                              ("u0_phys", 1), ("rho_reference", 1), ("velocity_reference", .035), ("length_z", .01)):
            close(meta[key], expected, "wrong "+key)
        shapes = []
        for lev in range(self.levels):
            ratio = 2**lev
            shapes.append((nd//10*ratio if self.dimensions == 3 else 1, (41*nd//10+1)*ratio, (22*nd+1)*ratio))
            close(meta[f"dt_{lev}"], self.dt/ratio, "wrong Re100 dt")
            close(meta[f"tau_{lev}"], .5+3*.001*(self.dt/ratio)/(self.h/ratio)**2, "wrong Re100 tau")
        finest_h = self.h/2**self.owner_level
        self.denominator = .5*.035**2*.1/finest_h*(.01/finest_h if self.dimensions == 3 else 1)
        for key, expected in (("diameter_lb", .1/finest_h), ("span_lb", .01/finest_h),
                              ("force_denominator", self.denominator)):
            close(meta[key], expected, "wrong force normalization")
        steps = sorted({0, self.last} | {self.full_steps*n//5 for n in range(1, 6) if self.full_steps*n//5 <= self.last})
        low, high = (-self.h/2, -self.h/2, 0), (2.2+self.h/2, .41+self.h/2, .01)
        self.plots = read_frames(directory, meta, steps, shapes, low, high, self.dt)
        self.fields = []
        for step, plot in self.plots.items():
            for lev, level in enumerate(plot.levels[1:], 1):
                x, y = coordinates(plot, lev)
                bounds = ((.0475, 1.2025, .0475, .3525), (.1, 1., .1, .3))[lev-1]
                request = ((y[:, None] > bounds[2]) & (y[:, None] < bounds[3]) &
                           (x[None, :] > bounds[0]) & (x[None, :] < bounds[1]))
                require(level.valid[:, request].all() and not
                        (level.valid[:, 0, :].any() or level.valid[:, -1, :].any() or
                         level.valid[:, :, 0].any() or level.valid[:, :, -1].any()),
                        "refinement misses request or reaches Guo boundary")
            masks = fluid_masks(plot)
            mass = sum(float(np.sum(level.values[0, mask]))*np.prod(level.spacing)
                       for level, mask in zip(plot.levels, masks))
            if self.dimensions == 2:
                mass /= .01
            self.fields.append(dict(step=step, time=plot.time, fluid_center_mass=mass,
                                    **flux(plot), **quasi_metrics(plot, masks)))
        coefficients, durations, ends = force_steps(self, self.last*2**self.owner_level)
        self.force_times = ends["boundary"]-.5*durations["boundary"]
        self.coefficients = coefficients["boundary"]
        self.quasi_pass = (all(row[key] <= 1e-10 for row in self.fields
                               for key in ("uz_max", "rho_z_max", "velocity_z_max")) and
                           np.max(np.abs(self.coefficients[:, 2])) <= 1e-10)
        self.samples = [Sample(index+1, time, 10*time, cd, cl)
                        for index, (time, (cd, cl, _)) in enumerate(zip(self.force_times, self.coefficients))]
        self.cycles = cycle_records(self.samples) if not engineering else []
        self.selected = select_window(self.cycles)
        self.metrics = summarize(self.samples, self.cycles, self.selected[0]) if self.selected else None

    def export(self, output):
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output/"fields.csv", list(self.fields[0]), self.fields)
        write_csv(output/"force-history.csv", ("time", "t_star", "Cd", "Cl", "Cz"),
                  (dict(time=t, t_star=10*t, Cd=c[0], Cl=c[1], Cz=c[2])
                   for t, c in zip(self.force_times, self.coefficients)))
        if self.cycles:
            write_csv(output/"cycles.csv", list(self.cycles[0]), self.cycles)
        if self.metrics:
            write_csv(output/"statistics.csv", list(self.metrics), [self.metrics])
            write_csv(output/"window-ranges.csv", ("metric", "normalized_range"),
                      (dict(metric=key, normalized_range=value) for key, value in self.selected[1].items()))
        for step, plot in self.plots.items():
            for lev, (level, mask) in enumerate(zip(plot.levels, fluid_masks(plot))):
                fields = dict(speed=np.linalg.norm(level.values[1:], axis=0),
                              pressure=(level.values[0]-1)/(3*.035**2))
                dump_planes(output/"planes", f"s{step}-L{lev}", fields, mask, level.spacing, self.dimensions,
                            origin=np.array(plot.low), level=lev, step=level.step, coarsest_step=step,
                            time=plot.time, phase="Sync", source_id=self.id["source_id"],
                            descriptor=self.id["descriptor"], backend=self.id["backend"])
        (output/"identity.txt").write_text((self.directory/"identity.txt").read_text())
