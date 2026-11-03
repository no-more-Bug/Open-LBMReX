#!/usr/bin/env python3
"""Dynamic DFG artifacts: 0 valid, 1 engineering relation failure, 2 invalid/incomplete."""
import argparse
import csv
import math
import shutil
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"common"))
from amr_plotfile import ArtifactError, require, identity, rows, read_plot, composite_masks
from amr_validation import common_identity, close, write_csv
from native_planes import dump_planes
from check_re20 import FORCES, coordinates, flux, time_matches
from check_tandem import validate_layout

DT = .000175
FULL_STEPS = 200000
LOW = (-.0025, -.0025, 0.)
HIGH = (2.2025, .4125, .01)
EPS = np.finfo(float).eps


def sample_steps(last):
    return sorted({0, last} | set(range(0, last+1, 4000)) | set(range(160000, last+1, 400)))


def csv_records(path, fields):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(fields), f"wrong columns: {path.name}")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),
                    f"incomplete row: {path.name}")
            yield row


def fluid_masks(plot, boundaries):
    masks = composite_masks(plot)
    band = 64*EPS*max(abs(v) for v in (*plot.low, *plot.high))
    for lev, mask in enumerate(masks):
        x, y = coordinates(plot, lev)
        for item in boundaries.values():
            mask &= (np.hypot(x[None, :]-item["x"], y[:, None]-item["y"]) >= item["radius"]-band)[None]
    return masks


def quasi_metrics(plot, masks):
    result = dict(uz_max=0., rho_z_max=0., velocity_z_max=0.)
    for level, mask in zip(plot.levels, masks):
        if not mask.any():
            continue
        count = mask.sum(axis=0)
        mean = np.where(mask[None], level.values, 0).sum(axis=1)/np.maximum(count, 1)[None]
        delta = np.where(mask[None], level.values-mean[:, None], 0)
        result["uz_max"] = max(result["uz_max"], float(np.max(np.abs(level.values[3][mask]))))
        rho_delta = np.abs(delta[0])/np.maximum(np.abs(mean[0][None]), np.finfo(float).tiny)
        result["rho_z_max"] = max(result["rho_z_max"], float(rho_delta.max()))
        result["velocity_z_max"] = max(result["velocity_z_max"], float(np.linalg.norm(delta[1:], axis=0).max()))
    return result


def connected(mask, first, second):
    """Face connectivity on actual coverage, including the periodic z direction."""
    seen = np.zeros(mask.shape, bool)
    stack = [first]
    if not mask[first] or not mask[second]:
        return False
    seen[first] = True
    nz,ny,nx = mask.shape
    while stack:
        z,y,x = stack.pop()
        if (z,y,x) == second:
            return True
        for zz,yy,xx in (((z-1)%nz,y,x),((z+1)%nz,y,x),
                         (z,y-1,x),(z,y+1,x),(z,y,x-1),(z,y,x+1)):
            if 0 <= yy < ny and 0 <= xx < nx and mask[zz,yy,xx] and not seen[zz,yy,xx]:
                seen[zz,yy,xx] = True
                stack.append((zz,yy,xx))
    return False


class DynamicCylinderRun:
    def __init__(self, directory, short=False):
        self.directory = Path(directory)
        self.id = identity(directory)
        meta = self.id
        self.dimensions = common_identity(meta, "dynamic-cylinder-v1")
        self.variant = meta["variant"]
        require(self.variant in ("single", "tandem-a", "tandem-b") and meta["layout"] == "dynamic" and
                meta["scheme"] in ("YMS", "Bouzidi") and meta["periodic"] == "0,0,1" and
                meta["initialization"] == "uniform-equilibrium-then-Guo" and
                meta["force_units"] == "allowed-finest-lattice", "wrong dynamic cylinder composition")
        require(int(meta["max_level"]) == 2 and int(meta["base_z"]) == (2 if self.dimensions == 3 else 1) and
                int(meta["max_grid_size"]) >= 2 and int(meta["max_grid_size"])%2 == 0 and
                int(meta["regrid_interval"]) >= 0, "invalid dynamic cylinder grid controls")
        self.last = int(meta["requested_steps"])
        require(int(meta["full_steps"]) == FULL_STEPS and 0 <= self.last <= FULL_STEPS and
                (short or self.last == FULL_STEPS), "incomplete dynamic cylinder horizon")
        self.steps = sample_steps(self.last)
        require(int(meta["frames"]) == len(self.steps) and int(meta["overview_stride"]) == 4000 and
                int(meta["detail_begin"]) == 160000 and int(meta["detail_stride"]) == 400,
                "wrong frame schedule identity")
        for key, expected in (("reynolds",100), ("viscosity",.001), ("u0_lb",.035), ("u0_phys",1),
                              ("rho_reference",1), ("velocity_reference",.035), ("length_z",.01),
                              ("gap_fill_radius",4), ("coarse_nesting_margin",4), ("diameter_lb",80), ("span_lb",8 if self.dimensions == 3 else 1)):
            close(meta[key], expected, "wrong "+key)
        self.denominator = .5*.035**2*80*(8 if self.dimensions == 3 else 1)
        close(meta["force_denominator"], self.denominator, "wrong force normalization")
        efficiency = float(meta["grid_efficiency"])
        require(math.isfinite(efficiency) and 0 < efficiency <= 1, "invalid grid efficiency")
        for lev in range(3):
            close(meta[f"dt_{lev}"], DT/2**lev, "wrong fixed allowed-level dt")
            close(meta[f"tau_{lev}"], .5+3*.001*(DT/2**lev)/(.005/2**lev)**2, "wrong relaxation")
            require(int(meta[f"tag_buffer_{lev}"]) >= 0, "invalid buffer")
            if lev < 2:
                refine, keep = (float(meta[f"theta_{key}_{lev}"]) for key in ("refine","keep"))
                require(math.isfinite(refine) and 0 <= keep < refine, "invalid thresholds")
        exits = identity(directory, "exit.txt")
        require(exits == {"solver_exit":"0", "tee_exit":"0"}, "solver/output did not complete")
        self.failures = []
        self.boundaries = {}
        records = rows(self.directory/"boundaries.csv")
        names = ("cylinder_a",) if self.variant == "single" else ("cylinder_a","cylinder_b")
        require(len(records) == len(names) and {r["name"] for r in records} == set(names), "wrong cylinder inventory")
        for row in records:
            bid, oid, level = (int(row[k]) for k in ("boundary_id","object_id","level"))
            require(bid >= 0 and oid >= 0 and bid not in self.boundaries and 0 <= level <= 2,
                    "invalid boundary identity")
            expected_x = .2 if row["name"] == "cylinder_a" else .4
            for key, expected in (("center_x",expected_x), ("center_y",.2), ("center_z",.005), ("radius",.05)):
                close(row[key],expected,"wrong cylinder "+key)
            self.boundaries[bid] = dict(name=row["name"], object=oid, level=level, x=expected_x, y=.2, radius=.05)
            self.check(level == 2, row["name"]+" is not owned by L2")
        self.check(len({b["object"] for b in self.boundaries.values()}) == len(names), "cylinders share an Object")
        self.samples = {}
        for row in rows(self.directory/"samples.csv"):
            step, lev = int(row["coarsest_step"]), int(row["level"])
            require(step in self.steps and 0 <= lev <= 2 and (step,lev) not in self.samples and
                    row["kind"] == "sync" and row["phase"] == "Sync" and row["path"] == f"sync-{step}" and
                    int(row["step"]) == step*2**lev and
                    time_matches(float(row["time"]),step*DT,step*2**lev,DT/2**lev),
                    "missing/duplicate/invalid synchronized sample")
            self.samples[step,lev] = row
        for step in self.steps:
            levels = sorted(lev for s,lev in self.samples if s == step)
            require(levels and levels == list(range(len(levels))), "missing frame or discontinuous level index")

    def check(self, condition, message):
        if not condition and len(self.failures) < 16:
            self.failures.append(message)

    def plot(self, step):
        plot = read_plot(self.directory/f"sync-{step}")
        levels = sum(s == step for s,_ in self.samples)
        require(len(plot.levels) == levels and np.allclose(plot.low,LOW,rtol=0,atol=1e-14) and
                np.allclose(plot.high,HIGH,rtol=0,atol=1e-14) and
                time_matches(plot.time,step*DT,step*4,DT/4), "wrong native geometry/clock")
        validate_layout(rows(self.directory/f"layout-{step}.csv"),plot,int(self.id["ranks"]))
        for lev, level in enumerate(plot.levels):
            factor = 2**lev
            shape = (2*factor if self.dimensions == 3 else 1,83*factor,441*factor)
            require(level.valid.shape == shape and level.step == step*factor and
                    time_matches(float(self.samples[step,lev]["time"]),plot.time,step*4,DT/4),
                    "wrong native extent/step")
            if lev:
                guard = 4*factor
                self.check(not (level.valid[:,:,:guard].any() or level.valid[:,:,-guard:].any() or
                                level.valid[:,:guard,:].any() or level.valid[:,-guard:,:].any()),
                           f"step {step}: L{lev} enters the domain protection band")
        require(plot.levels[0].valid.all(), "incomplete L0 domain")
        self.check(levels == 3, f"step {step}: missing required L2")
        if levels == 3:
            fine = plot.levels[2].valid
            rectangles = [(106,378 if self.variant == "tandem-a" else 218)]
            if self.variant == "tandem-b":
                rectangles.append((266,378))
            for lo,hi in rectangles:
                self.check(fine[:,106:218,lo:hi].all(), f"step {step}: L2 misses a fixed rectangle")
            if step == 0 and self.variant == "tandem-b":
                middle = fine.shape[0]//2
                self.check(not connected(fine,(middle,162,162),(middle,162,322)),
                           "initial tandem-b L2 regions are connected")
        return plot

    def forces(self, output):
        counters = {}
        pending = {}
        object_levels = {}
        for item in self.boundaries.values():
            object_levels.setdefault(item["object"],set()).add(item["level"])
        require(all(len(levels) == 1 for levels in object_levels.values()),
                "unexpected cross-level Object in dynamic cylinder artifacts")
        with (output/"force-history.csv").open("w",newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("boundary_id","object_id","level","population_step","time","Cd","Cl","Cz"))
            for row in csv_records(self.directory/"forces.csv",FORCES):
                kind, entity = row["kind"], int(row["id"])
                require(kind in ("boundary","object"), "unknown force record kind")
                level, step, coarse = (int(row[k]) for k in ("level","population_step","coarsest_step"))
                if kind == "boundary":
                    require(entity in self.boundaries, "unknown force BoundaryId")
                    item = self.boundaries[entity]
                    expected_level = item["level"]
                else:
                    require(entity in object_levels, "unknown force ObjectId")
                    expected_level = next(iter(object_levels[entity]))
                key = (kind,entity)
                expected_step = counters.get(key,0)+1
                require(level == expected_level and step == expected_step and step <= self.last*2**level and
                        coarse == (step-1)//2**level+1, "missing/duplicate/wrong force sample")
                dt = DT/2**level
                for name, value in (("begin_time",(step-1)*dt),("end_time",step*dt),("sample_time",(step-.5)*dt)):
                    require(time_matches(float(row[name]),value,step,dt),"wrong force "+name)
                force = np.array([float(row[k]) for k in ("Fx_lb","Fy_lb","Fz_lb")])
                coefficient = np.array([float(row[k]) for k in ("Cd","Cl","Cz")])
                require(np.isfinite(force).all() and np.isfinite(coefficient).all() and
                        np.all(np.abs(force/self.denominator-coefficient) <= 4096*EPS*(1+np.abs(coefficient))),
                        "nonfinite force or invalid common-unit normalization")
                counters[key] = step
                if kind == "boundary":
                    total = pending.setdefault((item["object"],step),np.zeros(3))
                    total += force
                    writer.writerow((entity,item["object"],level,step,row["sample_time"],*coefficient))
                else:
                    require((entity,step) in pending, "Object sample has no corresponding Boundary samples")
                    total = pending.pop((entity,step))
                    self.check(np.all(np.abs(total-force) <= 4096*EPS*(1+np.abs(total))),
                               f"Object {entity} population step {step} differs from its Boundary sum")
        expected = {("boundary",bid):self.last*2**b["level"] for bid,b in self.boundaries.items()}
        expected.update({("object",oid):self.last*2**next(iter(levels)) for oid,levels in object_levels.items()})
        require(all(counters.get(key,0) == count for key,count in expected.items()) and not pending,
                "missing final force sample/batch")
        return counters

    def history(self, frame_counts):
        fields = ("step","time","level","valid_cells","boxes")
        frame_levels = {}
        for step,lev in frame_counts:
            frame_levels[step] = max(frame_levels.get(step,0),lev+1)
        required_levels = max(item["level"] for item in self.boundaries.values())+1
        current, next_level = 0, 0
        def finish_step():
            if current:
                require(next_level >= required_levels, "grid history omits a boundary-owner level")
            if current and current in frame_levels:
                require(next_level == frame_levels[current], "grid history omits a saved-frame level")
        for row in csv_records(self.directory/"grid-history.csv",fields):
            step, lev, cells, boxes = (int(row[k]) for k in ("step","level","valid_cells","boxes"))
            if step != current:
                require(step == current+1 and lev == 0, "missing/duplicate grid-history step")
                finish_step()
                current, next_level = step, 0
            require(1 <= step <= self.last and lev == next_level and 0 <= lev < 3 and
                    time_matches(float(row["time"]),step*DT,step,DT) and cells > 0 and 0 < boxes <= cells,
                    "missing/duplicate/invalid grid history")
            if (step,lev) in frame_counts:
                require((cells,boxes) == frame_counts[step,lev], "grid history differs from saved layout")
            next_level += 1
        require(current == self.last, "incomplete per-L0-step grid history")
        if current:
            finish_step()

    def export(self, output):
        output.mkdir(parents=True,exist_ok=True)
        (output/"identity.txt").write_text((self.directory/"identity.txt").read_text())
        self.forces(output)
        records, frame_counts = [], {}
        for step in self.steps:
            plot = self.plot(step)
            shutil.copyfile(self.directory/f"layout-{step}.csv",output/f"layout-{step}.csv")
            masks = fluid_masks(plot,self.boundaries)
            mass = sum(float(level.values[0,mask].sum())*math.prod(level.spacing)
                       for level,mask in zip(plot.levels,masks))
            if self.dimensions == 2:
                mass /= .01
            flow = flux(plot)
            scale = .41*(.01 if self.dimensions == 3 else 1)
            record = dict(step=step,time=plot.time,fluid_center_mass=mass,**flow,
                          mass_flux_in=flow["phi_in"]*scale,mass_flux_out=flow["phi_out"]*scale,
                          mass_flux_imbalance=flow["flux_imbalance"]*scale,**quasi_metrics(plot,masks))
            require(all(math.isfinite(float(v)) for v in record.values()),"nonfinite diagnostic")
            records.append(record)
            for lev,(level,mask) in enumerate(zip(plot.levels,masks)):
                frame_counts[step,lev] = (int(level.valid.sum()),len(level.boxes))
                dump_planes(output/"planes",f"s{step}-L{lev}",
                    dict(speed=np.linalg.norm(level.values[1:],axis=0),
                         pressure=(level.values[0]-1)/(3*.035**2)),mask,level.spacing,self.dimensions,
                    origin=np.array(plot.low),level=lev,step=level.step,coarsest_step=step,time=plot.time,
                    phase="Sync",source_id=self.id["source_id"],descriptor=self.id["descriptor"],
                    backend=self.id["backend"],variant=self.variant,
                    schema=self.id["schema"],scheme=self.id["scheme"])
        self.history(frame_counts)
        shutil.copyfile(self.directory/"grid-history.csv",output/"grid-history.csv")
        write_csv(output/"fields.csv",list(records[0]),records)
        return 1 if self.failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--short",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    status, messages = 2, []
    try:
        run = DynamicCylinderRun(args.run,args.short)
        status = run.export(args.output)
        messages = run.failures
    except (ArtifactError,OSError,ValueError,KeyError,IndexError,csv.Error) as error:
        messages = [str(error)]
    (args.output/"result.txt").write_text(f"exit={status}\nscope=engineering-and-data-only\n"+
                                        "".join(f"finding={message}\n" for message in messages))
    print(f"dynamic-cylinder exit={status}; "+("; ".join(messages) if messages else "data and engineering relations valid"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
