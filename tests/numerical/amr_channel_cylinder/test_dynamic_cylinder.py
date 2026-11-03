#!/usr/bin/env python3
"""Small independent dynamic DFG artifact fixtures; only binary field reads are replaced."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import check_dynamic_cylinder as checker
import render_dynamic_cylinder as renderer
from amr_plotfile import ArtifactError, Level, Plot, rows
from amr_validation import write_csv
from synthetic_amr import write_run, read_from


def write_layout(directory,step,plot):
    records=[]
    for lev,level in enumerate(plot.levels):
        for index,(lo,hi) in enumerate(level.boxes):
            records.append(dict(level=lev,box=index,rank=0,ilo=lo[0],jlo=lo[1],klo=lo[2],
                                ihi=hi[0],jhi=hi[1],khi=hi[2]))
    write_csv(directory/f"layout-{step}.csv",list(records[0]),records)


def fixture(directory, variant="single", dimensions=2, last=2, bad_region=False,
            initially_connected=False, shared_object=False, levels=3):
    denominator=.5*.035**2*80*(8 if dimensions == 3 else 1)
    meta=dict(schema="dynamic-cylinder-v1",descriptor="D3Q27" if dimensions == 3 else "D2Q9",
              dimensions=dimensions,backend="cpu",collision="BGK",real_digits=53,source_id="a"*40,ranks=1,
              layout="dynamic",variant=variant,scheme="YMS",periodic="0,0,1",
              initialization="uniform-equilibrium-then-Guo",force_units="allowed-finest-lattice",
              max_level=2,base_z=2 if dimensions == 3 else 1,max_grid_size=64,regrid_interval=8,
              full_steps=200000,requested_steps=last,frames=2 if last else 1,
              overview_stride=4000,detail_begin=160000,detail_stride=400,profile=0,
              reynolds=100,viscosity=.001,u0_lb=.035,u0_phys=1,rho_reference=1,velocity_reference=.035,
              length_z=.01,gap_fill_radius=4,coarse_nesting_margin=4,diameter_lb=80,
              span_lb=8 if dimensions == 3 else 1,force_denominator=denominator,grid_efficiency=1)
    for lev in range(3):
        meta[f"dt_{lev}"]=checker.DT/2**lev
        meta[f"tau_{lev}"]=.5+3*.001*(checker.DT/2**lev)/(.005/2**lev)**2
        meta[f"tag_buffer_{lev}"]=2
        if lev < 2:
            meta[f"theta_refine_{lev}"]=20
            meta[f"theta_keep_{lev}"]=10
    plots={}
    for step in sorted({0,last}):
        frame_levels=[]
        for lev in range(levels):
            factor=2**lev
            shape=(2*factor if dimensions == 3 else 1,83*factor,441*factor)
            if lev == 0:
                intervals=[(0,441,0,83)]
            elif lev == 1:
                intervals=[(48,194 if variant != "single" else 118,48,114)]
            else:
                hi=216 if bad_region else 218
                intervals=[(106,hi,106,218)]
                if variant == "tandem-a" or (variant == "tandem-b" and (step or initially_connected)):
                    intervals=[(106,378,106,218)]
                elif variant == "tandem-b":
                    intervals.append((266,378,106,218))
                elif step and not bad_region:
                    intervals=[(106,226,106,218)]
            boxes=[((lo,y0,0),(hi,y1,shape[0])) for lo,hi,y0,y1 in intervals]
            valid=np.zeros(shape,bool)
            for lo,hi in boxes:
                valid[lo[2]:hi[2],lo[1]:hi[1],lo[0]:hi[0]]=True
            values=np.broadcast_to(np.array([1.,1.,0.,0.])[:,None,None,None],(4,*shape))
            spacing=(.005/factor,.005/factor,.005/factor if dimensions == 3 else .01)
            frame_levels.append(Level(values,valid,spacing,step*factor,boxes))
        plots[step]=Plot(step*checker.DT,checker.LOW,checker.HIGH,frame_levels)
    write_run(directory,meta,plots)
    for step,plot in plots.items():
        write_layout(directory,step,plot)
    boundaries=[]
    for index in range(1 if variant == "single" else 2):
        boundaries.append(dict(name="cylinder_a" if index == 0 else "cylinder_b",
            boundary_id=4+index,object_id=0 if shared_object else index,level=levels-1,
            center_x=.2+.2*index,center_y=.2,center_z=.005,radius=.05))
    write_csv(directory/"boundaries.csv",list(boundaries[0]),boundaries)
    forces=[]
    factor=2**(levels-1)
    for coarse in range(1,last+1):
        for kind in ("boundary","object"):
            for step in range(factor*(coarse-1)+1,factor*coarse+1):
                entities=boundaries if kind == "boundary" else (
                    [dict(boundary_id=-1,object_id=0)] if shared_object else boundaries)
                for item in entities:
                    bid=item["boundary_id"]
                    cd=2+.1*bid if kind == "boundary" or not shared_object else sum(2+.1*b["boundary_id"] for b in boundaries)
                    entity=bid if kind == "boundary" else item["object_id"]
                    dt=checker.DT/factor
                    forces.append(dict(kind=kind,id=entity,level=levels-1,population_step=step,coarsest_step=coarse,
                        begin_time=(step-1)*dt,end_time=step*dt,sample_time=(step-.5)*dt,
                        Fx_lb=cd*denominator,Fy_lb=0,Fz_lb=0,Cd=cd,Cl=0,Cz=0))
    write_csv(directory/"forces.csv",checker.FORCES,forces)
    history=[]
    for step in range(1,last+1):
        for lev,level in enumerate(plots[last].levels):
            history.append(dict(step=step,time=step*checker.DT,level=lev,valid_cells=int(level.valid.sum()),boxes=len(level.boxes)))
    write_csv(directory/"grid-history.csv",("step","time","level","valid_cells","boxes"),history)
    return plots,forces


class DynamicCylinderChecks(unittest.TestCase):
    def setUp(self):
        scratch=Path(os.environ.get("TMPDIR",Path(__file__).resolve().parent/"build"/"consumer-tests"))
        scratch.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch,prefix="dynamic-cylinder-checker-")
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self,directory,plots,short=True,planes=False):
        args=["check","--run",str(directory),"--output",str(self.root/"analysis")]
        if short:
            args.append("--short")
        with patch("check_dynamic_cylinder.read_plot",side_effect=read_from({directory:plots})),\
             (contextlib.nullcontext() if planes else patch("check_dynamic_cylinder.dump_planes")),\
             patch("sys.argv",args),contextlib.redirect_stdout(io.StringIO()):
            return checker.main()

    def test_141_frame_union_and_short_terminal(self):
        full=checker.sample_steps(200000)
        self.assertEqual(len(full),141)
        self.assertEqual(len([s for s in full if s%4000 == 0]),51)
        self.assertEqual(len([s for s in full if s >= 160000]),101)
        self.assertEqual(checker.sample_steps(32),[0,32])
        self.assertEqual(checker.sample_steps(160001)[-2:],[160000,160001])

    def test_all_three_layouts_allow_later_connectivity_and_field_changes(self):
        for variant in ("single","tandem-a","tandem-b"):
            directory=self.root/variant
            plots,_=fixture(directory,variant)
            self.assertEqual(self.evaluate(directory,plots),0)
        self.assertEqual(self.evaluate(directory,plots,False),2)

    def test_d3_complete_z_no_scientific_quasi_gate(self):
        directory=self.root/"d3"
        plots,_=fixture(directory,dimensions=3,last=0)
        level=plots[0].levels[2]
        values=np.array([1.,1.,0.,.03])[:,None,None,None]
        level.values=np.broadcast_to(values,level.values.shape)
        self.assertEqual(self.evaluate(directory,plots),0)

    def test_both_cylinders_are_excluded_from_composite_mask(self):
        directory=self.root/"mask"
        plots,_=fixture(directory,"tandem-b")
        run=checker.DynamicCylinderRun(directory,True)
        mask=checker.fluid_masks(plots[0],run.boundaries)[2]
        self.assertFalse(mask[0,162,162])
        self.assertFalse(mask[0,162,322])
        self.assertTrue(mask[0,110,110])

    def test_initial_connection_or_missing_fixed_cells_is_one(self):
        for name,kwargs in (("joined",dict(variant="tandem-b",initially_connected=True)),
                            ("missing",dict(bad_region=True))):
            directory=self.root/name
            plots,_=fixture(directory,**kwargs)
            self.assertEqual(self.evaluate(directory,plots),1)

    def test_complete_two_level_history_is_engineering_failure(self):
        directory=self.root/"two-level"
        plots,_=fixture(directory,levels=2)
        self.assertEqual(self.evaluate(directory,plots),1)

    def test_non_midplane_bridge_connects_initial_three_dimensional_regions(self):
        directory=self.root/"off-midplane"
        plots,_=fixture(directory,"tandem-b",dimensions=3,last=0)
        fine=plots[0].levels[2]
        self.assertFalse(checker.connected(fine.valid[4:5],(0,162,162),(0,162,322)))
        fine.boxes.append(((218,160,0),(266,164,2)))
        fine.valid[0:2,160:164,218:266]=True
        write_layout(directory,0,plots[0])
        self.assertFalse(checker.connected(fine.valid[4:5],(0,162,162),(0,162,322)))
        self.assertTrue(checker.connected(fine.valid,(4,162,162),(4,162,322)))
        self.assertEqual(self.evaluate(directory,plots),1)

    def test_complete_shared_object_or_wrong_sum_is_one(self):
        directory=self.root/"shared"
        plots,_=fixture(directory,"tandem-b",shared_object=True)
        self.assertEqual(self.evaluate(directory,plots),1)
        directory=self.root/"sum"
        plots,forces=fixture(directory)
        row=next(row for row in forces if row["kind"] == "object")
        row["Fx_lb"]*=2
        row["Cd"]*=2
        write_csv(directory/"forces.csv",checker.FORCES,forces)
        self.assertEqual(self.evaluate(directory,plots),1)

    def test_force_missing_duplicate_time_units_nonfinite_are_two(self):
        directory=self.root/"force"
        plots,original=fixture(directory)
        for problem in ("missing","duplicate","time","units","nonfinite"):
            forces=copy.deepcopy(original)
            if problem == "missing": forces.pop()
            elif problem == "duplicate": forces.insert(1,copy.deepcopy(forces[0]))
            elif problem == "time": forces[0]["sample_time"]+=checker.DT
            elif problem == "units": forces[0]["Fx_lb"]*=2
            else: forces[0]["Cl"]=float("nan")
            write_csv(directory/"forces.csv",checker.FORCES,forces)
            self.assertEqual(self.evaluate(directory,plots),2,problem)

    def test_missing_frame_wrong_clock_layout_or_history_is_two(self):
        for problem in ("missing","clock","layout","history"):
            directory=self.root/problem
            plots,_=fixture(directory)
            if problem in ("missing","clock"):
                samples=rows(directory/"samples.csv")
                if problem == "missing": samples=samples[:3]
                else: samples[-1]["time"]=checker.DT
                write_csv(directory/"samples.csv",list(samples[0]),samples)
            elif problem == "layout":
                layout=rows(directory/"layout-2.csv")
                layout[-1]["rank"]=1
                write_csv(directory/"layout-2.csv",list(layout[0]),layout)
            else:
                history=rows(directory/"grid-history.csv")[:-1]
                write_csv(directory/"grid-history.csv",list(history[0]),history)
            self.assertEqual(self.evaluate(directory,plots),2,problem)

    def test_missing_unsaved_step_finest_history_is_two(self):
        directory=self.root/"unsaved-history"
        plots,_=fixture(directory,"tandem-b")
        history=rows(directory/"grid-history.csv")
        history=[row for row in history if not (int(row["step"]) == 1 and int(row["level"]) == 2)]
        write_csv(directory/"grid-history.csv",list(history[0]),history)
        self.assertEqual(self.evaluate(directory,plots),2)

    def test_native_plane_export_and_render_identity(self):
        directory=self.root/"planes"
        plots,_=fixture(directory,last=0)
        self.assertEqual(self.evaluate(directory,plots,planes=True),0)
        output=self.root/"analysis"
        meta=checker.identity(output)
        planes,faces=renderer.load_planes(output,meta,0)
        self.assertEqual(len(planes),3)
        self.assertTrue(faces)
        self.assertTrue((output/"grid-history.csv").exists())
        path=output/"planes"/"s0-L2-xy.npz"
        with np.load(path) as data:
            damaged={key:data[key] for key in data.files}
        damaged["source_id"]="b"*40
        np.savez(path,**damaged)
        with self.assertRaises(ArtifactError):
            renderer.load_planes(output,meta,0)
        damaged["source_id"]=meta["source_id"]
        damaged["scheme"]="Bouzidi"
        np.savez(path,**damaged)
        with self.assertRaises(ArtifactError):
            renderer.load_planes(output,meta,0)

    def test_native_reader_invalid_field_error_is_two(self):
        directory=self.root/"field"
        plots,_=fixture(directory)
        args=["check","--run",str(directory),"--output",str(self.root/"analysis"),"--short"]
        with patch("check_dynamic_cylinder.read_plot",side_effect=ArtifactError("incomplete/nonfinite FAB")),\
             patch("sys.argv",args),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(),2)


if __name__ == "__main__":
    unittest.main()
