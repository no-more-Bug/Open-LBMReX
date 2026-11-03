"""Independent small CSV examples for work, windows, shares and comparison policy."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from functools import partial
from check_amr_performance_csv import ValidationError, validate as validate_artifact, artifact, rows
from summarize_amr_performance import comparison_groups, summarize


EXPECTED_SOURCE = "4ceb814db7d79c5858b13dab153fb04ecc3b129d"
EXPECTED_PREFIXES = ("/opt/test-amrex/amr-cpu", "/opt/test-amrex/amr-cuda")
validate = partial(validate_artifact, expected_source=EXPECTED_SOURCE, expected_prefixes=EXPECTED_PREFIXES)


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def fixture(directory, layout="amr", breakdown=False, ranks=16, backend="cpu"):
    path = directory/"window.csv"
    count = 16 if layout == "amr" else 32
    warmup = 4 if layout == "amr" else 8
    grids = [80,160] if layout == "amr" else [160]
    levels, boxes = [], []
    for lev,n in enumerate(grids):
        lo, hi = (40,119) if lev else (0,n-1)
        cells = (hi-lo+1)**2
        steps = count*2**lev
        levels.append(dict(level=lev,nx=n,ny=n,nz=1,valid_cells=cells,boxes=1,dt=5/(960*n),
                           tau=.5+3*.1*5*n/960,measured_steps=steps,valid_updates=cells*steps))
        boxes.append(dict(level=lev,box=0,rank=0,lo_x=lo,lo_y=lo,lo_z=0,hi_x=hi,hi_y=hi,hi_z=0))
    updates = sum(level["valid_updates"] for level in levels)
    record = dict(schema="amr-performance-v1",workload="tgv_d2q9",layout=layout,descriptor="D2Q9",backend=backend,
        run_mode="calibration",measurement_mode="stage_breakdown" if breakdown else "throughput",repeat=1,
        git_commit="a"*40,amrex_source_commit=EXPECTED_SOURCE,
        amrex_prefix=EXPECTED_PREFIXES[backend=="cuda"],build_type="Release",compiler="gcc",
        mpi="OpenMPI-test",cpu_model="fixture",mpi_ranks=ranks,omp_threads=1,node="test",partition="debug",qos="debug",job_id="1",
        initialization_s=1,warmup_steps=warmup,nsteps=count,sample_start=warmup+1,sample_end=warmup+count,
        physical_begin=warmup*levels[0]["dt"],physical_end=(warmup+count)*levels[0]["dt"],window_s=12,
        critical_rank=0 if breakdown else -1,critical_local_s=10 if breakdown else "",valid_updates=updates,
        mlups="" if breakdown else updates/12e6,finite_final_state=1)
    write(path, [record]); write(artifact(path,"levels"), levels); write(artifact(path,"boxes"), boxes)
    if breakdown:
        stages = []
        def stage(owner, lev, name, calls):
            seconds = .2 if calls else 0
            stages.append(dict(owner=owner,level=lev,stage=name,critical_s=seconds,calls=calls,
                rank_max_s=2 if calls else 0,rank_max_calls=calls,share_percent=10*seconds,
                mean_call_s=seconds/calls if calls else "",
                mean_level_step_s=seconds/(count*2**lev) if owner=="level" else ""))
        for lev,level in enumerate(levels):
            steps = level["measured_steps"]
            for name in ("collision","communication","streaming","boundary_post"):
                stage("level",lev,name,steps)
            stage("level",lev,"boundary_pre",steps if lev==0 else 0)
            stage("level",lev,"management",2*steps)
        for lev in range(len(levels)-1):
            for name in ("coarse_to_fine_communication","fine_to_coarse_communication","explode","coalesce"):
                stage("pair",lev,name,count*2**lev)
        stage("hierarchy",-1,"management",2*count)
        write(artifact(path,"stages"),stages)
    width = 1 if backend=="cpu" else 8 if ranks==1 else 3
    for rank in range(ranks):
        write(artifact(path,"rank-"+str(rank)), [dict(rank=rank,cpuset=",".join(str(rank*width+i) for i in range(width)),
            omp_threads=1,device_visible="MIG-"+str(rank),device_count=1,device_name="A100 MIG 1g.5gb",
            device_total_bytes=5_000_000_000,after_init_used_bytes=1_000_000_000,after_window_used_bytes=2_000_000_000,
            host_process_peak_rss_kib=1000,device_peak_status="not_measured")])
    if backend=="cuda":
        (directory/"nvidia-smi.txt").write_text("GPU 0: A100 (UUID: GPU-parent)\n"+
            "".join(f"  MIG 1g.5gb Device {rank}: (UUID: MIG-{rank})\n" for rank in range(ranks)))
    return path


class Consumers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_work_and_physical_pair(self):
        amr = validate(fixture(self.root/"amr"))
        uniform = validate(fixture(self.root/"uniform",layout="uniform"))
        self.assertEqual(int(amr["valid_updates"]), 307200)
        self.assertEqual(int(uniform["valid_updates"]), 819200)
        self.assertEqual(len(comparison_groups([amr,uniform])),1)
        plans = summarize([amr,uniform],self.root/"summary",True)
        self.assertEqual(plans[1]["measurement_steps"],2*plans[0]["measurement_steps"])
        self.assertAlmostEqual(plans[0]["physical_end"],plans[1]["physical_end"])

    def test_stage_critical_rank_not_independent_max_sum(self):
        amr = validate(fixture(self.root/"amr",breakdown=True))
        uniform = validate(fixture(self.root/"uniform",layout="uniform",breakdown=True))
        self.assertGreater(sum(float(row["rank_max_s"]) for row in amr["stages"]),10)
        summarize([amr,uniform],self.root/"summary")
        shares = [row for row in rows(self.root/"summary/stages.csv") if row["layout"]=="amr"]
        self.assertAlmostEqual(sum(float(row["share_percent"]) for row in shares),100)
        self.assertTrue(any(row["owner"]=="other" for row in shares))

    def test_wrong_work_and_subcycling(self):
        path=fixture(self.root/"run")
        levels=rows(artifact(path,"levels")); levels[1]["measured_steps"]="16"
        write(artifact(path,"levels"),levels)
        with self.assertRaises(ValidationError): validate(path)

    def test_overlapping_stage_or_duplicate_owner(self):
        path=fixture(self.root/"run",breakdown=True)
        stages=rows(artifact(path,"stages")); stages.append(stages[-1])
        write(artifact(path,"stages"),stages)
        with self.assertRaises(ValidationError): validate(path)

    def test_throughput_has_no_fake_stage_and_formal_minimum(self):
        path=fixture(self.root/"run")
        write(artifact(path,"stages"),[dict(stage="collision",critical_s=0)])
        with self.assertRaises(ValidationError): validate(path)
        artifact(path,"stages").unlink()
        record=rows(path); record[0]["run_mode"]="baseline"; write(path,record)
        with self.assertRaises(ValidationError): validate(path)

    def test_explicit_dependency_mismatch(self):
        for field, value in (("amrex_source_commit", "d"*40), ("amrex_prefix", "/different/package")):
            path=fixture(self.root/field)
            record=rows(path); record[0][field]=value; write(path,record)
            with self.assertRaises(ValidationError): validate(path)

    def test_mismatched_interval_and_duplicate_repeats(self):
        amr=validate(fixture(self.root/"amr"))
        uniform=validate(fixture(self.root/"uniform",layout="uniform"))
        with self.assertRaises(ValidationError): comparison_groups([amr,uniform,amr])
        uniform["physical_end"]=str(float(uniform["physical_end"])*2)
        with self.assertRaises(ValidationError): comparison_groups([amr,uniform])

    def test_mig_parent_and_cpu_overlap(self):
        path=fixture(self.root/"run",ranks=4,backend="cuda")
        validate(path)
        resource=rows(artifact(path,"rank-1")); resource[0]["cpuset"]="0,1,2"
        write(artifact(path,"rank-1"),resource)
        with self.assertRaises(ValidationError): validate(path)
        resource[0]["cpuset"]="3,4,5"; write(artifact(path,"rank-1"),resource)
        (path.parent/"nvidia-smi.txt").write_text(
            "GPU 0: A100 (UUID: GPU-a)\n  MIG 1g.5gb (UUID: MIG-0)\n"
            "GPU 1: A100 (UUID: GPU-b)\n  MIG 1g.5gb (UUID: MIG-1)\n"
            "  MIG 1g.5gb (UUID: MIG-2)\n  MIG 1g.5gb (UUID: MIG-3)\n")
        with self.assertRaises(ValidationError): validate(path)

    def test_same_mig_model_on_different_parent_is_not_a_scaling_group(self):
        group=[validate(fixture(self.root/f"{layout}-{ranks}",layout=layout,ranks=ranks,backend="cuda"))
               for layout in ("amr","uniform") for ranks in (1,4)]
        comparison_groups(group)
        group[0]["gpu_parent_uuid"]="GPU-another"
        with self.assertRaises(ValidationError): comparison_groups(group)


if __name__=="__main__":
    unittest.main()
