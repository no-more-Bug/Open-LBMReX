"""Small v3 consumer regressions, including real CLI exit statuses 0/1/2."""

import copy
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from static_hierarchy_artifacts import (ArtifactError, NumericalMismatch, analytic_macros, equilibrium, grow,
                                        load_plotfile_expectations, repair_tags, require, validate_geometry)
from check_static_hierarchy_parity import columns, compare_pair


def identity(dimensions, multilevel=False, efficiency=1):
    nz = 8 if dimensions == 3 else 1
    result = dict(schema="static-hierarchy-v3", sample="sample", source_id="test", profile=
                  "d3-periodic-xyz-v1" if dimensions == 3 else "layout-periodic-v1",
                  descriptor="D3Q27" if dimensions == 3 else "D2Q9", d=str(dimensions),
                  q="27" if dimensions == 3 else "9", backend="cpu", ranks="1", levels="1",
                  rho_reference="1", velocity_reference="0.02", base_cells=f"8,8,{nz}",
                  prob_lo="0,0,0", prob_hi="1,1,1" if dimensions == 3 else "1,1,0.0625",
                  periodic="1,1,1", ref_ratio="2,2,2" if dimensions == 3 else "2,2,1",
                  steps="0", time_0="0", regions="none", gap_fill_radius_cells="1",
                  coarse_nesting_margin_cells="1", grid_eff=str(efficiency),
                  coverage_contract="exact-repair" if efficiency == 1 else "minimum-repair",
                  max_grid_size=f"8,8,{nz}", domain_0=f"0,8,0,8,0,{nz}", boxes_0=f"0,8,0,8,0,{nz}")
    if multilevel:
        result.update(levels="2", regions="1,0.25,0.5,0.25,0.5,"+
                      ("0.25,0.5" if dimensions == 3 else "0,0.0625"),
                      domain_1="0,16,0,16,0,"+("16" if dimensions == 3 else "1"),
                      boxes_1="4,8,4,8,"+("4,8" if dimensions == 3 else "0,1"))
    return result


def rows_for(meta):
    geometry = validate_geometry(meta)
    rows = []
    for lev, coverage in enumerate(geometry.coverage):
        for cell in sorted(coverage):
            initial = analytic_macros(geometry, lev, cell, 0)
            rows.append([lev, 0, 0, *cell, *equilibrium(initial, geometry.dimensions), *initial])
    return rows


def overlapping_middle_identity(dimensions):
    """Hand-specified overlapping L1 requests with two separated L2 patches."""
    meta = identity(dimensions)
    z0, z1, z2 = (16,32,64) if dimensions == 3 else (1,1,1)
    middle_z = "4,20" if dimensions == 3 else "0,1"
    fine_z = "16,32" if dimensions == 3 else "0,1"
    request_z = "0.125,0.625" if dimensions == 3 else "0,0.0625"
    fine_request_z = "0.25,0.5" if dimensions == 3 else "0,0.0625"
    meta.update(base_cells=f"16,16,{z0}", max_grid_size=f"32,32,{z1}", levels="3",
                domain_0=f"0,16,0,16,0,{z0}", boxes_0=f"0,16,0,16,0,{z0}",
                domain_1=f"0,32,0,32,0,{z1}", boxes_1=f"4,28,4,20,{middle_z}",
                domain_2=f"0,64,0,64,0,{z2}",
                boxes_2=f"16,24,16,32,{fine_z};40,48,16,32,{fine_z}",
                regions=f"1,0.125,0.5625,0.125,0.625,{request_z};"
                        f"1,0.4375,0.875,0.125,0.625,{request_z};"
                        f"2,0.25,0.375,0.25,0.5,{fine_request_z};"
                        f"2,0.625,0.75,0.25,0.5,{fine_request_z}")
    return meta


def write(path, meta, rows, header=None):
    path.mkdir(parents=True, exist_ok=True)
    (path/"identity.txt").write_text("".join(f"{k}={v}\n" for k, v in sorted(meta.items())))
    with (path/"states.csv").open("w", newline="") as output:
        csv.writer(output).writerows([columns(int(meta["q"])) if header is None else header, *rows])


def run():
    # Independently specified discrete counts, including the batch-order counterexample.
    require(len(repair_tags({(2,2,0),(4,4,0)}, (12,12,1), 1, 2)) == 17, "diagonal repair count")
    require(len(repair_tags({(2,2,0),(3,3,0),(4,4,0)}, (12,12,1), 1, 2)) == 3, "batch union count")
    for gap, count in ((7,9), (8,10), (9,2)):
        tags = {(x,y,0) for x in (5,6+gap) for y in range(40)}
        require(len(repair_tags(tags,(40,40,1),4,2)) == 40*count, "default radius threshold")
    require(len(grow({(0,0,0)}, (2,2,2), 8, 3)) == 8, "multiple-period normalization")
    temporary_root = Path(os.environ.get("TMPDIR", Path(__file__).resolve().parent/"build"/"consumer-tests"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="static-v3-", dir=temporary_root) as temporary:
        root = Path(temporary)
        left, right = root/"left"/"sample", root/"right"/"sample"
        command = [sys.executable, "-B", str(Path(__file__).with_name("check_static_hierarchy_parity.py")),
                   "--left", str(left.parent), "--right", str(right.parent), "--samples", "sample",
                   "--source-id", "test"]
        calls = 0
        def check(code):
            nonlocal calls
            result = subprocess.run(command, capture_output=True, text=True)
            require(result.returncode == code, f"CLI expected {code}, got {result.returncode}: {result.stdout}{result.stderr}")
            calls += 1
            return result.stdout
        for dimensions in (2,3):
            meta = identity(dimensions, multilevel=True)
            rows = rows_for(meta)
            write(left,meta,rows); write(right,meta,rows); check(0)
            changed = copy.deepcopy(rows); changed[0][6] += 1e-4
            write(right,meta,changed); check(1)
            changed[0][6] = float("nan"); write(right,meta,changed); check(2)
            write(right,meta,rows+rows[:1]); check(2)
            write(right,meta,rows[:-1]); check(2)
            write(left,meta,rows[:-1]); check(2)  # Missing on both sides cannot pass by intersection.
            write(left,meta,rows)
            if dimensions == 3:
                write(right,meta,[row for row in rows if row[5] != 7]); check(2)
            for key, value in (("schema","static-hierarchy-v2"), ("q","8"), ("levels","3"),
                               ("gap_fill_radius_cells","0"), ("coarse_nesting_margin_cells","-1"),
                               ("grid_eff","nan"), ("regions","none"),
                               ("regions","0,0.25,0.5,0.25,0.5,0,0.0625"),
                               ("source_id","other"), ("time_0","nan"), ("boxes_1",meta["boxes_1"]+";"+meta["boxes_1"])):
                changed_meta = dict(meta, **{key:value})
                write(right,changed_meta,rows,header=columns(int(meta["q"]))); check(2)
            missing_q = columns(int(meta["q"])); del missing_q[6]
            write(right,meta,[row[:6]+row[7:] for row in rows],header=missing_q); check(2)
            changed = copy.deepcopy(rows); changed[0][0] = 8
            write(right,meta,changed); check(2)
            write(right,meta,rows)
            (right/"identity.txt").unlink(); check(2)
            print(f"static v3 consumer regression: PASS dimensions={dimensions}")
        # A shared wrong producer cannot impersonate an explicitly requested sample.
        meta = identity(2, True)
        rows = rows_for(meta)
        write(left,dict(meta,sample="different"),rows)
        write(right,dict(meta,sample="different"),rows)
        check(2)

        # All intermediate levels remain mandatory when only the finest is requested.
        meta = identity(2)
        meta.update(base_cells="2,2,1", max_grid_size="16,16,1", levels="4",
                    regions="3,0,1,0,1,0,0.0625")
        for level in range(4):
            box = f"0,{2 << level},0,{2 << level},0,1"
            meta[f"domain_{level}"] = meta[f"boxes_{level}"] = box
        rows = rows_for(meta)
        write(left,meta,rows); write(right,meta,rows); check(0)
        incomplete = [row for row in rows if row[0] != 1]
        write(left,meta,incomplete); write(right,meta,incomplete); check(2)

        # Both producers losing an entire disconnected patch must still fail.
        for dimensions in (2,3):
            meta = overlapping_middle_identity(dimensions)
            rows = rows_for(meta)
            write(left,meta,rows); write(right,meta,rows); check(0)
            incomplete = [row for row in rows if not (row[0] == 2 and row[3] >= 40)]
            write(left,meta,incomplete); write(right,meta,incomplete); check(2)
            lost = dict(meta, boxes_2=meta["boxes_2"].split(";")[0])
            write(left,lost,incomplete); write(right,lost,incomplete); check(2)
            middle_z = "4,20" if dimensions == 3 else "0,1"
            for boxes in (f"4,18,4,20,{middle_z}",
                          f"4,18,4,20,{middle_z};14,28,4,20,{middle_z}"):
                changed = dict(meta, boxes_1=boxes)
                write(left,changed,rows); write(right,changed,rows); check(2)

        # The plotfile oracle is a separate consumer chain, checked before ParaView.
        meta = identity(3, True)
        rows = rows_for(meta)
        output = root/"output"
        output.mkdir()
        write(output,dict(meta,sample="output"),rows)
        oracle = [[row[0], *row[3:6], row[-4], *(value/0.02 for value in row[-3:])] for row in rows]
        def output_check(code, data):
            with (output/"expected.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows([["level","i","j","k","rho_norm","ux_norm","uy_norm","uz_norm"], *data])
            actual = 0
            try:
                load_plotfile_expectations(output,"test")
            except NumericalMismatch:
                actual = 1
            except (ArtifactError, ValueError, KeyError):
                actual = 2
            require(actual == code, f"plotfile oracle expected {code}, got {actual}")
        output_check(0,oracle)
        output_check(2,oracle[:-1])
        output_check(2,oracle+oracle[:1])
        output_check(2,[row for row in oracle if row[3] != 7])
        bad = copy.deepcopy(oracle); bad[0][4] += 1e-4; output_check(1,bad)
        bad[0][4] = float("nan"); output_check(2,bad)
        print("static v3 plotfile oracle regressions: PASS cases=6")

        # A low grid efficiency does not exempt equal coverage from numerical parity.
        # Evolved nonuniform states have a moment oracle, but no pointwise analytic solution.
        for dimensions in (2,3):
            meta = identity(dimensions, efficiency=0.7)
            meta["profile"] = "tgv-bgk-nu0.05-u0.02-v1" if dimensions == 2 else "D3-S1-bgk-nu0.05-uref0.02-v1"
            initial = rows_for(meta)
            timestep = 0.000625 if dimensions == 2 else 0.00125
            evolved = copy.deepcopy(initial)
            for row in evolved:
                row[1], row[2] = 1, timestep
            meta.update(steps="0,1", time_1=str(timestep))
            rows = initial+evolved
            write(left,meta,rows); write(right,meta,rows)
            require("comparison=full-coordinate" in check(0), "equal coverage was not paired")
            changed = copy.deepcopy(rows)
            for row in changed:
                if row[1] == 1:
                    for component in range(int(meta["q"])):
                        row[6+component] *= 1.0001
                    row[-4] *= 1.0001
            write(right,meta,changed); check(1)

        # Low-efficiency layouts can differ, but each must preserve all requests,
        # nesting, fixed-point separation and independent analytic fields.
        meta = identity(2, True, 0.7)
        other = dict(meta, boxes_1="4,10,4,8,0,1")
        write(left,meta,rows_for(meta)); write(right,other,rows_for(other))
        report = check(0)
        require("comparison=independent-analytic" in report and "max_abs=not-compared" in report,
                "different coverage was reported as a numeric equality")
        changed = dict(other, regions="1,0.375,0.5,0.25,0.5,0,0.0625")
        write(right,changed,rows_for(changed)); check(2)  # A request lost on one side changes the scientific identity.
        write(right,other,rows_for(other))
        for directory in (left.parent, right.parent):
            shutil.copytree(directory/"sample", directory/"second")
            path = directory/"second"/"identity.txt"
            path.write_text(path.read_text().replace("sample=sample\n", "sample=second\n"))
        # Use an explicit complete collection, then remove the same sample from both.
        command = command[:-2] + ["second", "--source-id", "test"]
        check(0)
        for directory in (left.parent, right.parent):
            shutil.rmtree(directory/"second")
        check(2)
        print(f"static v3 CLI exit-status regressions: PASS cases={calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
