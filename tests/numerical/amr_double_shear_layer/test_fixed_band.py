#!/usr/bin/env python3
"""Independent fixed-band relations and 0/1/2 artifact checks."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import analyze_dynamic_double_shear_layer as analysis
import check_fixed_band as checker
import render_dynamic_double_shear_layer as rendering
from amr_plotfile import ArtifactError, Level, Plot, identity, rows
from synthetic_amr import read_from
from test_dynamic_double_shear_layer import artifact as old_artifact


def write_csv(path, fields, records):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(records)


def level(dimensions, lev, step, regions):
    n = 128*2**lev
    nz = n//32 if dimensions == 3 else 1
    depth = 1/32 if dimensions == 3 else 1/16
    valid = np.zeros((nz, n, n), dtype=bool)
    boxes = []
    for lo, hi in regions:
        valid[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = True
        for k in range(lo[2], hi[2], 32):
            for j in range(lo[1], hi[1], 32):
                for i in range(lo[0], hi[0], 32):
                    boxes.append(((i, j, k), (min(i+32, hi[0]), min(j+32, hi[1]), min(k+32, hi[2]))))
    values = np.ones((4, *valid.shape))
    values[1] = .8
    values[2] = .2
    values[3] = 0
    values[:, ~valid] = np.nan
    return Level(values, valid, (1/n, 1/n, depth/nz), step*2**lev, boxes)


def write_ledgers(directory, plots):
    samples = []
    for step, plot in plots.items():
        layout = []
        for lev, data in enumerate(plot.levels):
            samples.append(dict(kind="sync", level=lev, step=data.step, time=plot.time,
                                coarsest_step=step, phase="Sync", path=f"sync-{step}"))
            for box, (lo, hi) in enumerate(data.boxes):
                layout.append(dict(zip(("level", "box", "rank", "ilo", "jlo", "klo", "ihi", "jhi", "khi"),
                                       (lev, box, box%2, *lo, *hi))))
        write_csv(directory/f"layout-{step}.csv",
                  ("level", "box", "rank", "ilo", "jlo", "klo", "ihi", "jhi", "khi"), layout)
    write_csv(directory/"samples.csv", ("kind", "level", "step", "time", "coarsest_step", "phase", "path"), samples)
    history = []
    for step in range(1, 17):
        plot = plots[0 if step < 8 else 16]
        for lev, data in enumerate(plot.levels):
            history.append(dict(step=step, time=step*.1/128, level=lev,
                                valid_cells=int(data.valid.sum()), boxes=len(data.boxes)))
    write_csv(directory/"grid-history.csv", ("step", "time", "level", "valid_cells", "boxes"), history)


def artifact(directory, dimensions, version=2):
    old_artifact(directory, dimensions, grid_efficiency=.8)
    meta = identity(directory)
    meta.update(fixed_band_y_lo=.65, fixed_band_y_hi=.85, fixed_band_parent_jlo=83,
                fixed_band_parent_jhi=109, fixed_band_aligned_y_lo=83/128, fixed_band_aligned_y_hi=109/128)
    if version == 1:
        meta.update(schema="fixed-band-double-shear-v1", case_variant="fixed-band", fixed_band_level=1)
    else:
        meta.update(schema="fixed-band-double-shear-v2", case_variant="minimum-band",
                    fixed_band_min_level=1, fixed_band_max_level=2)
    (directory/"identity.txt").write_text("".join(f"{key}={value}\n" for key, value in meta.items()))
    plots = {}
    # D2 exercises three levels and a moving lower patch. D3 uses full-z L1,
    # keeping unit-test memory small; actual C5 D3 runs must exercise L2.
    for step in (0, 16):
        data = []
        for lev in range(3 if dimensions == 2 else 2):
            n = 128*2**lev
            nz = n//32 if dimensions == 3 else 1
            lo, hi = ((0, n) if lev < 2 else ((0, 128) if step == 0 else (128, 256)))
            data.append(level(dimensions, lev, step, [((0, lo, 0), (n, hi, nz))]))
        plots[step] = Plot(step*.1/128, (0, 0, 0), (1, 1, 1/32 if dimensions == 3 else 1/16), data)
    write_ledgers(directory, plots)
    return plots


def analyze(directory, plots):
    with patch.object(analysis, "read_plot", read_from({directory: plots})):
        analysis.analyze(analysis.open_fixed_band(directory, short=True), directory/"analysis")


def result(directory, plots):
    with patch.object(analysis, "read_plot", read_from({directory: plots})):
        return checker.check_run(directory, directory/"analysis", directory/"check", short=True)


class FixedBandChecks(unittest.TestCase):
    def test_valid_d2_and_full_z_d3_pipeline(self):
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/"run"
                plots = artifact(directory, dimensions)
                analyze(directory, plots)
                self.assertEqual(result(directory, plots), 0)
                report = identity(directory/"check", "result.txt")
                self.assertEqual(report["grid_steps"], "16")
                self.assertEqual(report["l2_present_steps"], "16" if dimensions == 2 else "0")
                run = analysis.open_fixed_band(directory, short=True)
                planes, faces = rendering.load_planes(run, directory/"analysis", 1)
                self.assertEqual(len(planes), 3 if dimensions == 2 else 2)
                self.assertTrue(faces)
                if dimensions == 2:
                    path = directory/"analysis"/"planes"/"frame1-L1-xy.npz"
                    with np.load(path, allow_pickle=False) as archive:
                        data = {key: archive[key] for key in archive.files}
                    data["schema"] = "fixed-band-double-shear-v1"
                    np.savez_compressed(path, **data)
                    with self.assertRaisesRegex(ArtifactError, "fixed-band slice identity"):
                        rendering.load_planes(run, directory/"analysis", 1)
                    data["schema"] = run.id["schema"]
                    data["case_variant"] = "fixed-band"
                    np.savez_compressed(path, **data)
                    with self.assertRaisesRegex(ArtifactError, "fixed-band slice identity"):
                        rendering.load_planes(run, directory/"analysis", 1)

    def test_valid_data_missing_fixed_band_returns_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            plots = artifact(directory, 2)
            for step, plot in plots.items():
                plot.levels[1] = level(2, 1, step, [((0, 0, 0), (256, 166, 1)),
                                                    ((0, 218, 0), (256, 256, 1))])
            write_ledgers(directory, plots)
            analyze(directory, plots)
            self.assertEqual(result(directory, plots), 1)
            self.assertEqual({row["relation"] for row in rows(directory/"check"/"violations.csv")},
                             {"missing_L1_fixed_band"})

    def test_l2_inside_band_depends_on_schema(self):
        for version, expected in ((1, 1), (2, 0)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)/"run"
                plots = artifact(directory, 2, version)
                analyze(directory, plots)
                self.assertEqual(result(directory, plots), 0)
                plots[16].levels[2] = level(2, 2, 16, [((0, 128, 0), (512, 256, 1)),
                                                      ((0, 332, 0), (512, 336, 1))])
                write_ledgers(directory, plots)
                analyze(directory, plots)
                self.assertEqual(result(directory, plots), expected)
                report = identity(directory/"check", "result.txt")
                self.assertEqual(report["fixed_band_l2_frames"], "1")
                self.assertEqual(report["fixed_band_l2_cells_max"], "2048")
                violations = rows(directory/"check"/"violations.csv")
                self.assertEqual([row["relation"] for row in violations],
                                 ["L2_inside_fixed_band"] if version == 1 else [])

    def test_unknown_schema_and_wrong_minimum_bounds_return_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            plots = artifact(directory, 2)
            analyze(directory, plots)
            path = directory/"identity.txt"
            original = path.read_text()
            for old, new in (("fixed-band-double-shear-v2", "fixed-band-double-shear-v3"),
                             ("fixed_band_min_level=1", "fixed_band_min_level=0"),
                             ("fixed_band_max_level=2", "fixed_band_max_level=1"),
                             ("case_variant=minimum-band", "case_variant=fixed-band")):
                with self.subTest(new=new):
                    path.write_text(original.replace(old, new))
                    self.assertEqual(result(directory, plots), 2)
            path.write_text(original)
            self.assertEqual(result(directory, plots), 0)

    def test_fixed_band_is_checked_through_entire_z(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            plots = artifact(directory, 3)
            plots[16].levels[1] = level(3, 1, 16, [((0, 0, 0), (256, 256, 6)),
                                                       ((0, 0, 6), (256, 166, 8)),
                                                       ((0, 218, 6), (256, 256, 8))])
            write_ledgers(directory, plots)
            analyze(directory, plots)
            self.assertEqual(result(directory, plots), 1)

    def test_native_time_field_and_layout_errors_return_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            plots = artifact(directory, 2)
            analyze(directory, plots)
            original_time = plots[16].time
            plots[16].time += .1
            self.assertEqual(result(directory, plots), 2)
            plots[16].time = original_time
            plots[16].levels[2].values[0, 0, 160, 100] = np.nan
            self.assertEqual(result(directory, plots), 2)
            plots[16].levels[2].values[0, 0, 160, 100] = 1
            path = directory/"layout-16.csv"
            path.write_text(path.read_text().splitlines()[0]+"\n")
            self.assertEqual(result(directory, plots), 2)

    def test_missing_frame_and_grid_history_rows_return_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            plots = artifact(directory, 2)
            analyze(directory, plots)
            path = directory/"samples.csv"
            original = path.read_text()
            path.write_text("\n".join(original.splitlines()[:4])+"\n")
            self.assertEqual(result(directory, plots), 2)
            path.write_text(original)
            path = directory/"grid-history.csv"
            original = path.read_text()
            path.write_text("\n".join(original.splitlines()[:-1])+"\n")
            self.assertEqual(result(directory, plots), 2)
            path.write_text(original+original.splitlines()[1]+"\n")
            self.assertEqual(result(directory, plots), 2)

    def test_old_new_identity_separation_and_full_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            artifact(directory, 2)
            with self.assertRaises(ArtifactError):
                analysis.open_run(directory, short=True)
            with self.assertRaises(ArtifactError):
                analysis.open_fixed_band(directory)
            path = directory/"identity.txt"
            original = path.read_text()
            path.write_text(original.replace("grid_efficiency=0.8", "grid_efficiency=0.7"))
            with self.assertRaises(ArtifactError):
                analysis.open_fixed_band(directory, short=True)
            path.write_text(original.replace("requested_steps=16", "requested_steps=2560"))
            records = [dict(kind="sync", level=lev, step=step*2**lev, time=step*.1/128,
                            coarsest_step=step, phase="Sync", path=f"sync-{step}")
                       for step in range(0, 2561, 64) for lev in range(3)]
            write_csv(directory/"samples.csv", ("kind", "level", "step", "time", "coarsest_step", "phase", "path"), records)
            self.assertEqual(analysis.open_fixed_band(directory).steps, list(range(0, 2561, 64)))

    def test_animation_uses_saved_physical_times_and_fixed_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"run"
            artifact(directory, 2)
            run = analysis.open_fixed_band(directory, short=True)
            output = directory/"figures"
            output.mkdir()
            for frame in (0, 1):
                (output/f"frame-{frame}.png").write_bytes(b"synthetic PNG placeholder")
            write_csv(output/"render-record.csv", ("frame", "color_limit"),
                      [dict(frame=frame, color_limit=80) for frame in (0, 1)])
            rendering.write_animation(run, output, [0, 1])
            page = (output/"animation.html").read_text()
            self.assertIn('"time": 0.0125', page)
            self.assertIn('"schema": "fixed-band-double-shear-v2"', page)
            self.assertIn("next.time-current.time", page)

    def test_runner_preserves_fixed_relation_exit_and_legacy_contract(self):
        # Only shell exit propagation is exercised here; these stubs are not
        # claimed as solver, analyzer or native-data evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root/"launcher"
            launcher.write_text("#!/bin/sh\nexit 0\n")
            python = root/"consumer"
            python.write_text('#!/bin/sh\ncase "$*" in *check_*.py*) exit 1;; *) exit 0;; esac\n')
            launcher.chmod(0o755)
            python.chmod(0o755)
            env = dict(os.environ, LBM_TEST_PYTHON=str(python), LBM_TEST_MAX_STEP="16", LBM_TEST_AMR_MPI_RANKS="1")
            for entry, expected in (("d2-fixed-band", 1), ("d2-3l", 2)):
                work = root/entry
                command = ["bash", str(Path(__file__).with_name("run_dynamic.sh")),
                           "/bin/true", "/bin/true", str(work), str(launcher), entry]
                completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, expected, completed.stdout+completed.stderr)
                output = Path((work/"latest-run.txt").read_text().strip())
                self.assertEqual(identity(output.parent, "result.txt")["exit"], str(expected))


if __name__ == "__main__":
    unittest.main()
