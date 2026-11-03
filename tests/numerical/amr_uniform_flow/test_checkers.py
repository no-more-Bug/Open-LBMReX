#!/usr/bin/env python3
import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import check_uniform_flow as checker
from amr_plotfile import ArtifactError, Plot, Level, composite_masks
from amr_validation import write_csv
from synthetic_amr import write_run, read_from


def fixture(directory, last=32000, dimensions=2):
    meta = dict(schema="amr-uniform-v1", descriptor="D2Q9" if dimensions == 2 else "D3Q27",
                dimensions=dimensions, backend="cpu", collision="BGK", real_digits=53, source_id="a"*40,
                ranks=4, levels=3, max_grid_size=16, periodic="0,1,1", boundary="guo_velocity_io",
                initialization="constant-equilibrium", u0_lb=.1, u0_phys=1, nu_fine_lb=1/12,
                rho_reference=1, velocity_reference=.1, length_z=.125, full_steps=32000, requested_steps=last)
    for lev, (dt, tau) in enumerate(((.003125, .5625), (.0015625, .625), (.00078125, .75))):
        meta[f"dt_{lev}"], meta[f"tau_{lev}"] = dt, tau
    plots = {}
    for step in sorted({0, last} | ({16000} if last >= 16000 else set())):
        levels = []
        for lev, bound in enumerate((None, (16, 48), (48, 80))):
            nx, ny = 33*2**lev, 32*2**lev
            nz = 4*2**lev if dimensions == 3 else 1
            shape = (nz, ny, nx)
            low, high = ((0, 0, 0), (nx, ny, nz)) if bound is None else ((bound[0], bound[0], 0), (bound[1], bound[1], nz))
            valid = np.zeros(shape, bool)
            valid[:, low[1]:high[1], low[0]:high[0]] = True
            values = np.zeros((4, *shape)); values[0:2] = 1
            levels.append(Level(values, valid, (1/ny, 1/ny, .125/nz), step*2**lev, [(low, high)]))
        plots[step] = Plot(step/320, (-1/64, 0, 0), (1+1/64, 1, .125), levels)
    write_run(directory, meta, plots)
    mass = 33/32*(.125 if dimensions == 3 else 1)
    records = [dict(step=step, time=step/320, mass=mass, relative_l2=0, uz_max=0, rho_z_max=0, velocity_z_max=0)
               for step in sorted({0, last} | set(range(320, last+1, 320)))]
    write_csv(directory/"observations.csv", list(records[0]), records)
    return plots, records


class UniformChecks(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("TMPDIR", Path(__file__).resolve().parent/"build"/"consumer-tests"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch, prefix="uniform-checker-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, directory, plots, engineering=False):
        args = ["check", "--run", str(directory), "--output", str(self.root/"analysis")]
        if engineering:
            args.append("--engineering")
        with patch("amr_validation.read_plot", side_effect=read_from({directory: plots})), patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            return checker.main()

    def test_constant_fields_and_dimension_weights(self):
        for dimensions in (2, 3):
            directory = self.root/str(dimensions)
            plots, _ = fixture(directory, dimensions=dimensions)
            self.assertEqual(self.evaluate(directory, plots), 0)
            self.assertAlmostEqual(checker.field_metrics(plots[0], dimensions)["mass"], 33/32*(.125 if dimensions == 3 else 1))

    def test_covered_coarse_values_do_not_enter_norm(self):
        plots, _ = fixture(self.root/"run")
        plot = plots[0]
        masks = composite_masks(plot)
        for level, mask in zip(plot.levels, masks):
            level.values[1:, level.valid & ~mask] = 100
        self.assertEqual(checker.field_metrics(plot, 2)["relative_l2"], 0)
        # An error of two over the uncovered L2 area 1/16 gives RMS 2/sqrt(16*33/32).
        plot.levels[2].values[1] = 3
        self.assertAlmostEqual(checker.field_metrics(plot, 2)["relative_l2"], 2/math.sqrt(16*33/32))

    def test_scalar_budget_failure_returns_one(self):
        directory = self.root/"run"
        plots, records = fixture(directory)
        records[1]["relative_l2"] = 2e-8
        write_csv(directory/"observations.csv", list(records[0]), records)
        self.assertEqual(self.evaluate(directory, plots), 1)

    def test_short_science_is_two_but_engineering_reads_artifacts(self):
        directory = self.root/"run"
        plots, _ = fixture(directory, last=8)
        self.assertEqual(self.evaluate(directory, plots), 2)
        self.assertEqual(self.evaluate(directory, plots, True), 0)

    def test_missing_frame_or_nonfinite_scalar_is_two(self):
        directory = self.root/"run"
        plots, records = fixture(directory)
        records[1]["relative_l2"] = float("nan")
        write_csv(directory/"observations.csv", list(records[0]), records)
        self.assertEqual(self.evaluate(directory, plots), 2)
        records[1]["relative_l2"] = 0
        write_csv(directory/"observations.csv", list(records[0]), records)
        lines = (directory/"samples.csv").read_text().splitlines()
        (directory/"samples.csv").write_text("\n".join(lines[:-1])+"\n")
        self.assertEqual(self.evaluate(directory, plots), 2)

    def test_native_scalar_disagreement_is_two(self):
        directory = self.root/"run"
        plots, records = fixture(directory)
        records[50]["mass"] += .01
        write_csv(directory/"observations.csv", list(records[0]), records)
        self.assertEqual(self.evaluate(directory, plots), 2)


if __name__ == "__main__":
    unittest.main()
