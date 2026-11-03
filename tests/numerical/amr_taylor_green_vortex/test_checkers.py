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
import check_tgv as checker
from amr_plotfile import ArtifactError, Plot, Level, composite_masks
from amr_validation import quasi_metrics
from synthetic_amr import write_run, read_from
from tgv_metrics import exact, errors, interface_neighborhood, observed_order
from summarize_convergence import same_family


def field(n, nz=1, central=False):
    bounds = ((n//4, n//4, 0), (3*n//4, 3*n//4, nz)) if central else ((0, 0, 0), (n, n, nz))
    low, high = bounds
    valid = np.zeros((nz, n, n), bool)
    valid[:, low[1]:high[1], low[0]:high[0]] = True
    values = np.zeros((4, nz, n, n)); values[0] = 1
    return Level(values, valid, (1/n, 1/n, .2/nz), 0, [bounds])


def fixture(directory, amr=False, short=False, offset=0):
    full = 192 if amr else 384
    last = 8 if short else full
    dt = 1/960 if amr else 1/1920
    meta = dict(schema="amr-tgv-v1", descriptor="D2Q9", dimensions=2, backend="cpu", collision="BGK",
                real_digits=53, source_id="b"*40, ranks=4, max_grid_size=16, coarse_cells=20,
                layout="amr" if amr else "uniform", levels=2 if amr else 1, periodic="1,1,1",
                initialization="tgv-equilibrium", reynolds=10, viscosity=.1, u0_lb=1/48, u0_phys=1,
                rho_reference=1, velocity_reference=1/48, length_z=.2, full_steps=full, requested_steps=last)
    for lev in range(meta["levels"]):
        meta[f"dt_{lev}"] = dt/2**lev
        meta[f"tau_{lev}"] = .625 if amr and lev == 0 else .75
    plots = {}
    for step in ([0, 8] if short else [0, full//4, full//2, full]):
        time = step*dt
        levels = [field(20), field(40, central=True)] if amr else [field(40)]
        for lev, level in enumerate(levels):
            n = level.valid.shape[-1]
            x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
            a = math.exp(-.8*math.pi**2*time)
            level.values[0, 0] = 1-(1/3072)*(np.cos(4*np.pi*x)+np.cos(4*np.pi*y))*a*a
            level.values[1, 0] = -np.sin(2*np.pi*y)*np.cos(2*np.pi*x)*a + (offset if step else 0)
            level.values[2, 0] = np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*a
            level.step = step*2**lev
        plots[step] = Plot(time, (0, 0, 0), (1, 1, .2), levels)
    write_run(directory, meta, plots)
    return plots


class TgvChecks(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("TMPDIR", Path(__file__).resolve().parent/"build"/"consumer-tests"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch, prefix="tgv-checker-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, directory, runs, reference=None, engineering=False):
        args = ["check", "--run", str(directory), "--output", str(self.root/"analysis")]
        if reference:
            args += ["--reference", str(reference)]
        if engineering:
            args.append("--engineering")
        with patch("amr_validation.read_plot", side_effect=read_from(runs)), patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            return checker.main()

    def test_cell_center_analytic_decay_and_pressure(self):
        level = field(4)
        values = exact(level, .1, .02)
        decay = math.exp(-.08*math.pi**2)
        self.assertAlmostEqual(values[1, 0, 0, 0], -.5*decay)
        self.assertAlmostEqual(values[2, 0, 0, 0], .5*decay)
        self.assertAlmostEqual(values[0, 0, 0, 0], 1.)
        level = field(2)
        self.assertAlmostEqual(exact(level, 0, .02)[0, 0, 0, 0], 1+1.5*.02**2)

    def test_uniform_and_matched_amr_pass(self):
        fine, amr = self.root/"fine", self.root/"amr"
        runs = {fine: fixture(fine), amr: fixture(amr, amr=True)}
        self.assertEqual(self.evaluate(fine, runs), 0)
        self.assertEqual(self.evaluate(amr, runs, fine), 0)

    def test_failure_and_unqualified_reference(self):
        fine, amr = self.root/"fine", self.root/"amr"
        runs = {fine: fixture(fine, offset=.2), amr: fixture(amr, amr=True)}
        self.assertEqual(self.evaluate(fine, runs), 1)
        self.assertEqual(self.evaluate(amr, runs, fine), 2)
        self.assertEqual(self.evaluate(amr, runs), 2)

    def test_short_and_missing_sample_are_two(self):
        run = self.root/"short"
        runs = {run: fixture(run, amr=True, short=True)}
        self.assertEqual(self.evaluate(run, runs), 2)
        self.assertEqual(self.evaluate(run, runs, engineering=True), 0)
        lines = (run/"samples.csv").read_text().splitlines()
        (run/"samples.csv").write_text("\n".join(lines[:-1])+"\n")
        self.assertEqual(self.evaluate(run, runs, engineering=True), 2)

    def test_covered_coarse_does_not_enter_error(self):
        plot = Plot(.1, (0, 0, 0), (1, 1, .2), [field(8), field(16, central=True)])
        for level in plot.levels:
            level.values = exact(level, .1, .02).copy()
        masks = composite_masks(plot)
        plot.levels[0].values[1:, ~masks[0]] = 100
        e2, peak = errors(plot, .02, 2)
        self.assertEqual((e2, peak), (0., 0.))

    def test_interface_distance_and_box_seams(self):
        plot = Plot(0, (0, 0, 0), (1, 1, .2), [field(8), field(16, central=True)])
        masks = composite_masks(plot)
        near = interface_neighborhood(plot, masks)
        self.assertTrue(near[1][0, 4, 4])  # 1/32 from each center-patch face.
        self.assertFalse(near[1][0, 7, 7])
        plot.levels[0].boxes = [((0, 0, 0), (4, 8, 1)), ((4, 0, 0), (8, 8, 1))]
        np.testing.assert_array_equal(near[0], interface_neighborhood(plot, masks)[0])

    def test_quasi_normalization(self):
        level = field(4, nz=2)
        level.values[0, 0] = 1.99; level.values[0, 1] = 2.01
        level.values[1, 0] = .9; level.values[1, 1] = 1.1
        level.values[3] = .001
        plot = Plot(0, (0, 0, 0), (1, 1, .2), [level])
        metrics = quasi_metrics(plot, [level.valid])
        self.assertAlmostEqual(metrics["rho_z_max"], .005)
        self.assertAlmostEqual(metrics["velocity_z_max"], .1)
        self.assertEqual(metrics["uz_max"], .001)

    def test_order_requires_matching_refinement(self):
        first = Plot(0, (0, 0, 0), (1, 1, .2), [field(8), field(16, central=True)])
        second = Plot(0, (0, 0, 0), (1, 1, .2), [field(16), field(32, central=True)])
        self.assertTrue(same_family(first, second))
        second.levels[1].boxes = [((6, 8, 0), (24, 24, 1))]
        self.assertFalse(same_family(first, second))
        self.assertEqual(observed_order(4, 1), 2)
        self.assertEqual(observed_order(1, 2), -1)
        self.assertIsNone(observed_order(0, 1))


if __name__ == "__main__":
    unittest.main()
