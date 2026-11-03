#!/usr/bin/env python3
import contextlib
import copy
import io
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import check_re100 as checker
from re100_artifacts import Re100Run
from re100_cycles import Sample, closed_integral, cycle_records, select_window, summarize
from amr_plotfile import Level, Plot, ArtifactError
from amr_validation import write_csv
from synthetic_amr import write_run, read_from


def short_fixture(directory):
    h, dt = .0025, .0000875
    denominator = .5*.035**2*40
    meta = dict(schema="amr-re100-v1", descriptor="D2Q9", dimensions=2, backend="cpu",
                collision="BGK", real_digits=53, source_id="a"*40, ranks=32,
                levels=1, owner_level=0, max_grid_size=64, periodic="0,0,1",
                initialization="uniform-equilibrium-then-Guo", layout="coarse", scheme="YMS",
                reynolds=100, viscosity=.001, u0_lb=.035, u0_phys=1, rho_reference=1,
                velocity_reference=.035, length_z=.01, full_steps=400000, requested_steps=8,
                dt_0=dt, tau_0=.542, diameter_lb=40, span_lb=4, force_denominator=denominator,
                boundary_id=4, object_id=0)
    plots = {}
    for step in (0, 8):
        values = np.zeros((4, 1, 165, 881)); values[0:2] = 1
        level = Level(values, np.ones((1, 165, 881), bool), (h, h, .01), step, [((0, 0, 0), (881, 165, 1))])
        plots[step] = Plot(step*dt, (-h/2, -h/2, 0), (2.2+h/2, .41+h/2, .01), [level])
    write_run(directory, meta, plots)
    force = []
    for step in range(1, 9):
        for kind, identity in (("boundary", 4), ("object", 0)):
            force.append(dict(kind=kind, id=identity, level=0, population_step=step, coarsest_step=step,
                              begin_time=(step-1)*dt, end_time=step*dt, sample_time=(step-.5)*dt,
                              Fx_lb=3*denominator, Fy_lb=0, Fz_lb=0, Cd=3, Cl=0, Cz=0))
    write_csv(directory/"forces.csv", list(force[0]), force)
    return plots, force


class Re100Checks(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get("TMPDIR", Path(__file__).resolve().parent/"build"/"consumer-tests"))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch, prefix="re100-checker-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_time_integral_closes_nonuniform_sample_endpoints(self):
        times = np.array([0., .1, .4, 1.])
        self.assertAlmostEqual(closed_integral(times, 2*times+3, .05, .7), (.7**2+3*.7)-(.05**2+3*.05))

    def test_known_period_means_amplitudes_and_legacy_midrange(self):
        samples = []
        for index, t in enumerate(np.arange(250, 290, .0073)):
            phase = 2*math.pi*(t-250)/3.3
            samples.append(Sample(index+1, t/10, t, 3.2+.03*math.cos(2*phase+.3), -.01+math.sin(phase)))
        cycles = cycle_records(samples)
        selected = select_window(cycles)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 0)
        values = summarize(samples, cycles, 0)
        for name, expected in (("St", 1/3.3), ("Cd_mean", 3.2), ("Cl_mean", -.01),
                               ("Cd_amplitude", .03), ("Cl_amplitude", 1),
                               ("Cd_peak_to_peak", .06), ("Cl_peak_to_peak", 2),
                               ("Cd_midrange", 3.2)):
            self.assertAlmostEqual(values[name], expected, delta=2e-6)
        # A non-sinusoidal cycle has different time mean and extrema midpoint.
        skewed = [Sample(s.step, s.force_time, s.t_star, s.cd, math.sin(2*math.pi*(s.t_star-250)/3.3)+
                         .2*math.cos(4*math.pi*(s.t_star-250)/3.3)) for s in samples]
        skew_cycles = cycle_records(skewed)
        skew_stats = summarize(skewed, skew_cycles, select_window(skew_cycles)[0])
        self.assertGreater(abs(skew_stats["Cl_mean"]-skew_stats["Cl_midrange"]), .1)

    def test_cumulative_drift_rejected_and_first_qualified_window_retained(self):
        template = dict(period_t_star=3.3, Cd_amplitude=.03, Cl_amplitude=1., Cd_mean=3.2,
                        Cl_mean=0., Cl_peak_to_peak=2., samples=300)
        cycles = [dict(template, Cd_mean=3.2*(1+.004*i)) for i in range(12)]
        self.assertIsNone(select_window(cycles))
        cycles = [template.copy() for _ in range(10)]
        cycles[0]["Cd_amplitude"] = .05
        self.assertEqual(select_window(cycles)[0], 1)
        cycles[1]["samples"] = 199
        self.assertEqual(select_window(cycles)[0], 2)
        self.assertIsNone(select_window(cycles[:7]))
        for row in cycles:
            row["Cl_amplitude"] = 0
        self.assertIsNone(select_window(cycles))

    def evaluate(self, directory, plots, engineering=False):
        args = ["check", "--run", str(directory), "--output", str(self.root/"analysis")]
        if engineering:
            args.append("--engineering")
        with patch("amr_validation.read_plot", side_effect=read_from({directory: plots})), patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            return checker.main()

    def test_short_engineering_readback_and_incomplete_science(self):
        directory = self.root/"run"
        plots, _ = short_fixture(directory)
        self.assertEqual(self.evaluate(directory, plots, True), 0)
        self.assertEqual(self.evaluate(directory, plots), 2)

    def test_force_identity_scale_or_missing_sample_is_two(self):
        directory = self.root/"run"
        plots, force = short_fixture(directory)
        original = copy.deepcopy(force)
        for kind in ("missing", "scale", "time", "nonfinite"):
            force = copy.deepcopy(original)
            if kind == "missing":
                force.pop()
            elif kind == "scale":
                force[0]["Fx_lb"] *= 2
            elif kind == "time":
                force[0]["sample_time"] += .001
            else:
                force[0]["Cd"] = float("nan")
            write_csv(directory/"forces.csv", list(original[0]), force)
            self.assertEqual(self.evaluate(directory, plots, True), 2)

    def test_external_failure_does_not_disqualify_matching_reference(self):
        metrics = {row["metric"]: float(row["value"]) for row in checker.rows(checker.REFERENCE)}
        metrics.update(Cd_mean=3.2, Cl_mean=0.)
        metrics["Cd_peak_to_peak"] *= 1.3
        identity = dict(source_id="a"*40, descriptor="D2Q9", backend="cpu", scheme="YMS")
        run = SimpleNamespace(layout="amr", id=identity, metrics=metrics, quasi_pass=True,
                              export=lambda output: output.mkdir(parents=True, exist_ok=True))
        reference = SimpleNamespace(layout="fine", id=identity.copy(), metrics=metrics)
        with patch("check_re100.Re100Run", return_value=reference):
            status, conclusions = checker.check(run, self.root/"reference", self.root/"comparison")
        self.assertEqual(status, 1)
        self.assertEqual(conclusions["literature"], 1)
        self.assertEqual(conclusions["amr_comparison"], 0)
        reference.metrics = None
        with patch("check_re100.Re100Run", return_value=reference), self.assertRaises(ArtifactError):
            checker.check(run, self.root/"reference", self.root/"invalid-reference")
        with self.assertRaises(ArtifactError):
            checker.check(run, None, self.root/"missing-reference")


if __name__ == "__main__":
    unittest.main()
