#!/usr/bin/env python3
"""Small independent sequences for the completed-export observation comparison."""
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compare_dynamic_observations as compare
from amr_plotfile import ArtifactError, identity, rows
from amr_validation import write_csv


def metadata(directory, meta):
    text = "".join(f"{key}={value}\n" for key, value in meta.items())
    (directory / "identity.txt").write_text(text)
    (directory / "analysis/identity.txt").write_text(text)


def fixture(root, label, cd=(1, 2, 3), cl=(3, 1, -1), legacy=False,
            dimensions=2, variant="single", reverse=False):
    directory = root / label
    analysis = directory / "analysis"
    analysis.mkdir(parents=True)
    full, owner = (800000, 0) if legacy else (200000, 2)
    meta = dict(schema="amr-re100-v1" if legacy else "dynamic-cylinder-v1",
                descriptor="D2Q9" if dimensions == 2 else "D3Q27", dimensions=dimensions,
                source_id=("a" if legacy else "b") * 40, backend="cpu",
                collision="BGK", real_digits=53, ranks=32 if legacy else 16,
                scheme="YMS", layout="fine" if legacy else "dynamic",
                periodic="0,0,1", initialization="uniform-equilibrium-then-Guo",
                reynolds=100, viscosity=.001, u0_lb=.035, u0_phys=1,
                rho_reference=1, velocity_reference=.035, length_z=.01, diameter_lb=80,
                span_lb=8 if legacy or dimensions == 3 else 1,
                force_denominator=.049 * (8 if dimensions == 3 else 1),
                full_steps=full, requested_steps=full, dt_0=35 / full)
    if legacy:
        meta.update(boundary_id=4, object_id=0, owner_level=owner)
    else:
        meta.update(variant=variant, max_level=owner, dt_2=35 / 800000,
                    force_units="allowed-finest-lattice")
    metadata(directory, meta)
    (directory / "exit.txt").write_text("solver_exit=0\ntee_exit=0\n")
    (analysis / "result.txt").write_text(
        "exit=2\nscope=scientific\nerror=no qualified continuous 3+5 window\n" if legacy
        else "exit=0\nscope=engineering-and-data-only\n")
    keys = ((4, 0),) if legacy or variant == "single" else ((4, 0), (5, 1))
    if not legacy:
        write_csv(directory / "boundaries.csv", ("boundary_id", "object_id"),
                  (dict(boundary_id=bid, object_id=oid) for bid, oid in keys))
    forces = []
    for index, (drag, lift) in enumerate(zip(cd, cl), 1):
        for bid, oid in (keys[::-1] if reverse else keys):
            row = dict(time=(index - .5) * .5, Cd=drag + 10 * oid, Cl=lift - 5 * oid, Cz=0)
            if legacy:
                row["t_star"] = row["time"] * 10
            else:
                row.update(boundary_id=bid, object_id=oid, level=2, population_step=index)
            forces.append(row)
    write_csv(analysis / "force-history.csv", list(forces[0]), forces)
    fields = [dict(step=999, time=time, fluid_center_mass=10 + time / 7,
                   phi_in=1, phi_out=1.2, flux_imbalance=.2,
                   uz_max=0, rho_z_max=0, velocity_z_max=0) for time in compare.TIMES]
    write_csv(analysis / "fields.csv", list(fields[0]), fields)
    run = compare.load_observation(label, "pair", directory)
    # Eligibility is checked above; the reduction oracle uses only three synthetic midpoints.
    return replace(run, force_dt=.5, sample_count=len(cd))


class DynamicObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_legacy_and_dynamic_known_difference_and_exports(self):
        old = fixture(self.root, "legacy", legacy=True)
        new = fixture(self.root, "dynamic", cd=(2, 4, 6), cl=(2, 1, 0))
        self.assertEqual((old.meta["span_lb"], new.meta["span_lb"]), ("8", "1"))
        self.assertEqual(old.checker_exit, 2)
        self.assertEqual(next(compare.iter_force_records(old)).cylinder, (4, 0))
        manifest = self.root / "manifest.csv"
        write_csv(manifest, ("label", "group", "run"),
                  [dict(label=r.label, group=r.group, run=str(r.directory)) for r in (old, new)])
        output = self.root / "comparison"
        with patch.object(compare, "load_observation", side_effect=(old, new)):
            compare.compare(manifest, output)
        differences = {row["component"]: row for row in rows(output / "force-differences.csv")}
        self.assertAlmostEqual(float(differences["Cd"]["absolute_rms"]), math.sqrt(14 / 3))
        self.assertAlmostEqual(float(differences["Cl"]["absolute_rms"]), math.sqrt(2 / 3))
        self.assertEqual(float(differences["Cd"]["maximum_absolute_difference"]), 3)
        self.assertEqual(float(differences["Cl"]["maximum_absolute_difference"]), 1)
        stats = {(r["label"], r["component"]): r for r in rows(output / "force-statistics.csv")}
        self.assertEqual(float(stats["legacy", "Cd"]["mean"]), 2)
        self.assertEqual(float(stats["dynamic", "Cd"]["mean"]), 4)
        self.assertEqual(float(stats["legacy", "Cl"]["minimum"]), -1)
        self.assertEqual(len(rows(output / "field-observations.csv")), 12)
        self.assertEqual(rows(output / "sources.csv")[0]["checker_exit"], "2")

    def test_two_cylinders_are_matched_by_identity(self):
        a = fixture(self.root, "a", variant="tandem-a")
        b = fixture(self.root, "b", cd=(2, 4, 6), cl=(2, 1, 0), variant="tandem-b", reverse=True)
        stats, deltas = compare.compare_forces([a, b])
        self.assertEqual(len(deltas), 4)
        for row in deltas:
            expected = math.sqrt((14 if row["component"] == "Cd" else 2) / 3)
            self.assertAlmostEqual(row["absolute_rms"], expected)
        second = next(row for row in stats if row["label"] == "a" and
                      row["object_id"] == 1 and row["component"] == "Cd")
        self.assertEqual(second["mean"], 12)

    def test_shifted_midpoint_and_missing_tail_are_rejected(self):
        a, b = fixture(self.root, "a"), fixture(self.root, "b")
        path = b.analysis / "force-history.csv"
        records = rows(path)
        records[1]["time"] = ".76"
        write_csv(path, list(records[0]), records)
        with self.assertRaises(ArtifactError):
            compare.compare_forces([a, b])
        records[1]["time"] = ".75"
        write_csv(path, list(records[0]), records[:-1])
        with self.assertRaises(ArtifactError):
            compare.compare_forces([a, b])

    def test_descriptor_and_geometry_mismatch_are_rejected(self):
        d2, d3 = fixture(self.root, "d2"), fixture(self.root, "d3", dimensions=3)
        tandem = fixture(self.root, "tandem", variant="tandem-a")
        for other in (d3, tandem):
            with self.assertRaises(ArtifactError):
                compare.compatible([d2, other])

    def test_wrong_force_denominator_is_rejected(self):
        run = fixture(self.root, "wrong")
        meta = dict(run.meta, force_denominator=".098")
        metadata(run.directory, meta)
        with self.assertRaises(ArtifactError):
            compare.load_observation("wrong", "pair", run.directory)

    def test_incomplete_gpu_horizon_or_unsuccessful_dynamic_check_is_rejected(self):
        run = fixture(self.root, "incomplete")
        meta = dict(run.meta, backend="cuda", requested_steps="199999")
        metadata(run.directory, meta)
        with self.assertRaises(ArtifactError):
            compare.load_observation("incomplete", "pair", run.directory)
        meta["requested_steps"] = "200000"
        metadata(run.directory, meta)
        (run.analysis / "result.txt").write_text("exit=2\nscope=engineering-and-data-only\n")
        with self.assertRaises(ArtifactError):
            compare.load_observation("incomplete", "pair", run.directory)

    def test_field_times_are_physical_and_flux_difference_stays_absolute(self):
        run = fixture(self.root, "fields")
        fields = compare.field_observations(run)
        self.assertEqual([row["time"] for row in fields], [0, 7, 14, 21, 28, 35])
        self.assertAlmostEqual(fields[1]["relative_mass_drift"], .1)
        self.assertEqual(fields[1]["flux_imbalance"], .2)
        original = rows(run.analysis / "fields.csv")
        write_csv(run.analysis / "fields.csv", list(original[0]), original[:2] + original[3:])
        with self.assertRaises(ArtifactError):
            compare.field_observations(run)


if __name__ == "__main__":
    unittest.main()
