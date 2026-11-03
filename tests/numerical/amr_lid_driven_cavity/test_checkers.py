#!/usr/bin/env python3
"""Independent small-field oracles and cavity artifact/exit-contract tests.

The end-to-end synthetic cases replace only the already shared binary reader;
identity, CSV clocks, native layout, composite statistics and all scientific
gates run through the real cavity checker. They do not represent solver data.
"""
import contextlib
import csv
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

import check_cavity as checker
from amr_plotfile import Level, Plot, composite_masks


def level(n, nz=1, bounds=None, depth=1/16):
    bounds = bounds or [((0, 0, 0), (n, n, nz))]
    valid = np.zeros((nz, n, n), dtype=bool)
    for lo, hi in bounds:
        valid[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = True
    values = np.zeros((4, nz, n, n))
    values[0] = 1
    return Level(values, valid, (1/n, 1/n, depth/nz), 0, bounds)


def small_amr():
    return Plot(0, (0, 0, 0), (1, 1, 1/16),
                [level(4), level(8, bounds=[((2, 2, 0), (6, 6, 1))])])


def ghia_field(n):
    """Satisfy the tabulated samples using an independent linear system."""
    result = np.zeros((4, 1, n, n))
    result[0] = 1
    for component, filename in ((1, 'Re1000_x0.5_Ux_Ghia.csv'), (2, 'Re1000_y0.5_Uy_Ghia.csv')):
        reference = np.loadtxt(checker.REFERENCE/filename, delimiter=',', skiprows=1)
        coordinate = (np.arange(n)+.5)/n
        profile = np.interp(coordinate, reference[:, 0], reference[:, 1])
        constraints = np.zeros((len(reference)-2, n))
        for row, (position, _) in enumerate(reference[1:-1]):
            location = position*n-.5
            first = int(math.floor(location))
            constraints[row, first] = first+1-location
            constraints[row, first+1] = location-first
        profile += np.linalg.lstsq(constraints, reference[1:-1, 1]-constraints@profile, rcond=None)[0]
        result[component, 0] = profile[:, None] if component == 1 else profile[None, :]
    return result


class SyntheticRun:
    def __init__(self, directory, layout='coarse', residuals=(0., 0., 0.), dimensions=2):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.n = 256 if layout == 'fine' else 128
        self.levels = 2 if layout == 'amr' else 1
        self.depth = 1/16 if dimensions == 2 else 1/32
        self.nz = [1 if dimensions == 2 else self.n*2**lev//32 for lev in range(self.levels)]
        self.residuals = dict(zip((198, 199, 200), residuals))
        self.fields = [np.repeat(ghia_field(self.n*2**lev), self.nz[lev], axis=1) for lev in range(self.levels)]
        self.boxes = [[((0, 0, 0), (self.n, self.n, self.nz[0]))]]
        if self.levels == 2:
            self.boxes.append([((64, 64, 0), (192, 192, self.nz[1]))])
        self.meta = dict(schema='cavity-v1', descriptor='D2Q9' if dimensions == 2 else 'D3Q27',
            dimensions=dimensions, source_id='a'*40,
            backend='cpu', collision='BGK', periodic='0,0,1', ranks=2, real_digits=53,
            layout=layout, base_cells=self.n, base_z=self.nz[0], length_z=self.depth, levels=self.levels,
            max_grid_size=256, reynolds=1000, u0_lb=.1, u0_phys=1, viscosity=.001,
            rho_reference=1, velocity_reference=.1, full_steps=2000*self.n,
            requested_steps=2000*self.n, comparison_steps=2 if layout == 'fine' else 1, initial_mass=self.depth)
        for lev in range(self.levels):
            self.meta[f'dt_{lev}'] = .1/(self.n*2**lev)
            self.meta[f'tau_{lev}'] = .5+.0003*self.n*2**lev
        self.write_identity()
        (self.directory/'exit.txt').write_text('solver_exit=0\ntee_exit=0\n')
        with (self.directory/'layout.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('level', 'box', 'rank', 'ilo', 'jlo', 'klo', 'ihi', 'jhi', 'khi'))
            for lev, boxes in enumerate(self.boxes):
                for index, (lo, hi) in enumerate(boxes):
                    writer.writerow((lev, index, 0, *lo, *hi))
        with (self.directory/'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('kind', 'level', 'step', 'time', 'coarsest_step', 'phase', 'path'))
            # Deliberately construct the frame schedule independently of checker.frame_steps.
            times = sorted(set(range(0, 201, 20)) | {198, 199, 198-1/1280, 199-1/1280, 200-1/1280})
            for time in times:
                step = round(time*self.n*10)
                for lev in range(self.levels):
                    writer.writerow(('sync', lev, step*2**lev, time, step, 'Sync', f'sync-{step}'))
        self.observations = [dict(step=t*self.n*10, time=t, comparison_begin_time=t-1/1280,
            relative_l2=self.residuals.get(t, 0), mass=self.depth, mass_relative_change=0,
            uz_max=0, rho_z_max=0, velocity_z_max=0) for t in range(1, 201)]
        self.write_observations()

    def write_identity(self):
        (self.directory/'identity.txt').write_text(''.join(f'{key}={value}\n' for key, value in self.meta.items()))

    def write_observations(self):
        with (self.directory/'observations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, checker.OBSERVATIONS)
            writer.writeheader(); writer.writerows(self.observations)

    def read(self, path):
        step = int(Path(path).name.split('-')[1])
        time = step/(10*self.n)
        levels = []
        for lev, source in enumerate(self.fields):
            data = level(self.n*2**lev, nz=self.nz[lev], bounds=self.boxes[lev], depth=self.depth)
            data.values = source.copy()
            data.step = step*2**lev
            if step == 0:
                data.values[1:] = 0
            for t, residual in self.residuals.items():
                if step == t*10*self.n-self.meta['comparison_steps']:
                    data.values[1:] *= 1+residual
            levels.append(data)
        return Plot(time, (0, 0, 0), (1, 1, self.depth), levels)


class CheckerTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get('TMPDIR', Path(__file__).resolve().parent/'build'/'consumer-tests'))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='cavity-checker-', dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, run, reference=None):
        by_directory = {item.directory: item for item in (run, reference) if item is not None}
        def read(path):
            return by_directory[Path(path).parent].read(path)
        with mock.patch.object(checker, 'read_plot', side_effect=read), contextlib.redirect_stdout(io.StringIO()):
            return checker.check(run.directory, reference.directory if reference else None, self.root/'check')

    def test_composite_volume_and_current_l2(self):
        current, previous = small_amr(), small_amr()
        current.levels[0].values[0, :, 1:3, 1:3] = 99
        current.levels[0].values[1] = 1
        current.levels[0].values[1, :, 1:3, 1:3] = 100
        current.levels[1].values[0] = 2
        current.levels[1].values[1] = 2
        previous.levels[1].values[1] = 2
        masks = composite_masks(current)
        self.assertEqual([int(m.sum()) for m in masks], [12, 16])
        self.assertAlmostEqual(checker.statistics(current, masks)['mass'], 1.25/16)
        self.assertAlmostEqual(checker.relative_l2(previous, current, masks), math.sqrt(3/7))

    def test_conservative_momentum_and_lid_normalization(self):
        coarse, fine = level(2), level(4)
        coarse.values[0] = 2; coarse.values[1] = 1.5
        fine.values[0, :, :, ::2] = 1; fine.values[0, :, :, 1::2] = 3
        fine.values[1, :, :, ::2] = 0; fine.values[1, :, :, 1::2] = 2
        a = Plot(200, (0, 0, 0), (1, 1, 1/16), [coarse])
        b = Plot(200, (0, 0, 0), (1, 1, 1/16), [fine])
        self.assertEqual(checker.field_comparison(a, b, [coarse.valid]), (0., 0., 0))
        coarse.values[1] += .01
        self.assertAlmostEqual(checker.field_comparison(a, b, [coarse.valid])[0], .01)

    def test_interface_both_sides_ignores_box_boundaries(self):
        plot = small_amr(); masks = composite_masks(plot)
        selected, faces = checker.interfaces(plot, masks)
        self.assertEqual([int(item.sum()) for item in selected], [8, 12])
        self.assertEqual(len(faces), 8)
        self.assertFalse(selected[0][0, 0, 0])
        plot.levels[1].boxes = [((2, 2, 0), (4, 6, 1)), ((4, 2, 0), (6, 6, 1))]
        split, _ = checker.interfaces(plot, composite_masks(plot))
        for a, b in zip(selected, split):
            np.testing.assert_array_equal(a, b)

    def test_centerline_interpolation_and_covered_exclusion(self):
        plot = small_amr()
        for data in plot.levels:
            n = data.values.shape[-1]
            x, y = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n)
            data.values[1, 0] = 2*x+3*y
            data.values[2, 0] = 4*x-y
        plot.levels[0].values[1:, :, 1:3, 1:3] = 100
        lines = checker.centerlines(plot, composite_masks(plot))
        self.assertEqual(len(lines['ux']), 8)
        for position, value, owner in lines['ux']:
            if owner >= 0: self.assertAlmostEqual(value, 1+3*position)
        for position, value, owner in lines['uy']:
            if owner >= 0: self.assertAlmostEqual(value, 4*position-.5)
        self.assertEqual(lines['ux'][-1], (1., 1., -1))

    def test_d3_uses_vector_deviation_not_component_max(self):
        data = level(2, nz=2, depth=.5)
        data.values[0, 0] = 1.2; data.values[0, 1] = .8
        data.values[1, 0] = .3; data.values[1, 1] = -.3
        data.values[2, 0] = .4; data.values[2, 1] = -.4
        data.values[3] = .1
        result = checker.statistics(Plot(0, (0, 0, 0), (1, 1, .5), [data]), [data.valid])
        self.assertAlmostEqual(result['rho_z_max'], .2)
        self.assertAlmostEqual(result['velocity_z_max'], .5)
        self.assertAlmostEqual(result['uz_max'], .1)

    def test_complete_native_analysis_passes(self):
        run = SyntheticRun(self.root/'run')
        self.assertEqual(self.evaluate(run), 0)
        self.assertEqual(len(checker.rows(self.root/'check/frames.csv')), 16)

    def test_each_ghia_component_is_a_gate(self):
        for component, field in ((1, 'ux'), (2, 'uy')):
            run = SyntheticRun(self.root/field)
            run.fields[0][component] *= 1.2
            self.assertEqual(self.evaluate(run), 1)
            failed = [row['metric'] for row in checker.rows(self.root/'check/gates.csv') if not int(row['passed'])]
            self.assertEqual(failed, [f'ghia_{field}'])

    def test_each_of_last_three_is_required(self):
        for index in range(3):
            residuals = [0., 0., 0.]; residuals[index] = 1e-6
            run = SyntheticRun(self.root/f'run{index}', residuals=residuals)
            self.assertEqual(self.evaluate(run), 1)

    def test_unqualified_fine_preserves_amr_own_gates(self):
        run = SyntheticRun(self.root/'amr', 'amr')
        fine = SyntheticRun(self.root/'fine', 'fine', (0., 0., 1e-6))
        self.assertEqual(self.evaluate(fine), 1)
        self.assertEqual(self.evaluate(run, fine), 2)
        self.assertIn('local_exit=0', (self.root/'check/result.txt').read_text())

    def test_qualified_reference_enables_amr_field_gates(self):
        run = SyntheticRun(self.root/'amr', 'amr')
        fine = SyntheticRun(self.root/'fine', 'fine')
        self.assertEqual(self.evaluate(run, fine), 0)
        # Stay away from both centerlines; identical final pairs retain R=0.
        run.fields[0][1, 0, 2:18, 2:18] += .2
        self.assertEqual(self.evaluate(run, fine), 1)
        self.assertIn('failed=velocity_volume_rms', (self.root/'check/result.txt').read_text())
        self.assertTrue(all(int(row['passed']) for row in checker.rows(self.root/'check/gates.csv')))
        comparison = {row['metric']: row for row in checker.rows(self.root/'check/comparison.csv')}
        self.assertEqual(comparison['velocity_interface_peak']['passed'], '1')
        run.fields[0][1, 0, 2:18, 2:18] -= .2
        run.fields[0][1, 0, 40:48, 31] += .1
        self.assertEqual(self.evaluate(run, fine), 1)
        self.assertIn('failed=velocity_interface_peak', (self.root/'check/result.txt').read_text())
        comparison = {row['metric']: row for row in checker.rows(self.root/'check/comparison.csv')}
        self.assertEqual(comparison['velocity_volume_rms']['passed'], '1')
        self.assertTrue(all(int(row['passed']) for row in checker.rows(self.root/'check/gates.csv')))

    def test_d3_gate_checks_nonterminal_observations(self):
        run = SyntheticRun(self.root/'d3', dimensions=3)
        self.assertEqual(self.evaluate(run), 0)
        for key in ('uz_max', 'rho_z_max', 'velocity_z_max'):
            run.observations[6][key] = 2e-10
            run.write_observations()
            self.assertEqual(self.evaluate(run), 1)
            failed = [row['metric'] for row in checker.rows(self.root/'check/gates.csv') if not int(row['passed'])]
            self.assertEqual(failed, [key])
            run.observations[6][key] = 0

    def test_failed_fine_process_is_not_a_reference(self):
        run = SyntheticRun(self.root/'amr', 'amr')
        fine = SyntheticRun(self.root/'fine', 'fine')
        for solver, tee in ((1, 0), (0, 1)):
            (fine.directory/'exit.txt').write_text(f'solver_exit={solver}\ntee_exit={tee}\n')
            self.assertEqual(self.evaluate(fine), 2)
            self.assertEqual(self.evaluate(run, fine), 2)
            self.assertIn('local_exit=0', (self.root/'check/result.txt').read_text())
            self.assertTrue(all(int(row['passed']) for row in checker.rows(self.root/'check/gates.csv')))
        (fine.directory/'exit.txt').unlink()
        self.assertEqual(self.evaluate(fine), 2)
        # Diagnostic rendering may inspect artifacts from an unfinished run.
        checker.CavityRun(fine.directory, complete=False)

    def test_missing_reference_preserves_local_results(self):
        run = SyntheticRun(self.root/'run', 'amr')
        self.assertEqual(self.evaluate(run), 2)
        self.assertTrue((self.root/'check/ghia.csv').is_file())

    def test_bad_identity_and_short_run_are_invalid(self):
        run = SyntheticRun(self.root/'run')
        run.meta['velocity_reference'] = 1; run.write_identity()
        self.assertEqual(self.evaluate(run), 2)
        run.meta['velocity_reference'] = .1; run.meta['requested_steps'] = 1280; run.write_identity()
        self.assertEqual(self.evaluate(run), 2)

    def test_wrong_interval_and_nonfinite_scalar_are_invalid(self):
        run = SyntheticRun(self.root/'run')
        run.observations[-1]['comparison_begin_time'] = 199
        run.write_observations(); self.assertEqual(self.evaluate(run), 2)
        run.observations[-1]['comparison_begin_time'] = 200-1/1280
        run.observations[17]['mass'] = float('nan')
        run.write_observations(); self.assertEqual(self.evaluate(run), 2)

    def test_scalar_residual_must_match_native_pair(self):
        run = SyntheticRun(self.root/'run', residuals=(0., 0., 1e-6))
        run.observations[-1]['relative_l2'] = 0
        run.write_observations(); self.assertEqual(self.evaluate(run), 2)

    def test_mass_gate_uses_all_observations(self):
        run = SyntheticRun(self.root/'run')
        run.observations[6]['mass'] *= 1+2e-8
        run.observations[6]['mass_relative_change'] = 2e-8
        run.write_observations(); self.assertEqual(self.evaluate(run), 1)

    def test_missing_native_file_is_invalid(self):
        run = SyntheticRun(self.root/'run')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.check(run.directory, None, self.root/'check'), 2)

    def test_accumulated_time_roundoff_does_not_allow_one_step_shift(self):
        self.assertTrue(checker.time_matches(200+1e-9, 200, 512000, 1/2560))
        self.assertFalse(checker.time_matches(200+1/2560, 200, 512000, 1/2560))


if __name__ == '__main__':
    unittest.main()
