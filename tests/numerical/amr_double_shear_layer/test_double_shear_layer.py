#!/usr/bin/env python3
"""Independent polynomial/coverage/normalization checks of DSL postprocessing."""
from pathlib import Path
import sys
import unittest
import tempfile
import csv
from itertools import product
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import Plot, Level, composite_masks, ArtifactError
from dsl_vorticity import CurlPlan, quartic_weights
from analyze_double_shear_layer import requested_rows, statistics, LAYOUTS, DslRun, COLUMNS, DERIVATIVE
import check_double_shear_layer as checker
import summarize_double_shear_layer as summary
from native_planes import dump_planes
from render_double_shear_layer import load_planes


def fixture(dimensions, layout, base=16):
    depth = 4/base if dimensions == 3 else 1/16
    count = 2 if layout == 'baseline' else 3
    levels = []
    for lev in range(count):
        n = base*2**lev; nz = 4*2**lev if dimensions == 3 else 1
        active = requested_rows(n, layout, lev)
        mask = np.broadcast_to(active[None, :, None], (nz, n, n)).copy()
        boxes = []
        edges = np.flatnonzero(np.diff(np.pad(active.astype(int), (1, 1))))
        for lo, hi in zip(edges[::2], edges[1::2]):
            boxes.append(((0, int(lo), 0), (n, int(hi), nz)))
        values = np.broadcast_to(np.array([1., 0., 0., 0.])[:, None, None, None], (4, nz, n, n))
        levels.append(Level(values, mask, (1/n, 1/n, depth/nz), 0, boxes))
    return Plot(0, (0, 0, 0), (1, 1, depth), levels)


class DslChecks(unittest.TestCase):
    def test_covered_values_do_not_enter_curl_or_integrals(self):
        plot = fixture(2, 'baseline')
        for level in plot.levels:
            nz, ny, nx = level.valid.shape
            y = (np.arange(ny)+.5)*level.spacing[1]
            x = (np.arange(nx)+.5)*level.spacing[0]
            level.values = np.ones((4, nz, ny, nx))
            level.values[1] = np.sin(2*np.pi*y)[None, :, None]
            level.values[2] = .05*np.sin(2*np.pi*x)[None, None, :]
            level.values[3] = 0
        masks = composite_masks(plot); plan = CurlPlan(plot, masks, 2)
        original = plan.curl(plot); first = statistics(plot, masks, original, 2)
        plot.levels[0].values[:, ~masks[0]] = 1e8
        contaminated = plan.curl(plot); second = statistics(plot, masks, contaminated, 2)
        for before, after, mask in zip(original, contaminated, masks):
            np.testing.assert_array_equal(before[:, mask], after[:, mask])
        self.assertEqual(first, second)

    def test_energy_normalization_and_density_independence(self):
        for dimensions in (2, 3):
            plot = fixture(dimensions, 'baseline')
            velocity = (1., 2., 3. if dimensions == 3 else 0.)
            for level in plot.levels:
                level.values = np.ones((4, *level.valid.shape))
                for d in range(3):
                    level.values[d+1] = velocity[d]
            masks = composite_masks(plot); plan = CurlPlan(plot, masks, dimensions)
            first = statistics(plot, masks, plan.curl(plot), dimensions)
            self.assertEqual(first['energy'], sum(v*v for v in velocity)/2)
            self.assertLess(first['enstrophy'], 1e-18)
            for level in plot.levels:
                level.values[0] = 17
            self.assertEqual(statistics(plot, masks, plan.curl(plot), dimensions)['energy'], first['energy'])

    def test_translated_interface_curl(self):
        for dimensions in (2, 3):
            for lev in (0, 1):
                plot = fixture(dimensions, 'baseline')
                target = plot.levels[lev]
                i = 13 if lev == 0 else 27
                j = target.valid.shape[1]//2-(1 if lev == 0 else 0)
                k = target.valid.shape[0]-1
                center = (np.array([i, j, k])+.5)*np.array(target.spacing)
                for level in plot.levels:
                    coordinate = []
                    for d in range(3):
                        delta = (np.arange(level.valid.shape[2-d])+.5)*level.spacing[d]-center[d]
                        delta[delta >= plot.high[d]/2] -= plot.high[d]
                        delta[delta < -plot.high[d]/2] += plot.high[d]
                        shape = [1, 1, 1]; shape[2-d] = len(delta)
                        coordinate.append(delta.reshape(shape))
                    x, y, z = coordinate
                    level.values = np.ones((4, *level.valid.shape))
                    level.values[1] = 2*y+x*y+3*x*z
                    level.values[2] = 3*x+y*z
                    level.values[3] = 5*y+2*x-z if dimensions == 3 else 0
                masks = composite_masks(plot)
                curl = CurlPlan(plot, masks, dimensions).curl(plot)[lev][:, k, j, i]
                np.testing.assert_allclose(curl, (5, -2, 1) if dimensions == 3 else (1,), atol=3e-10, rtol=0)

    def test_consumer_exit_classes_and_shared_wrong_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            amr, reference = root/'amr', root/'reference'
            for directory, layout, n, levels in ((amr, 'baseline', 128, 2), (reference, 'uniform', 256, 1)):
                synthetic_curve(directory, layout, n, levels)
            def evaluate(output=None):
                with patch.object(sys, 'argv', ['checker', '--amr', str(amr), '--reference', str(reference),
                                               '--output', str(output or root/'result')]):
                    return checker.main()
            self.assertEqual(evaluate(), 0)
            path = amr/'analysis/curves.csv'
            with path.open(newline='') as stream:
                data = list(csv.DictReader(stream))
            data[20]['energy'] = str(float(data[20]['energy'])*1.1)
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, COLUMNS); writer.writeheader(); writer.writerows(data)
            self.assertEqual(evaluate(), 1)
            for directory in (amr, reference):
                for path in (directory/'identity.txt', directory/'analysis/identity.txt'):
                    path.write_text(path.read_text().replace('reynolds=10000', 'reynolds=100'))
            self.assertEqual(evaluate(), 2)
            bad_output = root/'plain-file'; bad_output.write_text('not a directory')
            self.assertEqual(evaluate(bad_output), 2)

    def test_incomplete_source_preserves_valid_prefix_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); synthetic_curve(root, 'baseline', 128, 2)
            path = root/'samples.csv'
            lines = path.read_text().splitlines(); path.write_text('\n'.join(lines[:3])+'\n')
            run = DslRun(root)
            self.assertEqual(set(run.samples), {(0, 0), (0, 1)})
            with self.assertRaises(ArtifactError):
                run.require_complete()
            with self.assertRaisesRegex(ArtifactError, 'missing DSL frame 1'):
                run.plot(1)

    def test_summary_retains_failed_prefix_and_other_complete_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); partial = root/'partial'; complete = root/'complete'
            synthetic_curve(partial, 'baseline', 128, 2)
            synthetic_curve(complete, 'uniform', 256, 1)
            path = partial/'analysis/curves.csv'
            path.write_text('\n'.join(path.read_text().splitlines()[:4])+'\n')
            (partial/'analysis/analysis-result.txt').write_text('exit=2\nerror=frame=3 nonfinite data\n')
            output = root/'summary'
            self.assertEqual(summary.summarize([partial, complete], output, False), 2)
            with (output/'curves.csv').open(newline='') as stream:
                values = list(csv.DictReader(stream))
            self.assertEqual(sum(row['layout'] == 'baseline' for row in values), 3)
            self.assertEqual(sum(row['layout'] == 'uniform' for row in values), 41)
            with (output/'inputs.csv').open(newline='') as stream:
                records = list(csv.DictReader(stream))
            failed = next(row for row in records if row['case'] == 'baseline')
            self.assertEqual((failed['valid_frames'], failed['first_unavailable_frame']), ('3', '3'))
            self.assertIn('frame=3', failed['analysis_details'])
            self.assertTrue(any(row['error'] == 'missing configuration' for row in records))
            with self.assertRaises(ArtifactError):
                checker.curve(partial)

    def test_fourth_order_interface_moments_all_actual_layouts(self):
        for dimensions in (2, 3):
            powers = [q for q in product(range(5), repeat=dimensions) if sum(q) <= 4]
            for layout in LAYOUTS:
                plot = fixture(dimensions, layout, base=128)
                masks = composite_masks(plot)
                plan = CurlPlan(plot, masks, dimensions)
                expected = {(lev, row, px, pz)
                            for lev, (level, mask) in enumerate(zip(plot.levels, masks))
                            for row in range(mask.shape[1]) if mask[0, row, 0]
                            and any(not mask[0, (row+shift) % mask.shape[1], 0] for shift in (-2, -1, 1, 2))
                            for px in range(2**lev) for pz in range(2**lev if dimensions == 3 else 1)}
                self.assertEqual({(fit.level, fit.row, fit.phase_x, fit.phase_z) for fit in plan.fits}, expected)
                for fit in plan.fits:
                    target = plot.levels[fit.level]
                    center = (np.array([fit.phase_x, fit.row, fit.phase_z])+.5)*np.array(target.spacing)
                    scale = max(plot.levels[lev].spacing[1] for lev, z, y, x in fit.sources)
                    coordinates = []
                    self.assertEqual(len(fit.sources), len(set(fit.sources)))
                    for lev, z, y, x in fit.sources:
                        source = plot.levels[lev]
                        self.assertTrue(masks[lev][z, y, x])
                        point = (np.array([x, y, z])+.5)*np.array(source.spacing)
                        coordinates.append([min((point[d]-center[d]+shift*plot.high[d] for shift in (-1, 0, 1)),
                                                key=lambda value: (abs(value), value))/scale for d in range(dimensions)])
                    q = np.array(coordinates)
                    actual = [np.prod(q**power, axis=1) @ (scale*fit.weights) for power in powers]
                    expected = [int(power[1] == 1 and sum(power) == 1) for power in powers]
                    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=0)

    def test_fourth_order_curl_convergence_on_periodic_amr(self):
        for layout in ('baseline', 'p3'):
            errors = []
            for base in (32, 64, 128):
                plot = fixture(2, layout, base)
                exact = []
                for level in plot.levels:
                    nz, ny, nx = level.valid.shape
                    x = 2*np.pi*(np.arange(nx)+.5)/nx
                    y = 2*np.pi*(np.arange(ny)+.5)/ny
                    level.values = np.zeros((4, nz, ny, nx)); level.values[0] = 1
                    level.values[1] = np.sin(y)[None, :, None]*np.cos(x)[None, None, :]
                    level.values[2] = np.cos(x)[None, None, :]*np.sin(y)[None, :, None]
                    exact.append(-2*np.pi*(np.sin(x)[None, None, :]*np.sin(y)[None, :, None]+
                                           np.cos(x)[None, None, :]*np.cos(y)[None, :, None]))
                masks = composite_masks(plot)
                curls = CurlPlan(plot, masks, 2).curl(plot)
                errors.append(sum(np.sum((curl[0][mask]-reference[mask])**2)*level.spacing[0]*level.spacing[1]
                                  for level, mask, curl, reference in zip(plot.levels, masks, curls, exact))**.5)
            orders = np.log2(np.array(errors[:-1])/errors[1:])
            print(f'Fourth-order manufactured curl {layout}: errors={errors}, orders={orders}', flush=True)
            # Interface rows occupy O(h) area: their L2 contribution can converge faster than fourth order.
            self.assertTrue(np.all(np.isfinite(orders) & (orders > 3.6)))

    def test_fourth_order_full_curl_and_covered_cell_exclusion(self):
        plot = fixture(3, 'p3', base=32)
        for level in plot.levels:
            nz, ny, nx = level.valid.shape
            x = 2*np.pi*(np.arange(nx)+.5)/nx
            y = 2*np.pi*(np.arange(ny)+.5)/ny
            z = 2*np.pi*(np.arange(nz)+.5)/nz
            level.values = np.zeros((4, nz, ny, nx)); level.values[0] = 1
            level.values[1] = np.sin(z)[:, None, None]
            level.values[2] = np.sin(x)[None, None, :]
            level.values[3] = np.sin(y)[None, :, None]
        masks = composite_masks(plot)
        plan = CurlPlan(plot, masks, 3)
        curls = plan.curl(plot)
        result = statistics(plot, masks, curls, 3)
        self.assertGreater(result['enstrophy'], 1)
        self.assertGreater(result['z_ux'], .5)
        for level, mask, curl in zip(plot.levels, masks, curls):
            nz, ny, nx = level.valid.shape
            theta = 2*np.pi*(np.arange(nz)+.5)/nz
            coefficient = (8*np.sin(2*np.pi/nz)-np.sin(4*np.pi/nz))/(6*level.spacing[2])
            expected = np.broadcast_to((coefficient*np.cos(theta))[:, None, None], mask.shape)
            np.testing.assert_allclose(curl[1][mask], expected[mask], atol=2e-12, rtol=0)
        for level, mask in zip(plot.levels, masks):
            level.values[:, ~mask] = 1e8
        contaminated = plan.curl(plot)
        for before, after, mask in zip(curls, contaminated, masks):
            np.testing.assert_array_equal(before[:, mask], after[:, mask])

    def test_derivative_identity_in_all_consumers(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); synthetic_curve(root, 'baseline', 128, 2)
            run = DslRun(root)
            plot = fixture(2, 'baseline', base=128); masks = composite_masks(plot)
            metadata = {key: run.id[key] for key in ('source_id', 'descriptor', 'backend', 'layout', 'base_cells')}
            for lev, (level, mask) in enumerate(zip(plot.levels, masks)):
                dump_planes(root/'analysis/planes', f'frame0-L{lev}',
                            dict(vorticity=np.zeros(mask.shape), mach=np.zeros(mask.shape)), mask, level.spacing, 2,
                            time=0, frame=0, level=lev, step=0, coarsest_step=0, phase='Sync',
                            derivative=DERIVATIVE, **metadata)
            self.assertEqual(len(load_planes(run, 0, [mask[0] for mask in masks])), 2)
            plane = root/'analysis/planes/frame0-L0-xy.npz'
            with np.load(plane, allow_pickle=False) as archive:
                stale = {key: archive[key] for key in archive.files}
            stale['derivative'] = 'invalid-reconstruction'
            np.savez_compressed(plane, **stale)
            with self.assertRaisesRegex(ArtifactError, 'unexpected vorticity reconstruction'):
                load_planes(run, 0, [mask[0] for mask in masks])
            checker.curve(root)
            (root/'check').mkdir()
            (root/'check/result.txt').write_text('exit=0\nderivative=invalid-reconstruction\n')
            _, records = summary.load_runs([root])
            self.assertEqual(records[0]['summary_exit'], 2)
            self.assertIn('wrong scientific-result derivative', records[0]['error'])
            (root/'check/result.txt').write_text('exit=2\nerror=unusable reference\n')
            _, records = summary.load_runs([root])
            self.assertEqual(records[0]['scientific_exit'], '2')
            self.assertEqual(records[0]['summary_exit'], 2)
            self.assertIn('unusable reference', records[0]['scientific_details'])
            path = root/'analysis/identity.txt'
            path.write_text(path.read_text().replace(DERIVATIVE, 'invalid-reconstruction'))
            with self.assertRaisesRegex(ArtifactError, 'unexpected DSL derivative reconstruction'):
                checker.curve(root)
            record = {}
            self.assertEqual(summary.curve_prefix(run, record), {})
            self.assertEqual(record['summary_exit'], 2)

    def test_degenerate_fit_is_invalid(self):
        with self.assertRaises(ArtifactError):
            quartic_weights(np.zeros((12, 3)), .1, 3)


def synthetic_curve(directory, layout, n, levels):
    directory.mkdir(parents=True, exist_ok=True); (directory/'analysis').mkdir()
    meta = dict(schema='double-shear-v1', descriptor='D2Q9', dimensions=2, backend='cpu', source_id='a'*40,
                collision='BGK', periodic='1,1,1', initialization='cell-center-equilibrium', real_digits=53,
                layout=layout, base_cells=n, base_z=1, length_z=1/16, levels=levels, full_steps=20*n,
                requested_steps=20*n, frame_stride=n//2, reynolds=10000, u0_lb=.1, u0_phys=1,
                viscosity=.0001, kappa=80, delta=.05, rho_reference=1, velocity_reference=.1,
                gap_fill_radius=4, coarse_nesting_margin=4, grid_efficiency=1)
    for lev in range(levels):
        meta[f'dt_{lev}'] = .1/(n*2**lev); meta[f'tau_{lev}'] = .5+3*.1*n*2**lev/10000
    text = ''.join(f'{key}={value}\n' for key, value in meta.items())
    (directory/'identity.txt').write_text(text)
    (directory/'analysis/identity.txt').write_text(text+'energy_definition=mean_u_norm_squared/2\n'
        'enstrophy_definition=mean_curl_phys_u_norm_squared\nintegration=native-uncovered-valid\n'
        'derivative=central4-and-periodic-quartic-lsq\n')
    with (directory/'samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(('kind', 'level', 'step', 'time', 'coarsest_step', 'phase', 'path'))
        for frame in range(41):
            step = frame*n//2
            for lev in range(levels):
                writer.writerow(('sync', lev, step*2**lev, frame/20, step, 'Sync', f'sync-{step}'))
    (directory/'layout.csv').write_text('level,box,rank,ilo,jlo,klo,ihi,jhi,khi\n')
    paper = checker.minion()
    with (directory/'analysis/curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, COLUMNS); writer.writeheader()
        for frame in range(41):
            reference = paper[min(frame//2, 10)]
            row = {key: 0 for key in COLUMNS}
            row.update(frame=frame, time=frame/20, step=frame*n//2, mass=1,
                       energy=float(reference['energy']), enstrophy=float(reference['enstrophy']))
            writer.writerow(row)


if __name__ == '__main__':
    unittest.main()
