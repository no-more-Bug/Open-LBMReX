#!/usr/bin/env python3
"""Independent curl, composite integration and moving-layout artifact checks."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import analyze_dynamic_double_shear_layer as analysis
import check_dynamic_double_shear_layer as checker
import render_dynamic_double_shear_layer as rendering
from dynamic_vorticity import curls, prolong, restricted, composite_velocity
from amr_plotfile import ArtifactError, Plot, Level, composite_masks, read_plot, identity, rows
from synthetic_amr import write_run, read_from


def hierarchy(dimensions, depth=3, base=16):
    levels = []
    for lev in range(depth):
        n = base*2**lev
        nz = n if dimensions == 3 else 1
        lo, hi = ((0, n) if lev == 0 else ((base//2, 3*base//2) if lev == 1 else (3*base//2, 5*base//2)))
        lower = (lo, lo, lo if dimensions == 3 else 0)
        upper = (hi, hi, hi if dimensions == 3 else 1)
        valid = np.zeros((nz, n, n), dtype=bool)
        valid[lower[2]:upper[2], lo:hi, lo:hi] = True
        z, y, x = np.indices(valid.shape)
        x = (x+.5)/n; y = (y+.5)/n; z = (z+.5)/n
        values = np.ones((4, *valid.shape))
        values[1] = 2*y+(3*z if dimensions == 3 else 0)
        values[2] = 4*x+(5*z if dimensions == 3 else 0)
        values[3] = 6*x+7*y if dimensions == 3 else 0
        values[:, ~valid] = np.nan
        levels.append(Level(values, valid, (1/n, 1/n, 1/n if dimensions == 3 else .125), 0, [(lower, upper)]))
    return Plot(0, (0, 0, 0), (1, 1, 1 if dimensions == 3 else .125), levels)


def artifact(directory, dimensions, grid_efficiency=1):
    n = 128
    depth = 1/32 if dimensions == 3 else 1/16
    plots = {}
    for step, count in ((0, 1), (16, 2)):
        levels = []
        for lev in range(count):
            size = n*2**lev
            nz = size//32 if dimensions == 3 else 1
            lo, hi = ((0, size) if lev == 0 else (size//4, 3*size//4))
            valid = np.zeros((nz, size, size), dtype=bool)
            valid[:, lo:hi, lo:hi] = True
            values = np.ones((4, *valid.shape))
            values[1] = .8; values[2] = .2; values[3] = 0
            values[:, ~valid] = np.nan
            levels.append(Level(values, valid, (1/size, 1/size, depth/nz), step*2**lev,
                                [((lo, lo, 0), (hi, hi, nz))]))
        plots[step] = Plot(step*.1/n, (0, 0, 0), (1, 1, depth), levels)
    meta = dict(schema='dynamic-double-shear-v1', descriptor='D3Q27' if dimensions == 3 else 'D2Q9',
        dimensions=dimensions, max_level=2, max_grid_size=32, base_cells=128, base_z=4 if dimensions == 3 else 1,
        length_z=depth, backend='cuda', source_id='0'*40, layout='dynamic', collision='BGK', real_digits=53,
        periodic='1,1,1', initialization='cell-center-equilibrium', reynolds=10000, u0_lb=.1, u0_phys=1,
        viscosity=.0001, kappa=80, delta=.05, rho_reference=1, velocity_reference=.1,
        gap_fill_radius=4, coarse_nesting_margin=4, grid_efficiency=grid_efficiency, regrid_interval=8,
        full_steps=2560, requested_steps=16, frame_stride=64, ranks=2)
    for lev in range(3):
        meta.update({f'dt_{lev}': .1/(n*2**lev), f'tau_{lev}': .5+3*.1*n*2**lev/10000, f'tag_buffer_{lev}': 2})
        if lev < 2:
            meta.update({f'theta_refine_{lev}': 20, f'theta_keep_{lev}': 10})
    write_run(directory, meta, plots)
    for step, plot in plots.items():
        with (directory/f'layout-{step}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('level', 'box', 'rank', 'ilo', 'jlo', 'klo', 'ihi', 'jhi', 'khi'))
            for lev, level in enumerate(plot.levels):
                for box, (low, high) in enumerate(level.boxes):
                    writer.writerow((lev, box, lev%2, *low, *high))
    return plots


class NativeCurlChecks(unittest.TestCase):
    def test_affine_partial_patches_and_two_interfaces(self):
        for dimensions in (2, 3):
            plot = hierarchy(dimensions)
            values = curls(plot, dimensions)
            for lev, (level, curl) in enumerate(zip(plot.levels, values)):
                mask = level.valid.copy()
                if lev == 0:  # An affine field is not periodic at the external seam.
                    mask[:, :2] = False; mask[:, -2:] = False
                    mask[:, :, :2] = False; mask[:, :, -2:] = False
                    if dimensions == 3:
                        mask[:2] = False; mask[-2:] = False
                expected = (2, -3, 2) if dimensions == 3 else (0, 0, 2)
                for component, value in enumerate(expected):
                    np.testing.assert_allclose(curl[component, mask], value, atol=1e-12, rtol=0)

    def test_mc_extrema_and_child_average(self):
        velocity = np.zeros((3, 5, 5, 5))
        for axis in range(3):
            forward = [2, 2, 2]; backward = forward.copy()
            forward[axis] += 1; backward[axis] -= 1
            velocity[(0, *forward)] = .1
            velocity[(0, *backward)] = -1
        fine = prolong(velocity, (2, 2, 2))
        # At the center, three unscaled slopes .2 would overshoot .1;
        # the multidimensional limiter scales them by 2/3.
        self.assertAlmostEqual(fine[0, 5, 5, 5], .1, places=14)
        self.assertAlmostEqual(fine[0, 4, 4, 4], -.1, places=14)
        np.testing.assert_allclose(restricted(fine, (2, 2, 2)), velocity, atol=1e-14, rtol=0)

    def test_periodic_sine_second_order_and_full_z(self):
        for dimensions in (2, 3):
            plot = hierarchy(dimensions, depth=1)
            level = plot.levels[0]
            n = level.valid.shape[-1]
            z, y, x = np.indices(level.valid.shape)
            level.values[1] = np.sin(2*np.pi*((z if dimensions == 3 else y)+.5)/n)
            level.values[2] = .3*np.sin(2*np.pi*(x+.5)/n)
            level.values[3] = .2*np.sin(2*np.pi*(y+.5)/n) if dimensions == 3 else 0
            result = analysis.statistics(plot, composite_masks(plot), curls(plot, dimensions), dimensions)
            amplitude2 = 1+.3**2+(.2**2 if dimensions == 3 else 0)
            wave = n*np.sin(2*np.pi/n)
            self.assertAlmostEqual(result['energy'], .25*amplitude2, places=12)
            self.assertAlmostEqual(result['enstrophy'], .5*wave**2*amplitude2, places=11)

    def test_covered_coarse_does_not_enter_curl_or_energy(self):
        for dimensions in (2, 3):
            plot = hierarchy(dimensions)
            masks = composite_masks(plot)
            before = curls(plot, dimensions)
            initial = analysis.statistics(plot, masks, before, dimensions)
            for level, mask in zip(plot.levels, masks):
                level.values[:, level.valid & ~mask] = 1e8
            after = curls(plot, dimensions)
            for left, right, mask in zip(before, after, masks):
                np.testing.assert_array_equal(left[:, mask], right[:, mask])
            self.assertEqual(initial, analysis.statistics(plot, masks, after, dimensions))

    def test_d2_ignores_out_of_plane_curl(self):
        plot = hierarchy(2, depth=1)
        plot.levels[0].values[3] = np.indices(plot.levels[0].valid.shape)[1]
        np.testing.assert_array_equal(curls(plot, 2)[0][:2], 0)

    def test_constant_integrals_do_not_weight_energy_by_density(self):
        for dimensions in (2, 3):
            plot = hierarchy(dimensions)
            velocity = (1, 2, 3 if dimensions == 3 else 0)
            for level in plot.levels:
                level.values[0, level.valid] = 7
                for d, value in enumerate(velocity):
                    level.values[d+1, level.valid] = value
            result = analysis.statistics(plot, composite_masks(plot), curls(plot, dimensions), dimensions)
            self.assertAlmostEqual(result['energy'], .5*sum(v*v for v in velocity), places=13)
            self.assertEqual(result['enstrophy'], 0)


class DynamicArtifactChecks(unittest.TestCase):
    def test_new_level_and_short_data_pipeline(self):
        for dimensions in (2, 3):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)/'run'
                plots = artifact(directory, dimensions)
                with patch.object(analysis, 'read_plot', read_from({directory: plots})):
                    run = analysis.open_run(directory, short=True)
                    self.assertEqual([len(run.plot(f).levels) for f in range(2)], [1, 2])
                    output = directory/'analysis'
                    analysis.analyze(run, output)
                    self.assertEqual(checker.check(run, output), 2)
                    planes, faces = rendering.load_planes(run, output, 1)
                    self.assertEqual(len(planes), 2)
                    self.assertTrue(faces)
                    self.assertEqual(sum(int(p['valid'].sum()) for p in planes), 28672)
                    records = checker.rows(output/'curves.csv')
                    self.assertEqual([int(r['active_levels']) for r in records], [1, 2])
                    self.assertTrue(all(abs(float(r['energy'])-.34) < 1e-12 for r in records))
                    self.assertTrue(all(abs(float(r['enstrophy'])) < 1e-12 for r in records))

    def test_grid_efficiency_override_preserves_analysis_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            plots = artifact(directory, 2, grid_efficiency=.7)
            with patch.object(analysis, 'read_plot', read_from({directory: plots})):
                run = analysis.open_run(directory, short=True)
                output = directory/'analysis'
                analysis.analyze(run, output)
                self.assertEqual(identity(output)['grid_efficiency'], '0.7')
                self.assertEqual(checker.check(run, output), 2)
                metadata = output/'identity.txt'
                metadata.write_text(metadata.read_text().replace('grid_efficiency=0.7', 'grid_efficiency=1'))
                with self.assertRaisesRegex(ArtifactError, 'analysis identity differs'):
                    checker.check(run, output)

    def test_invalid_grid_efficiency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            artifact(directory, 2)
            metadata = directory/'identity.txt'
            original = metadata.read_text()
            for value in ('0', '-0.1', '1.01', 'nan', 'inf'):
                with self.subTest(efficiency=value):
                    metadata.write_text(original.replace('grid_efficiency=1', f'grid_efficiency={value}'))
                    with self.assertRaisesRegex(ArtifactError, 'invalid DSL grid_efficiency'):
                        analysis.open_run(directory, short=True)

    def test_short_evidence_is_not_complete_science(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            artifact(directory, 2)
            with self.assertRaises(ArtifactError):
                analysis.open_run(directory)

    def test_missing_frame_or_native_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            plots = artifact(directory, 2)
            with patch.object(analysis, 'read_plot', read_from({directory: plots})):
                run = analysis.open_run(directory, short=True)
                path = directory/'layout-16.csv'
                path.write_text(path.read_text().splitlines()[0]+'\n')
                with self.assertRaises(ArtifactError):
                    run.plot(1)
            path = directory/'samples.csv'
            path.write_text('\n'.join(path.read_text().splitlines()[:2])+'\n')
            with self.assertRaises(ArtifactError):
                analysis.open_run(directory, short=True)

    def test_nonfinite_raw_field_and_failed_solver_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            plots = artifact(directory, 2)
            with patch.object(analysis, 'read_plot', read_from({directory: plots})):
                run = analysis.open_run(directory, short=True)
                output = directory/'analysis'
                analysis.analyze(run, output)
                (directory/'exit.txt').write_text('solver_exit=1\ntee_exit=0\n')
                with self.assertRaises(ArtifactError):
                    checker.check(analysis.open_run(directory, short=True), output)
                (directory/'exit.txt').write_text('solver_exit=0\ntee_exit=0\n')
                plots[16].levels[1].values[1, 0, 64, 64] = np.nan
                with self.assertRaises(ArtifactError):
                    checker.check(run, output)

    def test_failed_uniform_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary)/'seed'
            artifact(seed, 2)
            meta = identity(seed)
            meta.update(schema='double-shear-v1', layout='uniform', levels=1, base_cells=256,
                        full_steps=5120, requested_steps=5120, frame_stride=128,
                        dt_0=.1/256, tau_0=.5+3*.1*256/10000)
            values = np.ones((4, 1, 256, 256))
            values[1] = .8; values[2] = .2; values[3] = 0
            valid = np.ones((1, 256, 256), dtype=bool)
            plots = {frame*128: Plot(frame/20, (0, 0, 0), (1, 1, 1/16),
                [Level(values, valid, (1/256, 1/256, 1/16), frame*128, [((0, 0, 0), (256, 256, 1))])])
                for frame in range(41)}
            directory = Path(temporary)/'reference'
            write_run(directory, meta, plots)
            with patch('analyze_double_shear_layer.read_plot', read_from({directory: plots})):
                reference = analysis.open_run(directory)
                self.assertEqual(len(reference.steps), 41)
                self.assertTrue(np.isfinite(reference.plot(40).levels[0].values).all())
            (directory/'exit.txt').write_text('solver_exit=1\ntee_exit=0\n')
            with self.assertRaises(ArtifactError):
                analysis.open_run(directory)

    def test_uniform_error_on_native_cells(self):
        for dimensions in (2, 3):
            plot = hierarchy(dimensions)
            reference = hierarchy(dimensions, depth=1, base=64)
            error = analysis.velocity_error(plot, reference, composite_masks(plot))
            np.testing.assert_allclose(error, (0, 0), atol=1e-12, rtol=0)


def check_static_initial(directory):
    """Actual old producer with zero evolution steps, retaining its default layout file."""
    meta = identity(directory)
    plot = read_plot(Path(directory)/'sync-0')
    if meta['schema'] != 'double-shear-v1' or int(meta['requested_steps']) != 0 or plot.time != 0:
        raise ArtifactError("wrong static initial producer")
    recorded = rows(Path(directory)/'layout.csv')
    if len(recorded) != sum(len(level.boxes) for level in plot.levels):
        raise ArtifactError("static default layout output changed")
    for level in plot.levels:
        z, y, x = np.indices(level.valid.shape)
        y = (y+.5)*level.spacing[1]
        x = (x+.5)*level.spacing[0]
        shear = np.where(y <= .5, y-.25, .75-y)
        expected = (np.ones_like(x), np.tanh(80*shear), .05*np.sin(2*np.pi*(x+.25)), np.zeros_like(x))
        for component, field in enumerate(expected):
            np.testing.assert_allclose(level.values[component, level.valid], field[level.valid], atol=1e-12, rtol=0)
    print("static_initial_native_pass descriptor="+meta['descriptor'])


if __name__ == '__main__':
    unittest.main()
