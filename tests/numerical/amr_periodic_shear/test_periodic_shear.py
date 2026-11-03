#!/usr/bin/env python3
"""Narrow independent checks of mapping, cell averages and legal diagnostics."""
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
from amr_plotfile import ArtifactError, conservative_average, read_plot, vismf_header, rows, totals, Plot, Level
from shear_metrics import exact, mode_vectors, interface_faces, face_distance, four_point, expected_refined, norm, U, NU, AMPLITUDE
import summarize_convergence
from check_periodic_shear import Run, SYNC, MAIN, scientific_failures


class ShearChecks(unittest.TestCase):
    def test_momentum_mapping(self):
        values = np.ones((4, 1, 2, 2))
        values[0, 0] = [[1, 2], [3, 4]]
        values[1, 0] = [[0, 1], [2, 3]]
        result = conservative_average(values, (1, 1, 1))
        self.assertEqual(result[0, 0, 0, 0], 2.5)
        self.assertEqual(result[1, 0, 0, 0], 2.0)
        self.assertNotEqual(result[1, 0, 0, 0], values[1].mean())

    def test_analytic_cell_average(self):
        nodes, weights = np.polynomial.legendre.leggauss(8)
        for dimensions, mode in ((2, 'axis'), (2, 'diagonal'), (3, 'axis'), (3, 'diagonal')):
            spacing = (0.125, 0.125, 0.125 if dimensions == 3 else 1/16)
            shape = (1, 1, 1); time = 0.7
            m, p = mode_vectors(dimensions, mode); k = 2*math.pi*m
            signal = 0.0
            for i, a in enumerate(nodes):
                for j, b in enumerate(nodes):
                    for l, c in enumerate(nodes):
                        x = (np.array([a, b, c])+1)*np.array(spacing)/2
                        signal += weights[i]*weights[j]*weights[l]*math.sin(np.dot(k, x)-U*np.linalg.norm(k)*time)/8
            expected = U*m/np.linalg.norm(m)+AMPLITUDE*p*math.exp(-NU*np.dot(k, k)*time)*signal
            np.testing.assert_allclose(exact(shape, spacing, time, dimensions, mode)[1:, 0, 0, 0], expected, atol=1e-15, rtol=0)

    def test_actual_periodic_interfaces(self):
        refined = np.zeros((1, 16, 16), dtype=bool)
        refined[:, :, :2] = True; refined[:, :, 14:] = True; refined[:, :, 6:10] = True
        faces = interface_faces(refined, (1/16, 1/16, 1/16), 2)
        self.assertEqual({round(lo[0]*16) for d, lo, hi in faces}, {2, 6, 10, 14})
        self.assertEqual(len(faces), 4)
        face = (0, np.array([0., .25, 0.]), np.array([0., .75, 1/16]))
        distance = face_distance((1, 8, 8), (1/8, 1/8, 1/16), (1, 1, 1/16), face, 2)
        self.assertAlmostEqual(distance[0, 0, 0], math.sqrt(10)/16)
        self.assertAlmostEqual(distance[0, 0, -1], math.sqrt(10)/16)

    def test_four_point_and_missing_neighbor(self):
        values = np.zeros((4, 1, 1, 8)); valid = np.ones((1, 1, 8), dtype=bool)
        values[1, 0, 0] = (-1.)**np.arange(8)
        axis, coefficient, legal = next(four_point(values, valid, 2))
        np.testing.assert_array_equal(coefficient[1], values[1])
        valid[0, 0, 7] = False
        values[1, 0, 0] = np.arange(8.)**2
        axis, coefficient, legal = next(four_point(values, valid, 2))
        np.testing.assert_array_equal(np.flatnonzero(legal), [0, 1, 2, 3])
        np.testing.assert_allclose(coefficient[:, legal], 0, atol=1e-15)

    def test_truncated_and_nonfinite_headers(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name); (path/'Header').write_text('HyperCLaw-V1.1\n4\nrho_norm\n')
            with self.assertRaises(ArtifactError):
                read_plot(path)
            (path/'Header').write_text('HyperCLaw-V1.1\n4\nrho_norm\nux_norm\nuy_norm\nuz_norm\n3\n0\n0\n0 0 0\ninf 1 1\n')
            with self.assertRaises(ArtifactError):
                read_plot(path)
            h = path/'Cell_H'
            h.write_text('1\n1\n4\n0\n(1 0\n((0,0,0) (1,1,0) (0,0,0))\n)\n1\nFabOnDisk: Cell_D_00000 0\n')
            with self.assertRaises(StopIteration):
                vismf_header(h, 1)

    def test_layout_identities_are_distinct(self):
        shape = (64, 64, 64); spacing = (1/64,)*3
        corner = expected_refined(shape, spacing, 3, 'corner')
        shell = expected_refined(shape, spacing, 3, 'shell')
        self.assertTrue(corner[25, 25, 25]); self.assertFalse(shell[25, 25, 25])
        self.assertTrue(corner[16, 48, 48]); self.assertFalse(corner[48, 48, 48])
        self.assertFalse(np.array_equal(corner, shell))

    def test_invalid_statistics_and_short_csv(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'samples.csv'; path.write_text('kind,level,step,time\nsync,0,0\n')
            with self.assertRaises(ArtifactError):
                rows(path)
        valid = np.ones((1, 2, 2), dtype=bool)
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(ArtifactError):
                norm(np.full(valid.shape, 1e308), valid, (.5, .5, 1), 2)
            level = Level(np.full((4, *valid.shape), 1e308), valid, (.5, .5, 1), 0, [])
            with self.assertRaises(ArtifactError):
                totals(Plot(0, (0, 0, 0), (1, 1, 1), [level]), [valid])

    def test_empty_convergence_is_invalid(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for n in (64, 128, 256):
                directory = root/f'half-N{n}/analysis'; directory.mkdir(parents=True)
                (directory/'metrics.csv').write_text('time,comparison,region,level,l1,l2,peak\n')
            with patch.object(sys, 'argv', ['summary', str(root), '64', '128', '256']):
                self.assertEqual(summarize_convergence.main(), 2)

    def test_complex_budget_scope(self):
        records = [dict(time=1, comparison=kind, region='axis-2', l2=.009, peak=peak)
                   for kind, peak in [('analytic-domain', .02), ('reference-domain', .02),
                                      ('analytic-interface', .059), ('reference-interface', .059),
                                      ('pressure', .014), ('spatial-velocity', .029),
                                      ('temporal-velocity', .029)]]
        for layout in ('corner', 'shell', 'separated'):
            run = SimpleNamespace(id={'layout': layout}, dimensions=3, mode='diagonal', nc=64)
            self.assertEqual(scientific_failures(run, records, []), [])
        for dimensions, mode, nc, layout in ((2, 'diagonal', 128, 'corner'),
                (2, 'diagonal', 128, 'shell'), (2, 'diagonal', 128, 'separated'),
                (3, 'diagonal', 128, 'half'), (3, 'axis', 128, 'half'),
                (3, 'axis', 64, 'corner'), (3, 'diagonal', 128, 'shell')):
            run = SimpleNamespace(id={'layout': layout}, dimensions=dimensions, mode=mode, nc=nc)
            with self.subTest(dimensions=dimensions, mode=mode, nc=nc, layout=layout):
                self.assertEqual(len(scientific_failures(run, records, [])), 5)

    def test_budget_boundaries_and_unchanged_limits(self):
        run = SimpleNamespace(id={'layout': 'shell'}, dimensions=3, mode='diagonal', nc=64)
        for time in MAIN:
            for comparison, field, limit in [('analytic-domain', 'l2', .01),
                    ('reference-domain', 'l2', .01), ('analytic-interface', 'peak', .06),
                    ('reference-interface', 'peak', .06), ('pressure', 'peak', .015),
                    ('spatial-velocity', 'peak', .03), ('temporal-velocity', 'peak', .03),
                    ('spatial-pressure', 'peak', .01), ('temporal-pressure', 'peak', .01)]:
                row = dict(time=time, comparison=comparison, region='axis-2', **{field: limit})
                with self.subTest(time=time, comparison=comparison):
                    self.assertEqual(scientific_failures(run, [row], []), [])
                    row[field] = np.nextafter(limit, math.inf)
                    self.assertEqual(len(scientific_failures(run, [row], [])), 1)
                    row['time'] = .5
                    self.assertEqual(scientific_failures(run, [row], []), [])
        for role in ('amr', 'reference'):
            for field in ('mass_drift', 'momentum_drift'):
                balance = dict(role=role, time=.5, mass_drift=1e-10, momentum_drift=1e-10)
                self.assertEqual(scientific_failures(run, [], [balance]), [])
                balance[field] = np.nextafter(1e-10, math.inf)
                self.assertEqual(scientific_failures(run, [], [balance]), [f'{role}: conservation at t=0.5'])

    def test_coarser_half_retains_conservation_gate(self):
        records = [dict(time=1, comparison='reference-domain', l2=.1),
                   dict(time=1, comparison='temporal-velocity', peak=.2)]
        balance = dict(role='amr', time=.125, mass_drift=1e-10, momentum_drift=2e-10)
        for dimensions, nc in ((2, 64), (2, 128), (3, 32), (3, 64)):
            run = SimpleNamespace(id={'layout': 'half'}, dimensions=dimensions, mode='diagonal', nc=nc)
            self.assertEqual(scientific_failures(run, records, [balance]), ['amr: conservation at t=0.125'])

    def test_identity_horizon_and_missing_samples(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = dict(collision='BGK', periodic='1,1,1', initialization='cell-center-equilibrium', schema='periodic-shear-v1', descriptor='D2Q9', dimensions=2, backend='cpu', source_id='a'*40,
                        real_digits=53, coarse_cells=64, base_cells=64, levels=2, layout='half', mode='axis',
                        full_steps=256, requested_steps=256, viscosity=NU, background=U, amplitude=AMPLITUDE,
                        rho_reference=1, velocity_reference=1, gap_fill_radius=4, coarse_nesting_margin=4,
                        grid_efficiency=1, dt_0=1/64, dt_1=1/128, tau_0=.65, tau_1=.8)
            def save():
                (root/'identity.txt').write_text(''.join(f'{k}={v}\n' for k, v in data.items()))
            save()
            (root/'samples.csv').write_text('kind,level,step,time,coarsest_step,phase,path\n')
            with self.assertRaisesRegex(ArtifactError, 'missing sync'):
                Run(root)
            data['requested_steps'] = 64; save()
            with self.assertRaisesRegex(ArtifactError, 'horizon'):
                Run(root)
            data['requested_steps'] = 256; data['descriptor'] = 'D3Q27'; save()
            with self.assertRaisesRegex(ArtifactError, 'descriptor'):
                Run(root)


if __name__ == '__main__':
    unittest.main()
