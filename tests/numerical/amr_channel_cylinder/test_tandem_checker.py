#!/usr/bin/env python3
"""Small independent force/field fixtures; the sixteen real runs cover native I/O."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

import check_tandem as check
from amr_plotfile import ArtifactError, Level, Plot, composite_masks


def boundaries(layout, grouping):
    return [dict(name=name, boundary_id=str(4+i), object_id=str(0 if grouping == 'shared' else i),
                 level=str(1 if i == 1 and layout == 'cross' else 2), center_x=str(.2+.2*i),
                 center_y='.2', center_z='.005', radius='.05')
            for i, name in enumerate(('cylinder_a', 'cylinder_b'))]


def row(kind, entity, level, step, force):
    interval = kind == 'object_interval'
    dt = check.DT if interval else check.DT/(2**level)
    return dict(zip(check.FORCE_FIELDS, (kind, str(entity), '' if interval else str(level),
                '' if interval else str(step), str(step if interval else (step-1)//(2**level)+1),
                repr((step-1)*dt), repr(step*dt), '' if interval else repr((step-.5)*dt),
                *(repr(float(v)) for v in force))))


def force_records(layout, grouping):
    result = []
    for i, level in ((0, 2), (1, 1 if layout == 'cross' else 2)):
        for n in range(1, 8*(2**level)+1):
            force = (2+n/8, -.25+n/64, 0) if i == 0 else (5+n/4, .5-n/32, 0)
            result.append(row('boundary', 4+i, level, n, force))
            if grouping == 'distinct':
                result.append(row('object_step', i, level, n, force))
    if grouping == 'shared':
        if layout == 'same':
            result.extend(row('object_step', 0, 2, n, (7+3*n/8, .25-n/64, 0)) for n in range(1, 33))
        else:
            # Closed-form arithmetic-series means, independent of checker summation.
            result.extend(row('object_interval', 0, None, n,
                              (7+(4*n-1.5)/8+(2*n-.5)/4,
                               .25+(4*n-1.5)/64-(2*n-.5)/32, 0)) for n in range(1, 9))
    return result


def fixture(layout='cross', grouping='shared'):
    meta = check.boundary_map(boundaries(layout, grouping), layout, grouping)
    forces = check.read_forces(force_records(layout, grouping), meta)
    levels = []
    for lev in range(3):
        ratio = 2**lev
        values = np.ones((4, 1, 8*ratio, 16*ratio))
        valid = np.zeros(values.shape[1:], dtype=bool)
        lo, hi = ((0, 0, 0), (16, 8, 1)) if lev == 0 else (
            ((2, 2, 0), (30, 14, 1)) if lev == 1 else ((8, 8, 0), (24, 24, 1)))
        valid[0, lo[1]:hi[1], lo[0]:hi[0]] = True
        levels.append(Level(values, valid, (.05/ratio, .05/ratio, .01), 8*ratio, [(lo, hi)]))
    plot = Plot(8*check.DT, (0, 0, 0), (.8, .4, .01), levels)
    return SimpleNamespace(id=dict(source_id='a'*40, backend='cpu', descriptor='D2Q9', scheme='YMS',
                                   layout=layout, grouping=grouping, ranks='4'),
                           boundaries=meta, forces=forces, plot=plot, masks=composite_masks(plot), layout=[])


class TandemCheckerTests(unittest.TestCase):
    def test_all_four_structures_and_complete_counts(self):
        for layout in ('cross', 'same'):
            for grouping in ('shared', 'distinct'):
                with self.subTest(layout=layout, grouping=grouping):
                    run = fixture(layout, grouping)
                    self.assertTrue(all(r['passed'] for r in check.force_checks(run.boundaries, run.forces)))
                    count = sum(k[0] == 'boundary' for k in run.forces)
                    self.assertEqual(count, 48 if layout == 'cross' else 64)
                    objects = 8 if layout == 'cross' else 32
                    self.assertEqual(len(run.forces)-count, objects if grouping == 'shared' else count)

    def test_wrong_grouping_or_ownership_is_invalid(self):
        data = boundaries('cross', 'shared')
        data[1]['object_id'] = '1'
        with self.assertRaises(ArtifactError):
            check.boundary_map(data, 'cross', 'shared')
        data = boundaries('cross', 'shared'); data[1]['level'] = '2'
        with self.assertRaises(ArtifactError):
            check.boundary_map(data, 'cross', 'shared')

    def test_missing_duplicate_and_extra_samples_are_invalid(self):
        meta = fixture().boundaries
        records = force_records('cross', 'shared')
        for broken in (records[:-1], records+[records[0]],
                       records+[row('boundary', 99, 2, 1, (1, 0, 0))]):
            with self.assertRaises(ArtifactError):
                check.read_forces(broken, meta)

    def test_native_and_interval_time_labels(self):
        for key, value in (('begin_time', '0.1'), ('sample_time', '0'),
                           ('coarsest_step', '2'), ('population_step', '3')):
            broken = force_records('cross', 'shared'); broken[0][key] = value
            with self.assertRaises(ArtifactError):
                check.read_forces(broken, fixture().boundaries)
        broken = force_records('cross', 'shared'); broken[-1]['sample_time'] = '0.007'
        with self.assertRaises(ArtifactError):
            check.read_forces(broken, fixture().boundaries)

    def test_cross_level_mean_is_not_mean_of_six_samples(self):
        run = fixture()
        wrong = sum(run.forces['boundary', 4, 2, n] for n in range(1, 5))
        wrong += sum(run.forces['boundary', 5, 1, n] for n in range(1, 3))
        run.forces['object_interval', 0, None, 1] = wrong/6
        self.assertFalse(check.force_checks(run.boundaries, run.forces)[-1]['passed'])

    def test_wrong_same_level_sum_and_distinct_result_fail(self):
        for grouping in ('shared', 'distinct'):
            run = fixture('same', grouping)
            run.forces['object_step', 0, 2, 32][1] += 1e-5
            self.assertFalse(check.force_checks(run.boundaries, run.forces)[-1]['passed'])

    def test_zero_or_nonfinite_output_cannot_pass(self):
        run = fixture()
        for value in run.forces.values(): value[:] = 0
        self.assertFalse(all(r['passed'] for r in check.force_checks(run.boundaries, run.forces)))
        records = force_records('cross', 'shared'); records[0]['Fx_lb'] = 'nan'
        with self.assertRaises(ArtifactError): check.read_forces(records, run.boundaries)

    def test_pair_requires_matching_identity_and_distinct_grouping(self):
        shared, distinct = fixture(), fixture(grouping='distinct')
        self.assertTrue(all(r['passed'] for r in check.paired_checks(shared, distinct)))
        for key, value in (('layout', 'same'), ('source_id', 'b'*40), ('backend', 'cuda'),
                           ('scheme', 'Bouzidi'), ('grouping', 'shared')):
            broken = copy.deepcopy(distinct); broken.id[key] = value
            with self.assertRaises(ArtifactError): check.paired_checks(shared, broken)
        distinct.layout = [{'rank': '1'}]
        with self.assertRaises(ArtifactError): check.paired_checks(shared, distinct)

    def test_boundary_force_grouping_parity(self):
        shared, distinct = fixture(), fixture(grouping='distinct')
        shared.forces['boundary', 4, 2, 32][0] += 1e-5
        self.assertFalse(check.paired_checks(shared, distinct)[0]['passed'])

    def test_field_pair_excludes_covered_and_solid_cells(self):
        shared, distinct = fixture(), fixture(grouping='distinct')
        shared.plot.levels[0].values[0, 0, 3, 5] += 100  # covered by L1
        shared.plot.levels[2].values[0, 0, 16, 16] += 100  # inside cylinder A
        self.assertTrue(all(r['passed'] for r in check.paired_checks(shared, distinct)))
        shared.plot.levels[0].values[1, 0, 0, 0] += 1e-5
        self.assertFalse(all(r['passed'] for r in check.paired_checks(shared, distinct)))

    def test_missing_run_returns_invalid_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', str(Path(check.__file__)), '--run', directory,
                                     '--output', str(Path(directory)/'analysis')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('exit=2', (Path(directory)/'analysis/result.txt').read_text())

    def test_layout_must_match_native_boxes_and_valid_ranks(self):
        plot = fixture().plot
        records = []
        for level, data in enumerate(plot.levels):
            lo, hi = data.boxes[0]
            records.append(dict(zip(('level', 'box', 'rank', 'ilo', 'jlo', 'klo', 'ihi', 'jhi', 'khi'),
                                    map(str, (level, 0, level, *lo, *hi)))))
        check.validate_layout(records, plot, 4)
        for broken in ([], records[:-1], records+[records[0]]):
            with self.assertRaises(ArtifactError): check.validate_layout(broken, plot, 4)
        for key, value in (('rank', '4'), ('rank', '-1'), ('box', '1'), ('level', '3'), ('ihi', '15')):
            broken = copy.deepcopy(records); broken[0][key] = value
            with self.assertRaises(ArtifactError): check.validate_layout(broken, plot, 4)


if __name__ == '__main__':
    unittest.main()
