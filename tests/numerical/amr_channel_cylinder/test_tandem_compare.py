#!/usr/bin/env python3
"""Independent arithmetic-series expectations for the C8 observation consumer."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np

import compare_tandem as compare
from test_tandem_checker import boundaries, force_records
from check_tandem import boundary_map, read_forces


class TandemComparisonTests(unittest.TestCase):
    def test_unit_span_and_four_two_sample_weights(self):
        for layout in ('same', 'cross'):
            meta = boundary_map(boundaries(layout, 'shared'), layout, 'shared')
            force = read_forces(force_records(layout, 'shared'), meta, 8)
            result = compare.coefficients(meta, force, 8,
                dict(dx_2='.00125,.00125,.01', rho_reference='1', velocity_reference='.035'))
            n = np.arange(1, 9)
            # Closed-form means of the independently specified arithmetic sequences.
            a = np.column_stack((2+(4*n-1.5)/8, -.25+(4*n-1.5)/64))/.049
            b_step = 4*n-1.5 if layout == 'same' else 2*n-.5
            b = np.column_stack((5+b_step/4, .5-b_step/32))/.049
            np.testing.assert_allclose(result['cylinder_a'], a, rtol=1e-14)
            np.testing.assert_allclose(result['cylinder_b'], b, rtol=1e-14)

    def test_missing_force_and_wrong_interval_are_rejected(self):
        meta = boundary_map(boundaries('cross', 'shared'), 'cross', 'shared')
        original = force_records('cross', 'shared')
        wrong = copy.deepcopy(original)
        wrong[0]['end_time'] = '0.5'
        for records in (original[1:], wrong):
            with self.assertRaises(compare.ArtifactError):
                read_forces(records, meta, 8)

    def test_snapshot_schedule_and_missing_or_wrong_clock(self):
        records = [dict(kind='sync', phase='Sync', level=str(lev), step=str(step*2**lev),
                        coarsest_step=str(step), time=str(step*compare.DT), path=f'sync-{step}')
                   for step in compare.SNAPSHOTS for lev in range(3)]
        compare.sample_index(records)
        wrong = copy.deepcopy(records)
        wrong[-1]['time'] = '17.6'
        for broken in (records[:-1], records+[records[-1]], wrong):
            with self.assertRaises(compare.ArtifactError):
                compare.sample_index(broken)

    def test_pair_source_and_layout_identity(self):
        same = SimpleNamespace(id=dict(layout='same', source_id='a'*40, full_steps='20000'),
                               layout=[dict(level='1', box='0', ilo='0', ihi='8')])
        cross = SimpleNamespace(id=dict(layout='cross', source_id='a'*40, full_steps='20000'),
                                layout=[dict(level='1', box='0', ilo='0', ihi='4'),
                                        dict(level='1', box='1', ilo='4', ihi='8')])
        compare.pair(same, cross)
        for key, value in (('source_id', 'b'*40), ('full_steps', '8'), ('layout', 'same')):
            wrong = copy.deepcopy(cross)
            wrong.id[key] = value
            with self.assertRaises(compare.ArtifactError):
                compare.pair(same, wrong)


if __name__ == '__main__':
    unittest.main()
