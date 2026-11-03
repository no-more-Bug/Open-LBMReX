#!/usr/bin/env python3
"""Independent analytic fields and raw-record fixtures for the Re20 checker.

Complete synthetic checks replace only binary/CSV transport. All case identity,
clock, force normalization, interpolation and scientific decisions are real.
The synthetic fields are mathematical fixtures, not a manufactured solver run.
"""
import contextlib
import csv
import io
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

import check_re20 as checker
from amr_plotfile import Level, Plot, composite_masks

CD, CL, CP = 5.57953523384, .010618948146, 2.93800417425


def native_plot(layout='coarse', dimensions=2, density=None, velocity=None):
    h = .00125 if layout == 'fine' else .005
    low = (-h/2, -h/2, 0.)
    nx, ny = round(2.2/h)+1, round(.41/h)+1
    levels = []
    for lev in range(3 if layout == 'amr' else 1):
        ratio = 2**lev; spacing = h/ratio
        nz = 1 if dimensions == 2 else round(.01/spacing)
        shape = (nz, ny*ratio, nx*ratio)
        bounds = ((0, 0, 0), (shape[2], shape[1], nz))
        if lev == 1: bounds = ((20, 20, 0), (482, 142, nz))
        if lev == 2: bounds = ((82, 82, 0), (802, 242, nz))
        lo, hi = bounds
        valid = np.zeros(shape, dtype=bool); valid[:, lo[1]:hi[1], lo[0]:hi[0]] = True
        x = low[0]+(np.arange(shape[2])+.5)*spacing
        y = low[1]+(np.arange(shape[1])+.5)*spacing
        x, y = np.meshgrid(x, y)
        values = np.zeros((4, *shape)); values[0] = 1
        if density is not None: values[0] = density(x, y)
        if velocity is not None:
            for component, value in enumerate(velocity(x, y), 1): values[component] = value
        levels.append(Level(values, valid, (spacing, spacing, .01 if dimensions == 2 else spacing), 0, [bounds]))
    return Plot(0., low, (2.2+h/2, .41+h/2, .01), levels)


class ForceRows:
    fieldnames = list(checker.FORCES)

    def __init__(self, run): self.run = run

    def __iter__(self):
        run = self.run
        lev = 2 if run.layout == 'amr' else 0
        ratio = 2**lev
        total = (320000 if run.layout == 'fine' else 80000)*ratio
        dt = .000875/(4 if run.layout == 'fine' else 1)/ratio
        denominator = .5*.035**2*(20 if run.layout == 'coarse' else 80)
        if run.dimensions == 3: denominator *= 2 if run.layout == 'coarse' else 8
        for step in range(1, total+1):
            coefficients = (run.cd, run.cl, run.cz if step == 17 else 0.)
            for kind, identifier in (('boundary', 4), ('object', 2)):
                yield dict(kind=kind, id=identifier, level=lev, population_step=step,
                    coarsest_step=(step+ratio-1)//ratio, begin_time=(step-1)*dt, end_time=step*dt,
                    sample_time=(step-.5)*dt, Fx_lb=coefficients[0]*denominator,
                    Fy_lb=coefficients[1]*denominator, Fz_lb=coefficients[2]*denominator,
                    Cd=coefficients[0], Cl=coefficients[1], Cz=coefficients[2])


class SyntheticRun:
    def __init__(self, directory, layout='coarse', dimensions=2, scheme='YMS'):
        self.directory = Path(directory); self.directory.mkdir()
        self.layout, self.dimensions = layout, dimensions
        self.cd, self.cl, self.cp, self.cz = CD, CL, CP, 0.
        self.residuals = [0., 0., 0.]
        self.speed, self.pulse, self.wake_pulse, self.outlet = 1., 0., 0., 0.
        self.factor = 4 if layout == 'fine' else 1
        self.h = .00125 if layout == 'fine' else .005
        self.meta = dict(schema='re20-v1', descriptor='D2Q9' if dimensions == 2 else 'D3Q27',
            dimensions=dimensions, source_id='a'*40, backend='cpu', ranks=2, real_digits=53,
            scheme=scheme, collision='BGK', periodic='0,0,1', layout=layout,
            levels=3 if layout == 'amr' else 1, max_grid_size=64, physical_lo='0,0,0', physical_hi='2.2,.41,.01',
            envelope_lo=f'{-self.h/2},{-self.h/2},0', envelope_hi=f'{2.2+self.h/2},{.41+self.h/2},.01',
            cylinder_center='.2,.2', cylinder_radius=.05, reynolds=20, u0_lb=.035, u0_phys=.2, viscosity=.001,
            rho_reference=1, velocity_reference=.035, comparison_dt=.000875, comparison_steps=self.factor,
            full_steps=80000*self.factor, requested_steps=80000*self.factor, boundary_id=4, object_id=2,
            owner_level=2 if layout == 'amr' else 0, diameter_lb=20 if layout == 'coarse' else 80,
            span_lb=2 if layout == 'coarse' else 8)
        self.meta['force_denominator'] = .5*.035**2*self.meta['diameter_lb']*(self.meta['span_lb'] if dimensions == 3 else 1)
        self.refresh_fields()
        for lev, data in enumerate(self.template.levels):
            self.meta[f'cells_{lev}'] = ','.join(map(str, reversed(data.valid.shape)))
            self.meta[f'dx_{lev}'] = ','.join(map(str, data.spacing))
            self.meta[f'dt_{lev}'] = .000875/self.factor/2**lev
            self.meta[f'tau_{lev}'] = .5+3*.001*self.meta[f'dt_{lev}']/data.spacing[0]**2
        self.write_identity()
        (self.directory/'exit.txt').write_text('solver_exit=0\ntee_exit=0\n')
        (self.directory/'forces.csv').touch()
        with (self.directory/'layout.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('level','box','rank','ilo','jlo','klo','ihi','jhi','khi'))
            for lev, data in enumerate(self.template.levels): writer.writerow((lev, 0, 0, *data.boxes[0][0], *data.boxes[0][1]))
        common = sorted(set(range(0, 80001, 8000)) | {100, 1000, 79799, 79800, 79899, 79900, 79999})
        with (self.directory/'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('kind','level','step','time','coarsest_step','phase','path'))
            for m in common:
                for lev in range(self.meta['levels']):
                    writer.writerow(('sync',lev,m*self.factor*2**lev,m*.000875,m*self.factor,'Sync',f'sync-{m*self.factor}'))
        self.write_science()

    def refresh_fields(self):
        slope = self.cp*3*.035**2/.1
        def rho(x, y): return 1+slope*np.maximum(0, np.minimum(x-.05, .25-x))
        def velocity(x, y):
            ux = self.speed+self.pulse*np.maximum(0,1-np.abs(y-.02)/.005)+self.wake_pulse*np.maximum(0,1-np.abs(x-1.6)/.005)
            ux[:, -1] += self.outlet
            return ux,np.zeros_like(x),np.zeros_like(x)
        self.template = native_plot(self.layout, self.dimensions, rho, velocity)

    def write_identity(self):
        (self.directory/'identity.txt').write_text(''.join(f'{k}={v}\n' for k,v in self.meta.items()))

    def write_science(self):
        self.refresh_fields()
        self.observations = []
        for n in range(801):
            initial = n == 0
            phi = self.speed+self.pulse*.005/.41
            interval_end, interval_begin = 100*n*.000875,(100*n-1)*.000875
            interval_weight = (interval_end-interval_begin)/.000875
            self.observations.append(dict(step=100*n*self.factor, common_step=100*n, time=n*.0875,
                comparison_begin_time='' if initial else n*.0875-.000875,
                relative_l2='' if initial else self.residuals[n-798] if n >= 798 else 0,
                force_begin_time='' if initial else interval_begin, force_end_time='' if initial else interval_end,
                Cd='' if initial else self.cd*interval_weight, Cl='' if initial else self.cl*interval_weight,
                rho_front=1+self.cp*3*.035**2, rho_rear=1, Cdp=self.cp, phi_in=phi, phi_out=phi+self.outlet,
                flux_imbalance=abs(self.outlet), uz_max=0 if self.dimensions == 3 else '',
                rho_z_max=0 if self.dimensions == 3 else '', velocity_z_max=0 if self.dimensions == 3 else ''))
        self.write_observations()
        with (self.directory/'profiles.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(('line','coordinate','x','y','ux_norm','uy_norm'))
            for name, axis, value, start, stop, count in (
                    ('x0.05',0,.05,0,.41,329), ('x0.30',0,.30,0,.41,329),
                    ('x0.60',0,.60,0,.41,329), ('x1.50',0,1.5,0,.41,329), ('y0.20',1,.2,.3,2.1,1441)):
                for coordinate in np.linspace(start, stop, count):
                    x,y = (value,coordinate) if axis == 0 else (coordinate,value)
                    u = self.speed+self.pulse*max(0,1-abs(y-.02)/.005)+self.wake_pulse*max(0,1-abs(x-1.6)/.005)
                    writer.writerow((name, coordinate, x, y, u, 0))

    def write_observations(self):
        with (self.directory/'observations.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, checker.OBSERVATIONS); writer.writeheader(); writer.writerows(self.observations)

    def read(self, path):
        step = int(Path(path).name.split('-')[1]); common = step//self.factor
        ratio = 1+self.residuals[(common+1)//100-798] if common in (79799,79899,79999) else 1
        data = []
        for lev, source in enumerate(self.template.levels):
            values = source.values if ratio == 1 else source.values.copy()
            if ratio != 1: values[1:] *= ratio
            data.append(Level(values, source.valid, source.spacing, step*2**lev, source.boxes))
        return Plot(step*.000875/self.factor, self.template.low, self.template.high, data)


class CheckerTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(os.environ.get('TMPDIR', Path(__file__).resolve().parent/'build'/'consumer-tests'))
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='re20-checker-', dir=scratch)
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def evaluate(self, run, reference=None, d2=None):
        runs = {item.directory: item for item in (run,reference,d2) if item is not None}
        original = csv.DictReader
        def reader(stream, *args, **kwargs):
            path = Path(getattr(stream, 'name', ''))
            return ForceRows(runs[path.parent]) if path.name == 'forces.csv' else original(stream, *args, **kwargs)
        with mock.patch.object(checker, 'read_plot', side_effect=lambda p:runs[Path(p).parent].read(p)), \
             mock.patch.object(checker.csv, 'DictReader', side_effect=reader), contextlib.redirect_stdout(io.StringIO()):
            return checker.check(run.directory, reference.directory if reference else None, self.root/'analysis',
                                 d2.directory if d2 else None)

    def test_shifted_pressure_is_linear_exact_from_fluid_side(self):
        for layout in ('coarse','fine','amr'):
            plot = native_plot(layout, density=lambda x,y:1+.2*x-.3*y)
            result = checker.pressure(plot)
            self.assertAlmostEqual(result['rho_front'], .97)
            self.assertAlmostEqual(result['rho_rear'], .99)
            self.assertAlmostEqual(result['Cdp'], -.02/(3*.035**2))
            masks = checker.fluid_masks(plot)
            self.assertTrue(all(mask.sum() < source.valid.sum() for mask,source in zip(masks,plot.levels)))

    def test_profiles_use_physical_coordinates_and_all_five_lines(self):
        for layout in ('coarse','fine','amr'):
            plot = native_plot(layout, velocity=lambda x,y:(1+2*x-3*y, -x+4*y, 0*x))
            profiles, _ = checker.profiles(plot, checker.fluid_masks(plot))
            self.assertEqual([len(v) for v in profiles.values()], [329]*4+[1441])
            for name, values in profiles.items():
                axis, position, *_ = checker.LINES[name]
                x,y = (position,values[:,0]) if axis == 0 else (values[:,0],position)
                np.testing.assert_allclose(values[:,1], 1+2*x-3*y, atol=2e-14)
                np.testing.assert_allclose(values[:,2], -x+4*y, atol=2e-14)

    def test_profile_trapezoidal_weights_and_vector_peak(self):
        a = {'line':np.array([[0.,.03,.04],[.5,0,0],[1.,0,0]])}
        b = {'line':np.array([[0.,0,0],[.5,0,0],[1.,0,0]])}
        result = checker.profile_errors(a,b)
        self.assertAlmostEqual(result[0][1], .025)
        self.assertAlmostEqual(result[1][1], .05)

    def test_flux_averages_product_then_trapezoid(self):
        plot = native_plot(dimensions=3)
        data = plot.levels[0]; _, y = checker.coordinates(plot,0)
        data.values[0,0] = 2; data.values[0,1] = 1
        data.values[1,0] = (1+y)[:,None]; data.values[1,1] = (3-y)[:,None]
        result = checker.flux(plot)
        self.assertAlmostEqual(result['phi_in'], 2.5+.41/4)
        self.assertAlmostEqual(result['phi_out'], 2.5+.41/4)

    def test_fluid_surface_covered_and_current_denominator(self):
        current = native_plot('amr', velocity=lambda x,y:(1+0*x,0*x,0*x))
        previous = native_plot('amr', velocity=lambda x,y:(.5+0*x,0*x,0*x))
        masks = checker.fluid_masks(current)
        for lev,(data,mask) in enumerate(zip(current.levels,masks)):
            data.values[1:,~mask] = 1e6
            previous.levels[lev].values[1:,~mask] = -1e6
        self.assertAlmostEqual(checker.relative_l2(previous,current,masks), .5)
        uniform = native_plot()
        mask = checker.fluid_masks(uniform)[0]
        self.assertTrue(mask[0,40,30]); self.assertTrue(mask[0,40,50])
        self.assertFalse(mask[0,40,40])

    def test_z_statistics_before_averaging(self):
        plot = native_plot(dimensions=3)
        data = plot.levels[0]
        data.values[0,0] = 1.2; data.values[0,1] = .8
        data.values[1,0] = .3; data.values[1,1] = -.3
        data.values[2,0] = .4; data.values[2,1] = -.4; data.values[3] = .1
        result = checker.z_statistics(plot, checker.fluid_masks(plot))
        self.assertAlmostEqual(result['rho_z_max'],.2)
        self.assertAlmostEqual(result['velocity_z_max'],.5)
        self.assertAlmostEqual(result['uz_max'],.1)

    def test_three_layouts_complete_and_qualified_reference(self):
        fine = SyntheticRun(self.root/'fine','fine')
        coarse = SyntheticRun(self.root/'coarse')
        amr = SyntheticRun(self.root/'amr','amr')
        self.assertEqual(self.evaluate(fine),0)
        self.assertEqual(self.evaluate(coarse,fine),0)
        self.assertEqual(self.evaluate(amr,fine),0)

    def test_external_and_reference_qualification_are_distinct(self):
        fine = SyntheticRun(self.root/'fine','fine'); fine.cd *= 1.02; fine.write_science()
        amr = SyntheticRun(self.root/'amr','amr'); amr.cd = fine.cd; amr.write_science()
        coarse = SyntheticRun(self.root/'coarse')
        self.assertEqual(self.evaluate(fine),1)
        self.assertEqual(self.evaluate(amr,fine),2)
        self.assertIn('local_exit=0',(self.root/'analysis/result.txt').read_text())
        self.assertIn('reference_exit=1',(self.root/'analysis/result.txt').read_text())
        self.assertEqual(self.evaluate(coarse,fine),0)
        fine.cd = CD
        for key,value in (('cl',CL+.003),('cp',CP*1.04)):
            original = getattr(fine,key); setattr(fine,key,value); fine.write_science()
            self.assertEqual(self.evaluate(fine),1)
            self.assertIn('failed=external_'+{'cl':'Cl','cp':'Cdp'}[key],(self.root/'analysis/result.txt').read_text())
            setattr(fine,key,original)
        fine.cd = 0.; fine.write_science()
        self.assertEqual(self.evaluate(amr,fine),2)

    def test_amr_coefficient_and_profile_gates(self):
        fine = SyntheticRun(self.root/'fine','fine')
        amr = SyntheticRun(self.root/'amr','amr')
        for key, value in (('cd',CD*1.02),('cl',CL+.003),('cp',CP*1.04)):
            original = getattr(amr,key); setattr(amr,key,value); amr.write_science()
            self.assertEqual(self.evaluate(amr,fine),1)
            self.assertIn('failed=fine_'+{'cd':'Cd','cl':'Cl','cp':'Cdp'}[key],(self.root/'analysis/result.txt').read_text())
            setattr(amr,key,original)
        amr.speed = 1.02; amr.write_science()
        self.assertEqual(self.evaluate(amr,fine),1)
        comparison = checker.rows(self.root/'analysis/fine-comparison.csv')
        self.assertTrue(all(row['passed']=='1' for row in comparison if row['metric'].endswith('_peak')))
        result = (self.root/'analysis/result.txt').read_text()
        for name in ('x0.05','x0.30','x0.60','x1.50','y0.20'): self.assertIn('failed=fine_'+name+'_rms',result)
        amr.speed = 1.; amr.pulse = .05; amr.write_science()
        self.assertEqual(self.evaluate(amr,fine),1)
        comparison = checker.rows(self.root/'analysis/fine-comparison.csv')
        self.assertTrue(all(row['passed']=='1' for row in comparison if row['metric'].endswith('_rms')))
        result = (self.root/'analysis/result.txt').read_text()
        for name in ('x0.05','x0.30','x0.60','x1.50'): self.assertIn('failed=fine_'+name+'_peak',result)
        amr.pulse = 0.; amr.wake_pulse = .05; amr.write_science()
        self.assertEqual(self.evaluate(amr,fine),1)
        result = (self.root/'analysis/result.txt').read_text()
        self.assertIn('failed=fine_y0.20_peak',result)
        self.assertNotIn('_rms',result)

    def test_last_three_residuals_and_only_late_flow_are_gates(self):
        run = SyntheticRun(self.root/'run')
        for index in range(3):
            run.residuals = [0.,0.,0.]; run.residuals[index] = 1e-6; run.write_science()
            self.assertEqual(self.evaluate(run),1)
        run.residuals = [0.,0.,0.]; run.write_science()
        # A non-saved early observation is diagnostic only.
        run.observations[7]['phi_out'] += .002; run.observations[7]['flux_imbalance'] = .002
        run.write_observations(); self.assertEqual(self.evaluate(run),0)
        run.outlet = .002; run.write_science()
        self.assertEqual(self.evaluate(run),1)
        result = (self.root/'analysis/result.txt').read_text()
        for m in (79800,79900,80000): self.assertIn(f'failed=flux_imbalance_m{m}',result)

    def test_old_endpoint_is_not_complete_or_a_qualified_reference(self):
        fine = SyntheticRun(self.root/'fine','fine'); amr = SyntheticRun(self.root/'amr','amr')
        fine.meta['requested_steps'] = 240000; fine.write_identity()
        self.assertEqual(self.evaluate(fine),2)
        self.assertEqual(self.evaluate(amr,fine),2)
        fine.meta['requested_steps'] = 320000; fine.write_identity()
        fine.observations = fine.observations[:601]; fine.write_observations()
        self.assertEqual(self.evaluate(fine),2)

    def test_d3_all_observations_and_raw_force_not_average(self):
        run = SyntheticRun(self.root/'run',dimensions=3)
        self.assertEqual(self.evaluate(run),0)
        for key in ('uz_max','rho_z_max','velocity_z_max'):
            run.observations[7][key] = 2e-10; run.write_observations()
            self.assertEqual(self.evaluate(run),1)
            self.assertIn('failed='+key,(self.root/'analysis/result.txt').read_text())
            run.observations[7][key] = 0
        run.write_observations(); run.cz = 2e-10
        self.assertEqual(self.evaluate(run),1)
        self.assertIn('failed=max_abs_Cz',(self.root/'analysis/result.txt').read_text())

    def test_d3_d2_is_a_report_and_references_must_match(self):
        d3 = SyntheticRun(self.root/'d3',dimensions=3)
        d2 = SyntheticRun(self.root/'d2')
        d2.speed = 1.1; d2.residuals = [0.,0.,1e-6]; d2.write_science()
        self.assertEqual(self.evaluate(d3,d2=d2),0)
        self.assertIn('d2_local_exit=1',(self.root/'analysis/result.txt').read_text())
        comparisons = checker.rows(self.root/'analysis/d2-comparison.csv')
        self.assertTrue(any(row['passed']=='0' for row in comparisons))
        self.assertTrue(all(row['gate']=='0' for row in comparisons))
        d2.meta['source_id'] = 'b'*40; d2.write_identity()
        self.assertEqual(self.evaluate(d3,d2=d2),2)

    def test_missing_frame_wrong_interval_units_and_reference_fail_are_invalid(self):
        run = SyntheticRun(self.root/'run')
        original = (run.directory/'samples.csv').read_text()
        (run.directory/'samples.csv').write_text('\n'.join(original.splitlines()[:-1])+'\n')
        self.assertEqual(self.evaluate(run),2)
        (run.directory/'samples.csv').write_text(original)
        run.observations[-1]['comparison_begin_time'] -= .000875
        run.write_observations(); self.assertEqual(self.evaluate(run),2)
        run.write_science(); run.meta['velocity_reference'] = .2; run.write_identity()
        self.assertEqual(self.evaluate(run),2)
        fine = SyntheticRun(self.root/'fine','fine'); amr = SyntheticRun(self.root/'amr','amr')
        (fine.directory/'exit.txt').write_text('solver_exit=1\ntee_exit=0\n')
        self.assertEqual(self.evaluate(amr,fine),2)
        self.assertIn('local_exit=0',(self.root/'analysis/result.txt').read_text())

    def test_raw_force_transport_interval_scale_and_missing_record(self):
        for dimensions, layout in ((2,'coarse'),(3,'amr')):
            directory = self.root/f'd{dimensions}'; directory.mkdir()
            ratio = 4 if layout == 'amr' else 1
            denominator = .5*.035**2*(80*8 if dimensions == 3 else 20)
            run = SimpleNamespace(directory=directory,owner_level=2 if ratio==4 else 0,full_steps=100,
                dt=.000875,factor=1,denominator=denominator,id=dict(boundary_id=4,object_id=2))
            raw = []
            for step in range(1,100*ratio+1):
                value = 1+step/(100*ratio)
                for kind,identifier in (('boundary',4),('object',2)):
                    raw.append(dict(kind=kind,id=identifier,level=run.owner_level,population_step=step,
                        coarsest_step=(step+ratio-1)//ratio,begin_time=(step-1)*run.dt/ratio,
                        end_time=step*run.dt/ratio,sample_time=(step-.5)*run.dt/ratio,
                        Fx_lb=value*denominator,Fy_lb=0,Fz_lb=0,Cd=value,Cl=0,Cz=0))
            expected = sum(1+step/(100*ratio) for step in range(100*ratio-ratio+1,100*ratio+1))/ratio
            observation = [{},dict(Cd=expected,Cl=0,force_begin_time=99*.000875,force_end_time=100*.000875)]
            checker.write_csv(directory/'forces.csv',checker.FORCES,raw)
            intervals,_ = checker.force_intervals(run,observation)
            self.assertAlmostEqual(intervals[0]['Cd'],expected)
            raw[-1]['begin_time'] -= .000875
            checker.write_csv(directory/'forces.csv',checker.FORCES,raw)
            with self.assertRaises(ValueError): checker.force_intervals(run,observation)
            raw[-1]['begin_time'] += .000875
            raw[-1]['Fx_lb'] *= 2; checker.write_csv(directory/'forces.csv',checker.FORCES,raw)
            with self.assertRaises(ValueError): checker.force_intervals(run,observation)
            checker.write_csv(directory/'forces.csv',checker.FORCES,raw[:-1])
            with self.assertRaises(ValueError): checker.force_intervals(run,observation)


if __name__ == '__main__':
    unittest.main()
