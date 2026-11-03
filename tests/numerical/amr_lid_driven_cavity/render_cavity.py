#!/usr/bin/env pvpython
"""Render cavity native xy cells, interpolated streamlines and Ghia centerlines.

D3 fields are explicitly averaged over native z planes. Only streamline drawing
uses cell-to-point interpolation. Rendering exit 0 is not scientific acceptance.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

from check_cavity import (CavityRun, REFERENCE, centerlines, composite_masks,
                          interfaces, require, write_csv)


def planes(plot):
    result = []
    for lev, (level, mask) in enumerate(zip(plot.levels, composite_masks(plot))):
        mean = level.values.mean(axis=1)
        result.append(dict(valid=mask[0], spacing=level.spacing, plane='xy',
            slice_coordinate=plot.high[2]/2, level=lev, step=level.step,
            coarsest_step=plot.levels[0].step, phase='Sync', time=plot.time,
            ux=mean[1], uy=mean[2], speed=np.sqrt(np.sum(mean[1:]**2, axis=0))))
    return result


def save_window(window, path):
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    window.Render()
    image = vtkWindowToImageFilter(); image.SetInput(window); image.ReadFrontBufferOff(); image.Update()
    writer = vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(image.GetOutputPort()); writer.Write()
    error = writer.GetErrorCode()
    window.Finalize()
    if error:
        raise OSError(f'VTK failed to write {path}: {error}')


def streamlines(path, native, faces, title):
    from pv_native_planes import surface, interface_actor
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkCellDataToPointData
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow, vtkTextActor

    mesh = surface(native, 'speed')
    velocity = np.concatenate([np.column_stack((item['ux'][item['valid']], item['uy'][item['valid']],
                                                np.zeros(int(item['valid'].sum())))) for item in native])
    vectors = numpy_to_vtk(velocity, deep=True); vectors.SetName('velocity_xy'); mesh.GetCellData().SetVectors(vectors)
    clean = vtkCleanPolyData(); clean.SetInputData(mesh)
    points = vtkCellDataToPointData(); points.SetInputConnection(clean.GetOutputPort()); points.Update()
    seeds = vtkPoints()
    for x in np.linspace(.05, .95, 12):
        for y in np.linspace(.05, .95, 12):
            seeds.InsertNextPoint(float(x), float(y), 0)
    source = vtkPolyData(); source.SetPoints(seeds)
    tracer = vtkStreamTracer(); tracer.SetInputConnection(points.GetOutputPort()); tracer.SetSourceData(source)
    tracer.SetInputArrayToProcess(0, 0, 0, 0, 'velocity_xy')
    tracer.SetIntegratorTypeToRungeKutta4(); tracer.SetIntegrationDirectionToBoth()
    tracer.SetMaximumPropagation(5); tracer.SetInitialIntegrationStep(.5); tracer.SetMaximumNumberOfSteps(6000)
    tracer.Update()
    curves = tracer.GetOutput().GetNumberOfCells()
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(tracer.GetOutputPort()); mapper.ScalarVisibilityOff()
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(.12, .28, .64)
    renderer = vtkRenderer(); renderer.SetBackground(1, 1, 1); renderer.AddActor(actor)
    renderer.AddActor(interface_actor(faces, 'xy', float(native[0]['slice_coordinate'])))
    text = vtkTextActor()
    text.SetInput(title + '\nStreamlines: cell-to-point interpolation; xy velocity only' +
                  ('\nFixed seeds produced no curves' if not curves else ''))
    text.SetPosition(20, 710); text.GetTextProperty().SetFontSize(17); text.GetTextProperty().SetColor(.1, .1, .1)
    renderer.AddViewProp(text)
    camera = renderer.GetActiveCamera(); camera.ParallelProjectionOn()
    camera.SetPosition(.5, .5, 3); camera.SetFocalPoint(.5, .5, 0); camera.SetViewUp(0, 1, 0); camera.SetParallelScale(.70)
    window = vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(900, 820); window.AddRenderer(renderer)
    save_window(window, path)
    return curves


def centerline_chart(path, field, computed, reference, title):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkTable
    from vtkmodules.vtkChartsCore import vtkChartXY, vtkChart, vtkAxis, vtkPlotPoints
    from vtkmodules.vtkViewsContext2D import vtkContextView
    import vtkmodules.vtkRenderingContextOpenGL2

    chart = vtkChartXY(); chart.SetTitle(title); chart.SetShowLegend(True)
    chart.GetAxis(vtkAxis.BOTTOM).SetTitle('y/L' if field == 'ux' else 'x/L')
    chart.GetAxis(vtkAxis.LEFT).SetTitle('u_x/U_lid' if field == 'ux' else 'u_y/U_lid')
    tables = []
    series = [('run', computed, (30, 70, 160, 255), False)]
    if reference is not None:
        series.append(('uniform fine', reference, (210, 100, 25, 255), False))
    filename = 'Re1000_x0.5_Ux_Ghia.csv' if field == 'ux' else 'Re1000_y0.5_Uy_Ghia.csv'
    series.append(('Ghia Re1000', np.loadtxt(REFERENCE/filename, delimiter=',', skiprows=1), (30, 30, 30, 255), True))
    for label, values, color, markers in series:
        values = np.asarray(values)
        table = vtkTable()
        for index, name in ((0, 'coordinate'), (1, label)):
            array = numpy_to_vtk(np.ascontiguousarray(values[:, index]), deep=True); array.SetName(name); table.AddColumn(array)
        tables.append(table)
        curve = chart.AddPlot(vtkChart.POINTS if markers else vtkChart.LINE)
        curve.SetInputData(table, 0, 1); curve.SetColor(*color); curve.SetWidth(2)
        if markers:
            curve.SetMarkerStyle(vtkPlotPoints.CROSS); curve.SetMarkerSize(8)
    view = vtkContextView(); view.GetScene().AddItem(chart); view.GetRenderer().SetBackground(1, 1, 1)
    window = view.GetRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1000, 780)
    save_window(window, path)


def render(run_directory, reference_directory, output, steps, speed_limit):
    import paraview
    from pv_native_planes import mosaic

    run = CavityRun(run_directory, complete=False)
    reference = CavityRun(reference_directory, complete=False) if reference_directory else None
    if reference:
        require(reference.layout == 'fine' and run.layout != 'fine' and
                all(run.id[key] == reference.id[key] for key in ('source_id', 'descriptor', 'backend')),
                'unmatched rendering reference')
    available = sorted({step for step, lev in run.samples if lev == 0})
    selected = available if steps is None else steps
    require(selected and selected == sorted(set(selected)) and set(selected) <= set(available), 'invalid requested rendering steps')
    output.mkdir(parents=True, exist_ok=True)
    (output/'render-identity.txt').write_text(
        f'paraview={paraview.__version__}\nsource_id={run.id["source_id"]}\nbackend={run.id["backend"]}\n'
        f'descriptor={run.id["descriptor"]}\nlayout={run.layout}\nrun={run.directory.resolve()}\n'
        f'reference={reference.directory.resolve() if reference else "none"}\n'
        'geometry=native-xy-cell-layout\nmask=actual-uncovered-valid\nz_reduction=arithmetic-native-z-mean\n'
        'speed=norm-of-z-mean-velocity-over-lid-speed\nstreamlines=cell-to-point-xy-visualization-only\n'
        'scientific_gates=not-evaluated\nwall-centerline-endpoints=imposed-not-independent-evidence\n', encoding='utf-8')
    settings = []
    for step in selected:
        plot = run.plot(step)
        native = planes(plot)
        _, faces = interfaces(plot, composite_masks(plot))
        title = f'{run.id["descriptor"]} {run.layout} t*={plot.time:.9g}; native xy / z-mean'
        panels = [(native, 'speed', 'Run: speed / U_lid', (0, speed_limit), False)]
        ref_plot = None
        if reference:
            require(step*reference.n % run.n == 0, 'reference does not share the synchronized frame')
            ref_plot = reference.plot(step*reference.n//run.n)
            label = 'Uniform fine; AMR interface guides' if run.layout == 'amr' else 'Uniform fine'
            panels.append((planes(ref_plot), 'speed', label, (0, speed_limit), False))
        mosaic(output/f'step-{step}-speed.png', panels, title, faces, view=(.5, .5, .72))
        curves = streamlines(output/f'step-{step}-streamlines.png', native, faces, title)
        clipped = sum(int(np.count_nonzero(item['speed'][item['valid']] > speed_limit)) for item in native)
        peak = max(float(item['speed'][item['valid']].max()) for item in native if item['valid'].any())
        settings.append(dict(step=step, time=plot.time, color_max=speed_limit, speed_peak=peak,
                             clipped_cells=clipped, streamline_curves=curves))
        if step == selected[-1]:
            lines = centerlines(plot, composite_masks(plot))
            ref_lines = centerlines(ref_plot, composite_masks(ref_plot)) if ref_plot is not None else None
            for field in ('ux', 'uy'):
                centerline_chart(output/f'centerline-{field}.png', field, lines[field],
                                 ref_lines[field] if ref_lines else None,
                                 title+'\nPhysical wall endpoints imposed; Ghia comparison is quantitative in checker')
        print(f'cavity rendered step={step} t*={plot.time:.9g}', flush=True)
    write_csv(output/'render-settings.csv', tuple(settings[0]), settings)
    (output/'render-result.txt').write_text(f'exit=0\nframes={len(selected)}\nimage_review=pending\nscientific_acceptance=not-evaluated\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', nargs='+', type=int)
    parser.add_argument('--speed-limit', type=float, default=1.)
    args = parser.parse_args()
    try:
        require(math.isfinite(args.speed_limit) and args.speed_limit > 0, 'invalid speed color limit')
        render(args.run, args.reference, args.output, args.steps, args.speed_limit)
        return 0
    except (OSError, ValueError, KeyError, IndexError, ImportError, RuntimeError, csv.Error) as error:
        print(f'INVALID cavity render: {error}', file=sys.stderr)
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'render-result.txt').write_text(f'exit=2\nerror={error}\n')
        except OSError:
            pass
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
