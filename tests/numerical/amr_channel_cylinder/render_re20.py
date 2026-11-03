#!/usr/bin/env pvpython
"""Render native Re20 fields and case-owned observations using pvpython.

The physical Guo channel is clipped at its face nodes. Native cells retain their
real envelope offset; z means and streamline interpolation are display choices.
Rendering success is not scientific acceptance.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np

from check_re20 import (Re20Run, ULB, DEPTH, LINES, fluid_masks, composite_masks,
                        profiles, rows, require, write_csv)


def planes(plot):
    result = []
    for lev, (data, mask) in enumerate(zip(plot.levels, fluid_masks(plot))):
        mean = data.values.mean(axis=1)
        result.append(dict(valid=mask[0], spacing=data.spacing, plane='xy', slice_coordinate=DEPTH/2,
            level=lev, step=data.step, time=plot.time, ux=mean[1], uy=mean[2],
            speed=np.linalg.norm(mean[1:], axis=0), pressure=(mean[0]-1)/(3*ULB**2)))
    return result


def interface_faces(plot):
    masks = composite_masks(plot); result = []
    for lev in range(len(masks)-1):
        covered = plot.levels[lev].valid[0] & ~masks[lev][0]
        spacing = plot.levels[lev].spacing
        for axis in (0,1):
            a = [slice(None),slice(None)]; b = a.copy()
            a[1-axis] = slice(None,-1); b[1-axis] = slice(1,None)
            for j,i in zip(*np.nonzero(covered[tuple(a)] != covered[tuple(b)])):
                lo = np.array([plot.low[0]+i*spacing[0],plot.low[1]+j*spacing[1],0.])
                hi = lo+np.array([spacing[0],spacing[1],DEPTH])
                lo[axis] += spacing[axis]; hi[axis] = lo[axis]
                result.append((axis,lo,hi))
    return result


def native_surface(plot, native, field):
    from pv_native_planes import surface
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkBox
    from vtkmodules.vtkFiltersCore import vtkClipPolyData

    mesh = surface(native,field)
    # The shared surface uses zero-based lattice faces; apply this run's envelope.
    points = vtk_to_numpy(mesh.GetPoints().GetData()); points[:,0] += plot.low[0]; points[:,1] += plot.low[1]
    mesh.GetPoints().Modified()
    velocity = np.concatenate([np.column_stack((p['ux'][p['valid']],p['uy'][p['valid']],
                                                np.zeros(int(p['valid'].sum())))) for p in native])
    vectors = numpy_to_vtk(velocity,deep=True); vectors.SetName('velocity_xy'); mesh.GetCellData().SetVectors(vectors)
    bounds = vtkBox(); bounds.SetBounds(0,2.2,0,.41,-1,1)
    clip = vtkClipPolyData(); clip.SetInputData(mesh); clip.SetClipFunction(bounds); clip.InsideOutOn(); clip.Update()
    return clip.GetOutput()


def save_window(window,path):
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    window.Render()
    image = vtkWindowToImageFilter(); image.SetInput(window); image.ReadFrontBufferOff(); image.Update()
    writer = vtkPNGWriter(); writer.SetFileName(str(path)); writer.SetInputConnection(image.GetOutputPort()); writer.Write()
    error = writer.GetErrorCode(); window.Finalize()
    if error: raise OSError(f'VTK PNG write failed: {path}: {error}')


def decorate(renderer, faces, title):
    from pv_native_planes import interface_actor
    from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
    from vtkmodules.vtkRenderingCore import vtkActor,vtkPolyDataMapper,vtkTextActor
    renderer.AddActor(interface_actor(faces,'xy',DEPTH/2))
    circle = vtkRegularPolygonSource(); circle.SetCenter(.2,.2,2e-5); circle.SetRadius(.05); circle.SetNumberOfSides(180)
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(circle.GetOutputPort())
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(1,1,1)
    actor.GetProperty().EdgeVisibilityOn(); actor.GetProperty().SetEdgeColor(.15,.15,.15); renderer.AddActor(actor)
    text = vtkTextActor(); text.SetInput(title); text.SetPosition(20,520)
    text.GetTextProperty().SetFontSize(18); text.GetTextProperty().SetColor(.1,.1,.1); renderer.AddViewProp(text)
    camera = renderer.GetActiveCamera(); camera.ParallelProjectionOn()
    camera.SetPosition(1.1,.205,3); camera.SetFocalPoint(1.1,.205,0); camera.SetViewUp(0,1,0); camera.SetParallelScale(.50)


def field_image(path, panels, field, limits):
    from vtkmodules.vtkRenderingCore import (vtkActor,vtkColorTransferFunction,vtkPolyDataMapper,
        vtkRenderer,vtkRenderWindow)
    from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
    window = vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1600,600*len(panels))
    for index,(run,plot,native) in enumerate(panels):
        bottom,top = 1-(index+1)/len(panels),1-index/len(panels)
        renderer = vtkRenderer(); renderer.SetBackground(1,1,1); renderer.SetViewport(0,bottom,.85,top)
        window.AddRenderer(renderer)
        color = vtkColorTransferFunction(); color.AddRGBPoint(limits[0],.12,.28,.64)
        color.AddRGBPoint((limits[0]+limits[1])/2,.95,.95,.95); color.AddRGBPoint(limits[1],.75,.16,.12)
        mapper = vtkPolyDataMapper(); mapper.SetInputData(native_surface(plot,native,field)); mapper.SetLookupTable(color)
        mapper.SetScalarRange(*limits); mapper.SetScalarModeToUseCellData(); mapper.InterpolateScalarsBeforeMappingOff()
        actor = vtkActor(); actor.SetMapper(mapper); renderer.AddActor(actor)
        label = '|mean_z u| / Ubar' if field == 'speed' else '(mean_z rho - 1) / (3 Ubar_lb^2)'
        decorate(renderer,interface_faces(plot),
            f'{run.id["descriptor"]} {run.id["scheme"]} {run.layout}; t={plot.time:.9g}\n{label}; native xy cells / z mean; physical Guo channel')
        legend = vtkRenderer(); legend.SetBackground(1,1,1); legend.SetViewport(.85,bottom,1,top); window.AddRenderer(legend)
        bar = vtkScalarBarActor(); bar.SetLookupTable(color); bar.SetNumberOfLabels(5); bar.SetPosition(.12,.20)
        bar.SetWidth(.65); bar.SetHeight(.60); bar.GetLabelTextProperty().SetColor(.1,.1,.1); legend.AddViewProp(bar)
    save_window(window,path)


def streamline_image(path,run,plot,native):
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData,vtkCellDataToPointData
    from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
    from vtkmodules.vtkRenderingCore import vtkActor,vtkPolyDataMapper,vtkRenderer,vtkRenderWindow
    clean = vtkCleanPolyData(); clean.SetInputData(native_surface(plot,native,'speed'))
    interpolate = vtkCellDataToPointData(); interpolate.SetInputConnection(clean.GetOutputPort())
    points = vtkPoints()
    for y in np.linspace(.01,.40,32): points.InsertNextPoint(.01,float(y),0)
    for x in np.linspace(.27,1.,10):
        for y in np.linspace(.08,.32,9): points.InsertNextPoint(float(x),float(y),0)
    seeds = vtkPolyData(); seeds.SetPoints(points)
    tracer = vtkStreamTracer(); tracer.SetInputConnection(interpolate.GetOutputPort()); tracer.SetSourceData(seeds)
    tracer.SetInputArrayToProcess(0,0,0,0,'velocity_xy'); tracer.SetIntegratorTypeToRungeKutta4()
    tracer.SetIntegrationDirectionToBoth(); tracer.SetMaximumPropagation(5); tracer.SetInitialIntegrationStep(.5)
    tracer.SetMaximumNumberOfSteps(12000); tracer.Update()
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(tracer.GetOutputPort()); mapper.ScalarVisibilityOff()
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(.12,.28,.64)
    renderer = vtkRenderer(); renderer.SetBackground(1,1,1); renderer.AddActor(actor)
    curves = tracer.GetOutput().GetNumberOfCells()
    decorate(renderer,interface_faces(plot),
        f'{run.id["descriptor"]} {run.id["scheme"]} {run.layout}; t={plot.time:.9g}\n'
        'Streamlines: z-mean xy, cell-to-point interpolation, fixed seeds'+(' (no curves)' if not curves else ''))
    window = vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1600,600); window.AddRenderer(renderer)
    save_window(window,path)
    return curves


def chart(path,title,xlabel,ylabel,series,log=False):
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkTable
    from vtkmodules.vtkChartsCore import vtkChartXY,vtkChart,vtkAxis,vtkPlotPoints
    from vtkmodules.vtkViewsContext2D import vtkContextView
    import vtkmodules.vtkRenderingContextOpenGL2
    single = all(len(x) == 1 for _,x,_ in series)
    display_log = log and not single
    note = '\nSingle observation / linear axis' if single else '\nLog axis; display floor 1e-18' if log else ''
    chart = vtkChartXY(); chart.SetTitle(title+note); chart.SetShowLegend(True)
    chart.GetAxis(vtkAxis.BOTTOM).SetTitle(xlabel); chart.GetAxis(vtkAxis.LEFT).SetTitle(ylabel)
    if display_log: chart.GetAxis(vtkAxis.LEFT).SetLogScale(True)
    if single:
        center = float(np.mean([x[0] for _,x,_ in series]))
        padding = max(abs(center)*.05,1e-6)
        chart.GetAxis(vtkAxis.BOTTOM).SetBehavior(vtkAxis.FIXED)
        chart.GetAxis(vtkAxis.BOTTOM).SetRange(center-padding,center+padding)
    tables = []
    colors = ((30,70,160,255),(210,100,25,255),(40,130,70,255))
    for index,(name,x,y) in enumerate(series):
        table = vtkTable()
        for label,value in ((xlabel,x),(name,np.maximum(y,1e-18) if display_log else y)):
            array = numpy_to_vtk(np.ascontiguousarray(value,dtype=float),deep=True); array.SetName(label); table.AddColumn(array)
        tables.append(table)
        curve = chart.AddPlot(vtkChart.POINTS if len(x) == 1 else vtkChart.LINE)
        curve.SetInputData(table,0,1); curve.SetColor(*colors[index%len(colors)]); curve.SetWidth(2)
        if len(x) == 1: curve.SetMarkerStyle(vtkPlotPoints.CIRCLE); curve.SetMarkerSize(9)
    view = vtkContextView(); view.GetRenderer().SetBackground(1,1,1); view.GetScene().AddItem(chart)
    window = view.GetRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1200,800); save_window(window,path)


def render(args):
    import paraview
    import vtkmodules.vtkRenderingOpenGL2
    import vtkmodules.vtkRenderingFreeType
    run = Re20Run(args.run,complete=False)
    runs = [run]
    for path,layout in ((args.coarse_reference,'coarse'),(args.reference,'fine')):
        if path is None: continue
        other = Re20Run(path,complete=False)
        require(other.layout == layout and other.layout != run.layout and
                all(other.id[k] == run.id[k] for k in ('source_id','backend','descriptor','scheme')),
                'wrong rendering reference identity')
        runs.append(other)
    available = sorted({step for step,lev in run.samples if lev == 0})
    selected = [run.requested_steps] if args.steps is None else args.steps
    require(selected and selected == sorted(set(selected)) and set(selected) <= set(available), 'invalid rendering steps')
    args.output.mkdir(parents=True,exist_ok=True)
    settings = []; final_profiles = []
    for step in selected:
        panels = []
        for item in runs:
            require(step*item.factor % run.factor == 0,'unmatched physical frame')
            plot = item.plot(step*item.factor//run.factor); native = planes(plot); panels.append((item,plot,native))
        field_image(args.output/f'step-{step}-speed.png',panels,'speed',(0,args.speed_limit))
        field_image(args.output/f'step-{step}-pressure.png',panels,'pressure',args.pressure_limits)
        for item,plot,native in panels:
            curves = streamline_image(args.output/f'step-{step}-{item.layout}-streamlines.png',item,plot,native)
            for field,limits in (('speed',(0,args.speed_limit)),('pressure',args.pressure_limits)):
                values = np.concatenate([p[field][p['valid']] for p in native])
                settings.append(dict(step=plot.levels[0].step,time=plot.time,layout=item.layout,field=field,
                    minimum=float(values.min()),maximum=float(values.max()),color_min=limits[0],color_max=limits[1],
                    clipped_cells=int(np.count_nonzero((values<limits[0])|(values>limits[1]))),streamline_curves=curves))
            if step == selected[-1]: final_profiles.append((item.layout,profiles(plot,fluid_masks(plot))[0]))
    for name in LINES:
        for component,label in ((1,'ux'),(2,'uy')):
            chart(args.output/f'profile-{name}-{label}.png',f'{name}; native physical-coordinate profiles',
                  'y [physical]' if name.startswith('x') else 'x [physical]',label+'/Ubar',
                  [(layout,data[name][:,0],data[name][:,component]) for layout,data in final_profiles])
    histories = [(item.layout,rows(item.directory/'observations.csv')) for item in runs]
    for field,title,log in (('Cd','Drag: common-interval average, plotted at interval endpoint',False),
            ('Cl','Lift: common-interval average, plotted at interval endpoint',False),
            ('Cdp','Synchronized surface pressure difference',False),
            ('phi_in','Published inlet mass flux / nominal inlet flux',False),
            ('phi_out','Published outlet mass flux / nominal inlet flux',False),
            ('relative_l2','Common-interval residual',True),
            ('flux_imbalance','Published inlet/outlet flux imbalance',True)):
        series = []
        for layout,records in histories:
            selected_rows = [r for r in records if r[field] != '']
            series.append((layout,np.array([float(r['time']) for r in selected_rows]),
                           np.array([float(r[field]) for r in selected_rows])))
        chart(args.output/f'history-{field}.png',title,'t [physical]',field,series,log)
    write_csv(args.output/'render-settings.csv',tuple(settings[0]),settings)
    (args.output/'render-identity.txt').write_text(
        f'paraview={paraview.__version__}\nsource_id={run.id["source_id"]}\nbackend={run.id["backend"]}\n'
        f'descriptor={run.id["descriptor"]}\nscheme={run.id["scheme"]}\n'
        'coordinates=native-envelope-shifted\nphysical_clip=0,2.2,0,.41\n'
        'mask=uncovered-fluid-centers-plus-analytic-cylinder-display\nz_reduction=arithmetic-native-mean\n'
        'pressure_gauge=rho_reference_1\nstreamlines=cell-to-point-xy-visualization-only\n'
        'scientific_acceptance=not-evaluated\n',encoding='utf-8')
    (args.output/'render-result.txt').write_text(f'exit=0\nframes={len(selected)}\nimage_review=pending\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--coarse-reference',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--steps',nargs='+',type=int)
    parser.add_argument('--speed-limit',type=float,default=2.5)
    parser.add_argument('--pressure-limits',nargs=2,type=float,default=(-1.,5.))
    args = parser.parse_args()
    try:
        require(math.isfinite(args.speed_limit) and args.speed_limit > 0 and
                all(math.isfinite(x) for x in args.pressure_limits) and args.pressure_limits[0]<args.pressure_limits[1],
                'invalid display limits')
        render(args); return 0
    except (OSError,ValueError,KeyError,IndexError,TypeError,ImportError,RuntimeError,csv.Error) as error:
        print(f'INVALID Re20 render: {error}',file=sys.stderr)
        try:
            args.output.mkdir(parents=True,exist_ok=True)
            (args.output/'render-result.txt').write_text(f'exit=2\nerror={error}\n',encoding='utf-8')
        except OSError:
            pass
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
