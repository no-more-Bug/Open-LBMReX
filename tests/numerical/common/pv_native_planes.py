"""pvpython helpers for exact native cell-plane artifacts, with explicit masks."""
from pathlib import Path
import csv
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import (vtkActor, vtkColorTransferFunction, vtkPolyDataMapper,
    vtkRenderer, vtkRenderWindow, vtkTextActor, vtkWindowToImageFilter)
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkIOImage import vtkPNGWriter
import vtkmodules.vtkRenderingOpenGL2  # register the offscreen renderer
import vtkmodules.vtkRenderingFreeType


def surface(planes, field):
    points, cells, values = [], [], []
    offset = 0
    for plane in planes:
        mask = plane['valid']; y, x = np.nonzero(mask)
        spacing = plane['spacing']
        origin = plane.get('origin', (0., 0., 0.))
        vertical = 1 if str(plane['plane']) == 'xy' else 2
        dx, dy = spacing[0], spacing[1 if str(plane['plane']) == 'xy' else 2]
        corners = np.empty((len(x), 4, 3))
        for q, (ox, oy) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            corners[:, q, 0] = origin[0]+(x+ox)*dx; corners[:, q, 1] = origin[vertical]+(y+oy)*dy; corners[:, q, 2] = 0
        points.append(corners.reshape(-1, 3))
        indices = np.arange(offset, offset+4*len(x), dtype=np.int64).reshape(-1, 4)
        cells.append(np.column_stack((np.full(len(x), 4, dtype=np.int64), indices)))
        offset += 4*len(x)
        values.append(plane[field][mask])
    mesh = vtkPolyData(); pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.concatenate(points), deep=True)); mesh.SetPoints(pts)
    polygons = vtkCellArray(); polygons.SetCells(sum(len(v) for v in values),
        numpy_to_vtkIdTypeArray(np.concatenate(cells).ravel(), deep=True)); mesh.SetPolys(polygons)
    array = numpy_to_vtk(np.concatenate(values), deep=True); array.SetName(field)
    mesh.GetCellData().SetScalars(array)
    return mesh


def interface_actor(faces, plane, position):
    shown = (0, 1) if plane == 'xy' else (0, 2)
    hidden = 3-sum(shown)
    points = vtkPoints(); lines = vtkCellArray()
    for face in faces:
        lo, hi = face[1], face[2]
        if not lo[hidden]-1e-12 <= position <= hi[hidden]+1e-12:
            continue
        a = [lo[shown[0]], lo[shown[1]], 1e-5]
        b = [hi[shown[0]], hi[shown[1]], 1e-5]
        if a == b:
            continue
        start = points.InsertNextPoint(a); end = points.InsertNextPoint(b)
        lines.InsertNextCell(2); lines.InsertCellPoint(start); lines.InsertCellPoint(end)
    mesh = vtkPolyData(); mesh.SetPoints(points); mesh.SetLines(lines)
    mapper = vtkPolyDataMapper(); mapper.SetInputData(mesh)
    actor = vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(.12, .12, .12)
    actor.GetProperty().SetLineWidth(1.2)
    return actor


def mosaic(path, panels, title, faces=(), mesh_edges=False, view=(.5, .5, .64)):
    """Panels: (native-plane-list, field, label, (min,max), signed)."""
    cols = min(3, len(panels)); rows = (len(panels)+cols-1)//cols
    window = vtkRenderWindow(); window.SetOffScreenRendering(1)
    window.SetSize(800*cols, 780*rows)
    settings = []
    for i, (planes, field, label, limits, signed) in enumerate(panels):
        renderer = vtkRenderer(); renderer.SetBackground(1, 1, 1)
        row, col = divmod(i, cols)
        renderer.SetViewport(col/cols, 1-(row+1)/rows, (col+.84)/cols, 1-row/rows)
        window.AddRenderer(renderer)
        legend = vtkRenderer(); legend.SetBackground(1, 1, 1)
        legend.SetViewport((col+.84)/cols, 1-(row+1)/rows, (col+1)/cols, 1-row/rows)
        window.AddRenderer(legend)
        mesh = surface(planes, field)
        color = vtkColorTransferFunction()
        if signed:
            color.AddRGBPoint(limits[0], .12, .28, .64); color.AddRGBPoint(0, .96, .96, .96)
            color.AddRGBPoint(limits[1], .75, .16, .12)
        else:
            color.AddRGBPoint(limits[0], .97, .97, .97); color.AddRGBPoint(limits[1], .12, .28, .64)
        mapper = vtkPolyDataMapper(); mapper.SetInputData(mesh); mapper.SetLookupTable(color)
        mapper.SetScalarRange(*limits); mapper.SetScalarModeToUseCellData(); mapper.InterpolateScalarsBeforeMappingOff()
        actor = vtkActor(); actor.SetMapper(mapper)
        if mesh_edges:
            actor.GetProperty().EdgeVisibilityOn(); actor.GetProperty().SetEdgeColor(.3, .3, .3)
        renderer.AddActor(actor)
        first = planes[0]
        settings.append(dict(image=Path(path).name, panel=i, field=field, label=label, color_min=limits[0], color_max=limits[1],
            plane=str(first['plane']), slice_coordinate=float(first['slice_coordinate']), time=float(first['time']),
            camera_x=view[0], camera_y=view[1], parallel_scale=view[2],
            levels=';'.join(str(int(p['level'])) for p in planes), steps=';'.join(str(p.get('step', '')) for p in planes),
            coarsest_steps=';'.join(str(p.get('coarsest_step', '')) for p in planes), phases=';'.join(str(p.get('phase', '')) for p in planes)))
        renderer.AddActor(interface_actor(faces, str(first['plane']), float(first['slice_coordinate'])))
        bar = vtkScalarBarActor(); bar.SetLookupTable(color); bar.SetNumberOfLabels(4)
        bar.SetPosition(.08, .14); bar.SetWidth(.84); bar.SetHeight(.64)
        bar.GetLabelTextProperty().SetColor(.1, .1, .1); bar.GetLabelTextProperty().SetFontSize(16)
        legend.AddActor2D(bar)
        text = vtkTextActor(); text.SetInput(f'{title}\n{label}')
        text.SetPosition(18, 690); text.GetTextProperty().SetFontSize(18); text.GetTextProperty().SetColor(.1, .1, .1)
        renderer.AddActor2D(text)
        camera = renderer.GetActiveCamera(); camera.ParallelProjectionOn()
        camera.SetPosition(view[0], view[1], 3); camera.SetFocalPoint(view[0], view[1], 0); camera.SetViewUp(0, 1, 0); camera.SetParallelScale(view[2])
    # Keep unused cells of a multirow mosaic white instead of the window's black default.
    for i in range(len(panels), rows*cols):
        renderer = vtkRenderer(); renderer.SetBackground(1, 1, 1)
        row, col = divmod(i, cols)
        renderer.SetViewport(col/cols, 1-(row+1)/rows, (col+1)/cols, 1-row/rows)
        window.AddRenderer(renderer)
    window.Render()
    image = vtkWindowToImageFilter(); image.SetInput(window); image.ReadFrontBufferOff(); image.Update()
    writer = vtkPNGWriter(); writer.SetFileName(str(Path(path))); writer.SetInputConnection(image.GetOutputPort()); writer.Write()
    error = writer.GetErrorCode()
    window.Finalize()
    if error:
        raise OSError(f"VTK PNG write failed with error code {error}: {path}")
    with Path(path).with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(settings[0])); writer.writeheader(); writer.writerows(settings)
