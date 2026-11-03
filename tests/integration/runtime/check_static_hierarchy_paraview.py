#!/usr/bin/env pvpython
"""Read real AMReX output with ParaView 6.1.0; run with its bundled pvpython."""

import argparse
import csv
import math
from pathlib import Path

import paraview
from paraview import simple
from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes

from static_hierarchy_artifacts import (NumericalMismatch, load_plotfile_expectations)


VARIABLES = ("rho_norm", "ux_norm", "uy_norm", "uz_norm")
SAMPLES = ("G2", "G3-P", "P-through-2", "P-through-3", "P3-P", "P-corner", "S0-step2",
           "D2-Topo-Gap-8", "D2-Topo-PeriodicFace-Seam", "D2-Topo-Finest")
D3_SAMPLES = ("D3-G2-P", "D3-G3-P", "D3-Vertex-P", "D3-T2-z-P",
              "D3-T3-z-P", "D3-Slab-z-P", "D3-Full-P", "D3-S0-step2",
              "D3-Topo-Union", "D3-Topo-Vertex", "D3-Topo-Through-z", "D3-Topo-Sparse-L1-L3",
              "D3-Topo-Sparse-L2-L3", "D3-Topo-TightCoarse", "D3-Topo-Feedback")


class ReadbackError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReadbackError(message)


def close(actual, expected, message):
    require(math.isfinite(actual) and math.isfinite(expected) and
            abs(actual-expected) <= max(1e-12, 1e-12*max(abs(actual), abs(expected))),
            f"{message}: actual={actual:.17g} expected={expected:.17g}")


def check_sample(directory, report, source_id=None, descriptor="D2Q9"):
    geometry, expected = load_plotfile_expectations(directory,source_id)
    identity = geometry.identity
    time = geometry.times[geometry.steps[0]]
    levels = int(identity["levels"])
    cells = tuple(map(int, identity["base_cells"].split(",")))
    lo = tuple(map(float, identity["prob_lo"].split(",")))
    hi = tuple(map(float, identity["prob_hi"].split(",")))
    ratio = (2,2,1) if descriptor == "D2Q9" else (2,2,2)
    dimensions, q = (2,9) if descriptor == "D2Q9" else (3,27)
    require(identity["descriptor"] == descriptor and int(identity["d"]) == dimensions and
            int(identity["q"]) == q, "output descriptor identity differs")
    require(tuple(map(int,identity["ref_ratio"].split(","))) == ratio and
            (dimensions == 3 or cells[2] == 1), "unsupported fixture topology")
    require((directory / "plot" / "Header").is_file(), "plotfile Header missing")
    # The reader takes the plotfile directory. A bare Header has no extension
    # for OpenDataFile to identify, and Fetch's generic grouping cannot carry
    # this AMR type. pvpython's built-in connection exposes the reader output.
    reader = simple.AMReXBoxLibGridReader(FileNames=[(directory / "plot").as_posix()])
    reader.UpdatePipelineInformation()
    require(reader.SMProxy.GetXMLName() == "AMReXGridReader", "unexpected reader proxy")
    require(set(reader.CellArrayStatus.Available) == set(VARIABLES), "reader variable list differs")
    reader.CellArrayStatus = list(VARIABLES)
    reader.Level = levels - 1
    require(len(reader.TimestepValues) == 1, "unexpected reader time sequence")
    close(float(reader.TimestepValues[0]), time, "reader time")
    reader.UpdatePipeline(time=time)
    data = reader.GetClientSideObject().GetOutputDataObject(0)
    require(data is not None and data.IsA("vtkOverlappingAMR"), "reader did not produce native AMR")
    require(data.GetNumberOfLevels() == levels, "reader lost a level")
    require({key[0] for key in expected} == set(range(levels)), "oracle lost a level")
    hidden = {(level-1,i//ratio[0],j//ratio[1],k//ratio[2]) for level,i,j,k in expected if level > 0}
    actual = set()
    hidden_count = 0
    mask_errors = 0
    first_mask_error = None
    previous_spacing = None
    volumes, masses, expected_volumes, expected_masses = [], [], [], []
    rho_reference = float(identity["rho_reference"])
    for level in range(levels):
        spacing = tuple((hi[d]-lo[d]) / (cells[d]*ratio[d]**level) for d in range(3))
        cell_volume = math.prod(spacing)
        require(data.GetNumberOfBlocks(level) > 0, "empty reader level")
        for block in range(data.GetNumberOfBlocks(level)):
            grid = data.GetDataSetAsImageData(level, block)
            require(grid is not None, "reader left an unloaded block")
            extent, origin, actual_spacing, bounds = grid.GetExtent(), grid.GetOrigin(), grid.GetSpacing(), grid.GetBounds()
            actual_cell_volume = math.prod(actual_spacing)
            if block == 0:
                if previous_spacing is not None:
                    for d, refinement in enumerate(ratio):
                        close(previous_spacing[d]/actual_spacing[d], refinement, "directional reader refinement ratio")
                previous_spacing = actual_spacing
            sizes = tuple(extent[2*d+1]-extent[2*d] for d in range(3))
            require(min(sizes) > 0 and (dimensions == 3 or sizes[2] == 1), "reader changed cell centering or Nz")
            for d in range(3):
                close(actual_spacing[d], spacing[d], "block directional spacing")
                close(bounds[2*d], origin[d]+extent[2*d]*spacing[d], "block lower bound")
                close(bounds[2*d+1], origin[d]+extent[2*d+1]*spacing[d], "block upper bound")
            if dimensions == 2:
                close(bounds[4],lo[2],"block lower z")
                close(bounds[5],hi[2],"block upper z")
            cell_data = grid.GetCellData()
            arrays = [cell_data.GetArray(name) for name in VARIABLES]
            count = grid.GetNumberOfCells()
            require(count == sizes[0]*sizes[1]*sizes[2], "block cell count differs from extent")
            require(all(a is not None and a.GetNumberOfComponents() == 1 and a.GetNumberOfTuples() == count
                        for a in arrays), "reader field shape mismatch")
            ghosts = cell_data.GetArray("vtkGhostType")
            visible_density, expected_density = [], []
            for cell in range(count):
                local = (cell % sizes[0], (cell // sizes[0]) % sizes[1], cell // (sizes[0]*sizes[1]))
                indices = []
                for d in range(3):
                    center = origin[d] + (extent[2*d]+local[d]+0.5)*actual_spacing[d]
                    index = (center-lo[d])/spacing[d]-0.5
                    rounded = round(index)
                    close(index, rounded, "reader cell center is off the prescribed lattice")
                    indices.append(rounded)
                key = (level, *indices)
                require(key in expected and key not in actual, f"missing/duplicate/unexpected reader cell: {key}")
                actual.add(key)
                for component, array in enumerate(arrays):
                    close(array.GetComponent(cell, 0), expected[key][component], f"{VARIABLES[component]} {key}")
                flags = int(ghosts.GetComponent(cell, 0)) if ghosts is not None else 0
                masked = bool(flags & (vtkDataSetAttributes.REFINEDCELL | vtkDataSetAttributes.HIDDENCELL))
                if masked != (key in hidden):
                    mask_errors += 1
                    if first_mask_error is None:
                        first_mask_error = key
                require(not flags & vtkDataSetAttributes.DUPLICATECELL, "plotfile contains duplicate ghost cells")
                hidden_count += masked
                if not masked:
                    visible_density.append(arrays[0].GetComponent(cell,0))
                if key not in hidden:
                    expected_density.append(expected[key][0])
            volumes.append(len(visible_density)*actual_cell_volume)
            masses.append(math.fsum(visible_density)*rho_reference*actual_cell_volume)
            expected_volumes.append(len(expected_density)*cell_volume)
            expected_masses.append(math.fsum(expected_density)*rho_reference*cell_volume)
            report.writerow([directory.name, level, block, *extent, *origin, *actual_spacing, *bounds, count])
    require(actual == expected.keys(), "reader coverage is incomplete")
    volume, mass = math.fsum(volumes), math.fsum(masses)
    expected_volume, expected_mass = math.fsum(expected_volumes), math.fsum(expected_masses)
    close(expected_volume,math.prod(b-a for a,b in zip(lo,hi)),"oracle composite volume")
    integral_ok = all(abs(a-b) <= max(1e-12,1e-12*max(abs(a),abs(b)))
                      for a,b in ((volume,expected_volume),(mass,expected_mass)))
    print(f"paraview_composite sample={directory.name} volume={volume:.17g} expected_volume={expected_volume:.17g} "
          f"mass={mass:.17g} expected_mass={expected_mass:.17g} status={'PASS' if integral_ok else 'FAIL'}")
    if mask_errors == 0:
        require(integral_ok,"reader composite integrals differ from independent coverage/fields")
    print(f"paraview_readback sample={directory.name} proxy=AMReXGridReader levels={levels} "
          f"cells={len(actual)} hidden_coarse_cells={hidden_count} expected_hidden={len(hidden)} "
          f"geometry_fields=PASS visibility_mask={'FAIL' if mask_errors else 'PASS'} "
          f"mask_errors={mask_errors} first_mask_error={first_mask_error}")
    return reader, mask_errors


def render(reader, path, descriptor):
    view = simple.CreateView("RenderView")
    view.ViewSize = [1000, 1000]
    display = simple.Show(reader, view)
    display.Representation = "Surface With Edges"
    simple.ColorBy(display, ("CELLS", "rho_norm"))
    display.RescaleTransferFunctionToDataRange(False, True)
    view.CameraParallelProjection = 1
    view.CameraPosition = [0.5,0.5,3.0] if descriptor == "D2Q9" else [2.2,1.8,1.9]
    view.CameraFocalPoint = [0.5,0.5,0.03125] if descriptor == "D2Q9" else [0.5,0.5,0.5]
    view.CameraViewUp = [0,1,0] if descriptor == "D2Q9" else [0,0,1]
    view.CameraParallelScale = 0.55 if descriptor == "D2Q9" else 0.9
    simple.SaveScreenshot(path.as_posix(), view, ImageResolution=[1000, 1000])
    simple.Delete(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="C++ output producer's artifacts/output directory")
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--samples", nargs="+", help="Explicit complete list for the selected output set")
    parser.add_argument("--source-id")
    parser.add_argument("--descriptor", choices=("D2Q9","D3Q27"), default="D2Q9")
    args = parser.parse_args()
    try:
        require(paraview.__version__ == "6.1.0", f"expected ParaView 6.1.0, got {paraview.__version__}")
        args.report_dir.mkdir(parents=True, exist_ok=True)
        mask_errors = 0
        with (args.report_dir / "reader-blocks.csv").open("w", newline="") as stream:
            report = csv.writer(stream)
            report.writerow(["sample", "level", "block", "ilo", "ihi", "jlo", "jhi", "klo", "khi",
                             "origin_x", "origin_y", "origin_z", "dx", "dy", "dz",
                             "xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "cells"])
            samples = args.samples or (SAMPLES if args.descriptor == "D2Q9" else D3_SAMPLES)
            require(len(samples) == len(set(samples)), "duplicate requested sample")
            for sample in samples:
                reader, mismatches = check_sample(args.root / sample,report,args.source_id,args.descriptor)
                mask_errors += mismatches
                if args.render:
                    render(reader,args.report_dir / f"{sample}.png",args.descriptor)
                simple.Delete(reader)
        print(f"ParaView {paraview.__version__}: {len(samples)} fixtures checked; visibility_mask_errors={mask_errors}")
        return int(mask_errors != 0)
    except (ReadbackError, NumericalMismatch, RuntimeError) as error:
        print(f"paraview_readback mismatch: {error}")
        return 1
    except (OSError, ValueError, KeyError, csv.Error) as error:
        print(f"paraview_readback artifact_error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
