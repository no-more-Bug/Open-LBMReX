"""Read the frozen AMReX cell-centered FP64 plotfiles, without resampling.

NumPy is used for bounded dense arrays per native level. Missing valid coverage
stays masked; covered coarse cells are removed only when forming a composite.
This is a reader for the project's writer, not a general plotfile converter.
"""
from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re
import numpy as np

VARIABLES = ('rho_norm', 'ux_norm', 'uy_norm', 'uz_norm')
BOX = re.compile(r'\(\(([-\d, ]+)\)\s*\(([-\d, ]+)\)\s*\(([-\d, ]+)\)\)')


class ArtifactError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def identity(directory, filename="identity.txt"):
    result = {}
    for line in (Path(directory)/filename).read_text().splitlines():
        key, separator, value = line.partition('=')
        require(separator and key not in result and value, 'invalid run identity')
        result[key] = value
    return result


def rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'missing/duplicate CSV columns')
        result = list(reader)
        require(all(None not in row and all(value is not None for value in row.values()) for row in result),
                'incomplete/extra CSV fields')
        return result


def box(text):
    match = BOX.search(text)
    require(match is not None, 'missing cell box')
    lo, hi, kind = [tuple(map(int, s.replace(',', ' ').split())) for s in match.groups()]
    require(len(lo) == len(hi) == 3 and kind == (0, 0, 0), 'expected cell-centered 3D storage')
    return lo, tuple(v+1 for v in hi)


@dataclass
class Level:
    values: np.ndarray  # component,z,y,x; velocity normalization belongs to the case
    valid: np.ndarray  # z,y,x
    spacing: tuple     # physical x,y,z
    step: int
    boxes: list        # inclusive lower, exclusive upper indices


@dataclass
class Plot:
    time: float
    low: tuple
    high: tuple
    levels: list


def read_plot(path):
    try:
        return _read_plot(path)
    except (StopIteration, IndexError) as error:
        raise ArtifactError('truncated native plotfile header') from error


def vismf_header(path, count):
    lines = iter(line.strip() for line in path.read_text().splitlines() if line.strip())
    require(int(next(lines)) == 1 and int(next(lines)) in (0, 1), 'expected VisMF v1')
    require(int(next(lines)) == 4 and next(lines) == '0', 'expected four zero-grow fields')
    declaration = next(lines)
    require(re.fullmatch(r'\(\s*'+str(count)+r'\s+0', declaration) is not None, 'wrong BoxArray count')
    boxes = [box(next(lines)) for _ in range(count)]
    require(next(lines) == ')' and int(next(lines)) == count, 'wrong FAB count')
    files = [next(lines).split() for _ in range(count)]
    require(all(len(row) == 3 and row[0] == 'FabOnDisk:' and int(row[2]) >= 0 for row in files),
            'invalid FAB location')
    # Both min and max tables are mandatory in the writer's Version_v1.
    for _ in range(2):
        require(tuple(map(int, next(lines).split(','))) == (count, 4), 'wrong native min/max count')
        for _ in range(count):
            values = next(lines).rstrip(',').split(',')
            require(len(values) == 4 and all(math.isfinite(float(v)) for v in values), 'invalid native min/max')
    require(next(lines, None) is None, 'unexpected VisMF trailer')
    return boxes, files


def _read_plot(path):
    path = Path(path)
    lines = iter((path/'Header').read_text().splitlines())
    require(next(lines) == 'HyperCLaw-V1.1', 'unsupported plotfile header')
    nc = int(next(lines))
    require(tuple([next(lines) for _ in range(nc)]) == VARIABLES, 'wrong macro fields')
    require(int(next(lines)) == 3, 'expected AMReX 3D storage')
    time = float(next(lines)); levels = int(next(lines))+1
    low = tuple(map(float, next(lines).split())); high = tuple(map(float, next(lines).split()))
    require(len(low) == len(high) == 3 and all(math.isfinite(v) for v in (*low, *high)) and
            all(b > a for a, b in zip(low, high)), 'invalid physical domain')
    next(lines)  # Legacy scalar ratios; domain lengths and spacings give the actual vector ratio.
    domains = [box(m.group()) for m in BOX.finditer(next(lines))]
    steps = tuple(map(int, next(lines).split()))
    spacing = [tuple(map(float, next(lines).split())) for _ in range(levels)]
    require(all(len(dx) == 3 and all(math.isfinite(h) and h > 0 for h in dx) for dx in spacing),
            'invalid native cell spacing')
    require(len(domains) == len(steps) == levels and int(next(lines)) == 0 and int(next(lines)) == 0,
            'unsupported domain/coordinate/ghost metadata')
    result = []
    for lev in range(levels):
        level, count, level_time = next(lines).split()
        require(int(level) == lev and float(level_time) == time and int(next(lines)) == steps[lev],
                'inconsistent level clock')
        for _ in range(int(count)*3):
            next(lines)  # Physical bounds are reconstructed from exact index boxes below.
        prefix = path/next(lines)
        boxes, files = vismf_header(Path(str(prefix)+'_H'), int(count))
        dlo, dhi = domains[lev]
        require(dlo == (0, 0, 0), 'this case requires zero-based domains')
        shape = tuple(reversed(dhi))
        data = np.full((4, *shape), np.nan)
        valid = np.zeros(shape, dtype=bool)
        for (lo, hi), (_, filename, offset) in zip(boxes, files):
            slices = tuple(slice(lo[d], hi[d]) for d in (2, 1, 0))
            require(all(0 <= lo[d] < hi[d] <= dhi[d] for d in range(3)) and not valid[slices].any(),
                    'overlapping/out-of-domain valid boxes')
            with (prefix.parent/filename).open('rb') as stream:
                stream.seek(int(offset))
                header = stream.readline().decode('ascii')
                require(header.startswith('FAB ') and box(header) == (lo, hi) and
                        header.rstrip().endswith(' 4'), 'FAB extent/components differ')
                require('(64 11 52 0 1 12 0 1023)' in header, 'expected IEEE binary64')
                if '(8 7 6 5 4 3 2 1)' in header:
                    dtype = '<f8'
                elif '(1 2 3 4 5 6 7 8)' in header:
                    dtype = '>f8'
                else:
                    raise ArtifactError('unsupported FAB byte order')
                size = tuple(hi[d]-lo[d] for d in (2, 1, 0))
                block = np.fromfile(stream, dtype=dtype, count=4*math.prod(size))
            require(block.size == 4*math.prod(size) and np.isfinite(block).all(), 'incomplete/nonfinite FAB')
            data[(slice(None), *slices)] = block.reshape((4, *size))
            valid[slices] = True
        require((data[0][valid] > 0).all(), 'nonpositive density')
        for d in range(3):
            require(math.isclose(dhi[d]*spacing[lev][d], high[d]-low[d], rel_tol=1e-12, abs_tol=1e-14),
                    'inconsistent physical lattice spacing')
        result.append(Level(data, valid, spacing[lev], steps[lev], boxes))
    require(math.isfinite(time), 'nonfinite time')
    return Plot(time, low, high, result)


def composite_masks(plot):
    masks = [level.valid.copy() for level in plot.levels]
    for lev in range(len(masks)-1):
        coarse, fine = plot.levels[lev:lev+2]
        ratio = tuple(round(coarse.spacing[d]/fine.spacing[d]) for d in range(3))
        require(ratio in ((2, 2, 1), (2, 2, 2)) and
                all(math.isclose(coarse.spacing[d], ratio[d]*fine.spacing[d], rel_tol=1e-13) and
                    fine.valid.shape[2-d] == coarse.valid.shape[2-d]*ratio[d] for d in range(3)),
                'unexpected refinement ratio')
        for lo, hi in fine.boxes:
            require(all(lo[d] % ratio[d] == hi[d] % ratio[d] == 0 for d in range(3)), 'unaligned fine box')
            slices = tuple(slice(lo[d]//ratio[d], hi[d]//ratio[d]) for d in (2, 1, 0))
            masks[lev][slices] = False
    return masks


def conservative_average(values, shape):
    """Uniform fine density and momentum averaged onto another native lattice."""
    rz, ry, rx = (a//b for a, b in zip(values.shape[1:], shape))
    require(tuple(a*b for a, b in zip((rz, ry, rx), shape)) == values.shape[1:], 'unmatched uniform grid')
    nz, ny, nx = shape
    def average(v):
        return v.reshape(nz, rz, ny, ry, nx, rx).mean(axis=(1, 3, 5))
    rho = average(values[0])
    return np.array([rho, *(average(values[0]*v)/rho for v in values[1:])])


def totals(plot, masks):
    mass = 0.0
    momentum = np.zeros(3)
    volume = 0.0
    for level, mask in zip(plot.levels, masks):
        dv = math.prod(level.spacing)
        rho = level.values[0][mask]
        mass += np.sum(rho, dtype=np.float64)*dv
        momentum += np.sum(level.values[1:, mask]*rho, axis=1, dtype=np.float64)*dv
        volume += int(mask.sum())*dv
    require(math.isclose(volume, math.prod(b-a for a, b in zip(plot.low, plot.high)), rel_tol=1e-12),
            'incomplete composite domain')
    require(math.isfinite(mass) and mass > 0 and np.isfinite(momentum).all(), 'nonfinite composite integral')
    return mass, momentum
