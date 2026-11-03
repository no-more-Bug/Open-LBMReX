"""DSL native-grid curl: fourth-order derivative reconstruction.

The approved layouts are full x/z bands. Only y derivatives need mixed-grid
stencils; coefficients repeat by x/z coarse-cell phase and are reused in time.
No population ghosts, covered values, z averaging or uniform resampling enter.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np
from amr_plotfile import require


def displacement(x, center, length):
    return (x-center+length/2) % length-length/2


def quartic_weights(points, scale, dimensions):
    p = np.asarray(points)[:, :dimensions]/scale
    columns = [np.ones(len(p))]
    for degree in range(1, 5):
        columns.extend(np.prod(p[:, axes], axis=1)
                       for axes in combinations_with_replacement(range(dimensions), degree))
    matrix = np.column_stack(columns)
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    require(len(singular) == len(columns) and singular[-1] > 1e-12*singular[0],
            'rank-deficient polynomial interface stencil')
    # Row 2 of the pseudoinverse selects the derivative in physical y.
    return ((vt[:, 2]/singular) @ u.T)/scale


def central_derivative(values, axis, spacing):
    first = np.roll(values, -1, axis=axis)-np.roll(values, 1, axis=axis)
    return (8*first-np.roll(values, -2, axis=axis)+np.roll(values, 2, axis=axis))/(12*spacing)


@dataclass
class Fit:
    level: int
    row: int
    phase_x: int
    phase_z: int
    sources: list  # (level,z,y,x), each physical periodic cell used once
    weights: np.ndarray


class CurlPlan:
    def __init__(self, plot, masks, dimensions):
        self.dimensions = dimensions
        self.fits = []
        self.rows = []
        for lev, (level, mask) in enumerate(zip(plot.levels, masks)):
            active = mask[0, :, 0]
            require(np.array_equal(mask, np.broadcast_to(active[None, :, None], mask.shape)),
                    'DSL actual coverage must span complete x/z bands')
            self.rows.append(active)
        offsets = (-2, -1, 1, 2)
        for lev, (level, active) in enumerate(zip(plot.levels, self.rows)):
            boundary = active & np.logical_or.reduce([~np.roll(active, shift) for shift in offsets])
            for row in np.flatnonzero(boundary):
                y = (row+0.5)*level.spacing[1]
                scale = level.spacing[1]
                # Identify the actual adjacent coarse spacing at each missing neighbor.
                for shift in offsets:
                    position = (y+shift*level.spacing[1]) % 1
                    for source, source_rows in zip(plot.levels, self.rows):
                        if source_rows[int(position/source.spacing[1]) % len(source_rows)]:
                            scale = max(scale, source.spacing[1])
                for px in range(2**lev):
                    for pz in range(2**lev if dimensions == 3 else 1):
                        center = np.array([(px+.5)*level.spacing[0], y, (pz+.5)*level.spacing[2]])
                        indices, points = [], []
                        for source_level, (source, source_rows) in enumerate(zip(plot.levels, self.rows)):
                            selected, delta = [], []
                            for d in range(3):
                                length = plot.high[d]-plot.low[d]
                                coordinates = (np.arange(source.valid.shape[2-d])+.5)*source.spacing[d]
                                dist = displacement(coordinates, center[d], length)
                                keep = np.abs(dist) <= 3*scale+1e-14
                                if d == 1:
                                    keep &= source_rows
                                if dimensions == 2 and d == 2:
                                    keep[:] = True
                                selected.append(np.flatnonzero(keep)); delta.append(dist)
                            for k in selected[2]:
                                for j in selected[1]:
                                    for i in selected[0]:
                                        indices.append((source_level, k, j, i))
                                        points.append([delta[0][i], delta[1][j], delta[2][k]])
                        weights = quartic_weights(points, scale, dimensions)
                        self.fits.append(Fit(lev, int(row), px, pz, indices, weights))

    def curl(self, plot):
        # Only d(ux)/dy and (in D3) d(uz)/dy occur in curl.
        components = (1, 3) if self.dimensions == 3 else (1,)
        dy = []
        for level in plot.levels:
            velocity = level.values[list(components)]
            dy.append(central_derivative(velocity, 2, level.spacing[1]))
        for fit in self.fits:
            level = plot.levels[fit.level]
            sx = 2**fit.level; sz = sx if self.dimensions == 3 else 1
            xs = np.arange(fit.phase_x, level.valid.shape[2], sx)
            zs = np.arange(fit.phase_z, level.valid.shape[0], sz)
            value = np.zeros((len(components), len(zs), len(xs)))
            for (lev, k, j, i), weight in zip(fit.sources, fit.weights):
                source = plot.levels[lev]
                ix = (i+(xs-fit.phase_x)//sx*2**lev) % source.valid.shape[2]
                iz = (k+(zs-fit.phase_z)//sz*(2**lev if self.dimensions == 3 else 1)) % source.valid.shape[0]
                for q, component in enumerate(components):
                    value[q] += weight*source.values[component, iz[:, None], j, ix[None, :]]
            for q in range(len(components)):
                dy[fit.level][q, zs[:, None], fit.row, xs[None, :]] = value[q]
        result = []
        for lev, level in enumerate(plot.levels):
            u = level.values[1:]
            dx = central_derivative(u, 3, level.spacing[0])
            omega_z = dx[1]-dy[lev][0]
            if self.dimensions == 2:
                result.append(np.array([omega_z]))
            else:
                dz = central_derivative(u, 1, level.spacing[2])
                result.append(np.array([dy[lev][1]-dz[1], dz[0]-dx[2], omega_z]))
        return result
