"""Second-order curl on each frame's native composite AMR hierarchy.

Velocity donors are restricted from finest to coarse, then missing fine
neighbours use the frozen AMReX cell_cons_interp MC reconstruction.
Integration remains on uncovered native valid cells; no uniform-grid resampling.
"""
import itertools
import numpy as np

from amr_plotfile import require


def restricted(values, ratio):
    nz, ny, nx = values.shape[-3:]
    rz, ry, rx = ratio
    shape = (*values.shape[:-3], nz//rz, rz, ny//ry, ry, nx//rx, rx)
    axes = tuple(values.ndim-3+i for i in (1, 3, 5))
    return values.reshape(shape).mean(axis=axes)


def coverage(valid, ratio):
    rz, ry, rx = ratio
    nz, ny, nx = valid.shape
    grouped = valid.reshape(nz//rz, rz, ny//ry, ry, nx//rx, rx)
    whole = grouped.all(axis=(1, 3, 5))
    require(np.array_equal(whole, grouped.any(axis=(1, 3, 5))), "fine coverage splits a coarse cell")
    return whole


def prolong(velocity, ratio):
    """Periodic MC slopes, multidimensional extrema limiter, ratio 1 or 2."""
    slopes = []
    lower = velocity.copy()
    upper = velocity.copy()
    for axis, refinement in enumerate(ratio, start=1):
        if refinement == 1:
            slopes.append(np.zeros_like(velocity))
            continue
        ahead = np.roll(velocity, -1, axis=axis)
        behind = np.roll(velocity, 1, axis=axis)
        central = 0.5*(ahead-behind)
        forward, backward = 2*(ahead-velocity), 2*(velocity-behind)
        slope = np.copysign(np.minimum(np.abs(central),
                           np.minimum(np.abs(forward), np.abs(backward))), central)
        slopes.append(np.where(forward*backward >= 0, slope, 0))
    for offsets in itertools.product(*(range(-1, 2) if r == 2 else (0,) for r in ratio)):
        neighbour = np.roll(velocity, offsets, axis=(1, 2, 3))
        np.minimum(lower, neighbour, out=lower)
        np.maximum(upper, neighbour, out=upper)
    excursion = sum(np.abs(slope)*(r-1)/(2*r) for slope, r in zip(slopes, ratio))
    denominator = np.where(excursion > 0, excursion, 1)
    alpha = np.minimum(1, np.minimum((upper-velocity)/denominator, (velocity-lower)/denominator))
    result = np.empty((3, *(n*r for n, r in zip(velocity.shape[1:], ratio))))
    for child in itertools.product(*(range(r) for r in ratio)):
        value = velocity.copy()
        for slope, subcell, r in zip(slopes, child, ratio):
            value += ((subcell+0.5)/r-0.5)*slope*alpha
        result[(slice(None), *(slice(i, None, r) for i, r in zip(child, ratio)))] = value
    return result


def composite_velocity(plot, dimensions):
    ratio = (2 if dimensions == 3 else 1, 2, 2)
    velocities = [level.values[1:].copy() for level in plot.levels]
    for lev in range(len(velocities)-2, -1, -1):
        mask = coverage(plot.levels[lev+1].valid, ratio)
        averaged = restricted(velocities[lev+1], ratio)
        velocities[lev][:, mask] = averaged[:, mask]
    require(np.all(plot.levels[0].valid), "L0 must cover the periodic domain")
    for lev in range(1, len(velocities)):
        missing = ~plot.levels[lev].valid
        if np.any(missing):
            interpolated = prolong(velocities[lev-1], ratio)
            velocities[lev][:, missing] = interpolated[:, missing]
    return velocities


def curls(plot, dimensions):
    require(dimensions in (2, 3), "unsupported curl dimensions")
    result = []
    for level, velocity in zip(plot.levels, composite_velocity(plot, dimensions)):
        derivative = []
        for d in range(3):
            axis = 3-d
            derivative.append((np.roll(velocity, -1, axis=axis)-np.roll(velocity, 1, axis=axis))/
                              (2*level.spacing[d]) if d < dimensions else np.zeros_like(velocity))
        curl = np.zeros_like(velocity)
        curl[2] = derivative[0][1]-derivative[1][0]
        if dimensions == 3:
            curl[0] = derivative[1][2]-derivative[2][1]
            curl[1] = derivative[2][0]-derivative[0][2]
        require(np.all(np.isfinite(curl[:, level.valid])), "nonfinite native curl")
        result.append(curl)
    return result
