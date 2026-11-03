"""TGV analytic reference and composite metrics on the native AMR lattices."""
import math
import numpy as np
from amr_plotfile import composite_masks, conservative_average, require
from amr_validation import integrals, quasi_metrics


def exact(level, time, speed):
    nz, ny, nx = level.valid.shape
    x = (np.arange(nx)+.5)*level.spacing[0]
    y = (np.arange(ny)+.5)*level.spacing[1]
    xx, yy = x[None, :], y[:, None]
    decay = np.exp(-8*np.pi**2*.1*time)
    rho = 1-.75*speed**2*(np.cos(4*np.pi*xx)+np.cos(4*np.pi*yy))*decay**2
    ux = -np.cos(2*np.pi*xx)*np.sin(2*np.pi*yy)*decay
    uy = np.sin(2*np.pi*xx)*np.cos(2*np.pi*yy)*decay
    return np.broadcast_to(np.array([rho, ux, uy, np.zeros((ny, nx))])[:, None], (4, nz, ny, nx))


def interface_neighborhood(plot, masks):
    """Distance to actual coarse/fine faces; same-level Box edges are excluded."""
    if len(plot.levels) == 1:
        return [np.zeros_like(masks[0])]
    covered = (~masks[0] & plot.levels[0].valid)[0]
    h = plot.levels[0].spacing
    faces = []
    for axis in (0, 1):
        for j, i in np.argwhere(covered != np.roll(covered, 1, axis=1-axis)):
            low = np.array([i*h[0], j*h[1]])
            high = low+np.array(h[:2])
            high[axis] = low[axis]
            faces.append((low, high))
    require(faces, "AMR has no coarse/fine interface")
    result = []
    for level, mask in zip(plot.levels, masks):
        ny, nx = mask.shape[1:]
        x = (np.arange(nx)+.5)*level.spacing[0]
        y = (np.arange(ny)+.5)*level.spacing[1]
        distance2 = np.full((ny, nx), np.inf)
        for low, high in faces:
            dx = np.maximum(np.maximum(low[0]-x, x-high[0]), 0)
            dy = np.maximum(np.maximum(low[1]-y, y-high[1]), 0)
            distance2 = np.minimum(distance2, dy[:, None]**2+dx[None, :]**2)
        result.append(mask & np.broadcast_to(distance2 <= (1/16)**2, mask.shape))
    return result


def errors(plot, speed, dimensions, reference=None):
    masks = composite_masks(plot)
    near = interface_neighborhood(plot, masks)
    numerator = denominator = peak = 0.
    if reference is not None:
        require(len(reference.levels) == 1 and reference.levels[0].valid.all() and
                plot.low == reference.low and plot.high == reference.high and
                math.isclose(plot.time, reference.time, rel_tol=1e-11, abs_tol=1e-14),
                "unmatched reference domain/time")
    for level, mask, interface in zip(plot.levels, masks, near):
        analytic = exact(level, plot.time, speed)
        expected = analytic if reference is None else conservative_average(reference.levels[0].values, mask.shape)
        error = np.linalg.norm(level.values[1:]-expected[1:], axis=0)
        weight = math.prod(level.spacing[:dimensions])
        numerator += float(np.sum(error[mask]**2))*weight
        denominator += float(np.sum(analytic[1:, mask]**2))*weight
        if interface.any():
            peak = max(peak, float(np.max(error[interface])))
    require(denominator > 0 and math.isfinite(numerator+denominator+peak), "invalid TGV error norm")
    return math.sqrt(numerator/denominator), peak


def statistics(plot, speed, dimensions, initial_mass, initial_momentum):
    masks = composite_masks(plot)
    mass, momentum = integrals(plot, masks, dimensions)
    e2, interface = errors(plot, speed, dimensions)
    return dict(time=plot.time, analytic_e2=e2, analytic_interface_peak=interface,
                mass=mass, mass_drift=abs(mass-initial_mass)/initial_mass,
                momentum_drift=float(np.linalg.norm(momentum-initial_momentum))/initial_mass,
                **quasi_metrics(plot, masks))


def passes(row, coarse_cells, amr):
    limit = .02 if coarse_cells == 80 else .05
    return row["analytic_e2"] <= limit and (not amr or row["analytic_interface_peak"] <= .05) and all(
        row[key] <= 1e-10 for key in ("mass_drift", "momentum_drift", "uz_max", "rho_z_max", "velocity_z_max"))


def observed_order(first, second):
    return math.log(first/second)/math.log(2) if first > 0 and second > 0 else None
