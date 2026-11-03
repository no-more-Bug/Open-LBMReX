"""Periodic-shear mathematics and native-grid diagnostic fields (tests only)."""
import math
import numpy as np
from amr_plotfile import require

U, AMPLITUDE, NU = 0.05, 0.001, 1/1280
CS = 1/math.sqrt(3)


def mode_vectors(dimensions, mode):
    if mode == 'axis':
        return (np.array([1., 0., 0.]), np.array([0., 1., 0.])) if dimensions == 2 else (
            np.array([0., 0., 1.]), np.array([1., 0., 0.]))
    return (np.array([1., 1., 0.]), np.array([-1., 1., 0.])/math.sqrt(2)) if dimensions == 2 else (
        np.ones(3), np.array([1., 1., -2.])/math.sqrt(6))


def exact(shape, spacing, time, dimensions, mode):
    m, p = mode_vectors(dimensions, mode)
    k = 2*math.pi*m
    coordinates = [(np.arange(shape[2-d])+0.5)*spacing[d] for d in range(3)]
    phase = (k[0]*coordinates[0][None, None, :] + k[1]*coordinates[1][None, :, None] +
             k[2]*coordinates[2][:, None, None] - U*np.linalg.norm(k)*time)
    sinc = math.prod(np.sinc(k[d]*spacing[d]/(2*math.pi)) for d in range(dimensions))
    signal = AMPLITUDE*np.exp(-NU*np.dot(k, k)*time)*sinc*np.sin(phase)
    background = U*m/np.linalg.norm(m)
    return np.array([np.ones(shape), *(background[d]+p[d]*signal for d in range(3))])


def residual(actual, reference):
    result = actual-reference
    result[0] *= CS/AMPLITUDE
    result[1:] /= AMPLITUDE
    return result


def rectangles(mask):
    """Merge identical consecutive row runs into nonoverlapping rectangles."""
    active = {}
    result = []
    for row in range(mask.shape[0]+1):
        runs = set()
        if row < mask.shape[0]:
            edges = np.flatnonzero(np.diff(np.pad(mask[row].astype(np.int8), (1, 1))))
            runs = set(zip(edges[::2], edges[1::2]))
        for run in list(active):
            if run not in runs:
                result.append((active.pop(run), row, *run))
        for run in runs:
            active.setdefault(run, row)
    return result


def expected_refined(shape, spacing, dimensions, layout):
    xyz = [((np.arange(shape[2-d])+.5)*spacing[d]).reshape(
        tuple(shape[2-d] if a == 2-d else 1 for a in range(3))) for d in range(3)]
    if layout == 'half':
        return np.broadcast_to(xyz[2 if dimensions == 3 else 0] >= .5, shape)
    if layout == 'separated':
        coordinate = xyz[2 if dimensions == 3 else 0]
        return np.broadcast_to((coordinate < .125) | (coordinate >= .875) |
                               ((coordinate >= .375) & (coordinate < .625)), shape)
    outer = np.ones(shape, dtype=bool); inner = np.ones(shape, dtype=bool)
    a, b = (.5, .875) if layout == 'corner' else (5/16, 11/16)
    for d in range(dimensions):
        outer &= (xyz[d] >= .125) & (xyz[d] < .875)
        inner &= (xyz[d] >= a) & (xyz[d] < b)
    return outer & ~inner


def interface_faces(refined, spacing, dimensions):
    """Exact axis-aligned boundary facets of the actual coarse-cell coverage.

    Taking periodic differences excludes same-level Box boundaries and the
    periodic seam when fine coverage continues through it.
    """
    result = []
    for d in range(dimensions):
        axis = 2-d
        changes = refined != np.roll(refined, 1, axis=axis)
        tangential = [2-a for a in range(3) if a != axis]
        for index in np.flatnonzero(np.any(changes, axis=tuple(a for a in range(3) if a != axis))):
            for r0, r1, c0, c1 in rectangles(np.take(changes, index, axis=axis)):
                lo = np.zeros(3); hi = np.zeros(3)
                lo[d] = hi[d] = index*spacing[d]
                for a, start, end in zip(tangential, (r0, c0), (r1, c1)):
                    lo[a] = start*spacing[a]; hi[a] = end*spacing[a]
                result.append((d, lo, hi))
    return result


def face_distance(shape, spacing, domain_length, face, dimensions):
    _, lo, hi = face
    distance2 = np.zeros(shape)
    for d in range(dimensions):
        coordinate = (np.arange(shape[2-d])+0.5)*spacing[d]
        length = domain_length[d]
        delta = np.abs((coordinate-(lo[d]+hi[d])/2+length/2) % length-length/2)
        gap = np.maximum(delta-(hi[d]-lo[d])/2, 0)
        broadcast = [1, 1, 1]; broadcast[2-d] = shape[2-d]
        distance2 += gap.reshape(broadcast)**2
    return np.sqrt(distance2)


def four_point(values, valid, dimensions):
    for d in range(dimensions):
        mask = valid.copy()
        coefficient = values.copy()
        for j, weight in ((1, -3), (2, 3), (3, -1)):
            mask &= np.roll(valid, -j, axis=2-d)
            coefficient += weight*np.roll(values, -j, axis=3-d)
        yield d, coefficient/8, mask


def norm(values, mask, spacing, dimensions):
    measure = int(mask.sum())*math.prod(spacing[:dimensions])
    selected = values[mask]
    if not selected.size:
        return None
    rms = float(np.sqrt(np.mean(selected*selected)))
    require(math.isfinite(rms), 'unrepresentable diagnostic norm')
    indices = np.unravel_index(np.argmax(np.where(mask, values, -np.inf)), mask.shape)
    return dict(measure=measure, l1=float(np.mean(np.abs(selected))),
                l2=rms, peak=float(np.max(np.abs(selected))),
                i=int(indices[2]), j=int(indices[1]), k=int(indices[0]))


def merge_norms(parts):
    measure = sum(p['measure'] for p in parts)
    return dict(measure=measure, l1=sum(p['l1']*p['measure'] for p in parts)/measure,
                l2=math.sqrt(sum(p['l2']**2*p['measure'] for p in parts)/measure),
                peak=max(p['peak'] for p in parts))


def visual_fields(actual, analytic, mapped, dimensions, mode, ra, rr):
    _, p = mode_vectors(dimensions, mode)
    m, _ = mode_vectors(dimensions, mode)
    background = U*m/np.linalg.norm(m)
    def perturbation(v):
        return sum(p[d]*(v[d+1]-background[d]) for d in range(3))/AMPLITUDE
    return dict(amr=perturbation(actual), analytic=perturbation(analytic), reference=perturbation(mapped),
                error_analytic=np.sqrt(np.sum(ra[1:]**2, axis=0)),
                error_reference=np.sqrt(np.sum(rr[1:]**2, axis=0)), pressure=rr[0],
                error_ux=rr[1], error_uy=rr[2], error_uz=rr[3])
