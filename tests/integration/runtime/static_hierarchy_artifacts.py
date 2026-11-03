"""Tests-owned v3 identity, periodic cell-set oracle and analytic LBM fields."""

from collections import deque
from dataclasses import dataclass
import csv
from itertools import product
import math


class ArtifactError(ValueError):
    pass


class NumericalMismatch(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def close(actual, expected, message):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > max(1e-12, 1e-12 * max(abs(actual), abs(expected))):
        raise NumericalMismatch(f"{message}: actual={actual:.17g} expected={expected:.17g}")


def numbers(text, cast=float):
    result = tuple(cast(value) for value in text.split(","))
    require(all(math.isfinite(value) for value in result), "nonfinite metadata")
    return result


def box_cells(bounds):
    return product(*(range(bounds[2*d], bounds[2*d+1]) for d in range(3)))


def grow(cells, shape, radius, dimensions):
    """Separable set dilation on the periodic lattice; no AMReX/HTML dependency."""
    result = cells
    for axis in range(dimensions):
        shifts = range(shape[axis]) if 2*radius+1 >= shape[axis] else range(-radius, radius+1)
        result = {tuple((cell[d]+shift) % shape[d] if d == axis else cell[d] for d in range(3))
                  for cell in result for shift in shifts}
    return result


def close_gaps(cells, shape, radius, dimensions):
    result = grow(cells, shape, radius, dimensions)
    if len(result) == math.prod(shape):
        return result
    for axis in range(dimensions):
        shifts = range(shape[axis]) if 2*radius+1 >= shape[axis] else range(-radius, radius+1)
        result = {cell for cell in result if all(
            tuple((cell[d]+shift) % shape[d] if d == axis else cell[d] for d in range(3)) in result
            for shift in shifts)}
    return result


def components(cells, shape, dimensions):
    remaining = set(cells)
    offsets = list(product(*([(-1, 0, 1)]*dimensions + [(0,)]*(3-dimensions))))
    offsets.remove((0, 0, 0))
    result = []
    while remaining:
        seed = remaining.pop()
        component, pending = {seed}, deque([seed])
        while pending:
            cell = pending.popleft()
            for offset in offsets:
                neighbor = tuple((cell[d]+offset[d]) % shape[d] for d in range(3))
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        result.append(component)
    return result


def repair_tags(cells, shape, radius, dimensions):
    """Close first, then grow all too-close components together until fixed."""
    while cells:
        repaired = close_gaps(cells, shape, radius, dimensions)
        pieces = components(repaired, shape, dimensions)
        # gap <= 2*r means center distance <= 2*r+1, including diagonal proximity.
        neighborhoods = [grow(piece, shape, radius, dimensions) for piece in pieces]
        participants = set()
        for i, neighborhood in enumerate(neighborhoods):
            adjacent = grow(neighborhood, shape, 1, dimensions)
            for j in range(i+1, len(pieces)):
                if not adjacent.isdisjoint(neighborhoods[j]):
                    participants.update((i, j))
        for index in participants:
            repaired |= neighborhoods[index]
        if repaired == cells:
            return cells
        cells = repaired
    return cells


def refine(cells, ratio):
    offsets = tuple(product(*(range(r) for r in ratio)))
    return {tuple(cell[d]*ratio[d]+offset[d] for d in range(3))
            for cell in cells for offset in offsets}


def coarsen(cells, ratio):
    return {tuple(cell[d]//ratio[d] for d in range(3)) for cell in cells}


@dataclass
class Geometry:
    identity: dict
    dimensions: int
    q: int
    ratio: tuple
    shapes: list
    coverage: list
    steps: tuple
    times: dict


REQUIRED = {"schema", "sample", "source_id", "profile", "levels", "ranks", "backend",
            "base_cells", "prob_lo", "prob_hi", "periodic", "ref_ratio", "max_grid_size",
            "gap_fill_radius_cells", "coarse_nesting_margin_cells", "grid_eff", "regions",
            "coverage_contract", "steps", "descriptor", "d", "q", "rho_reference", "velocity_reference"}


def read_identity(path):
    identity = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        require(separator and key and value and key not in identity, f"invalid identity line: {line}")
        identity[key] = value
    require(REQUIRED <= identity.keys(), f"missing identity keys: {REQUIRED-identity.keys()}")
    require(identity["schema"] == "static-hierarchy-v3", "unknown hierarchy schema")
    return identity


def validate_geometry(identity, source_id=None):
    if source_id is not None:
        require(identity["source_id"] == source_id, "source differs from requested execution identity")
    dimensions, q = int(identity["d"]), int(identity["q"])
    require((identity["descriptor"], dimensions, q) in {("D2Q9", 2, 9), ("D3Q27", 3, 27)},
            "inconsistent descriptor/d/q")
    require(identity["backend"] in {"cpu", "gpu"} and int(identity["ranks"]) > 0, "invalid backend/ranks")
    for key in ("rho_reference", "velocity_reference"):
        require(math.isfinite(float(identity[key])) and float(identity[key]) > 0, "invalid reference unit")
    cells, ratio = numbers(identity["base_cells"], int), numbers(identity["ref_ratio"], int)
    lo, hi = numbers(identity["prob_lo"]), numbers(identity["prob_hi"])
    levels = int(identity["levels"])
    require(levels > 0 and len(cells) == len(lo) == len(hi) == 3 and min(cells) > 0 and
            all(a < b for a, b in zip(lo, hi)), "invalid mesh identity")
    require(ratio == ((2, 2, 1) if dimensions == 2 else (2, 2, 2)) and
            numbers(identity["periodic"], int) == (1, 1, 1) and
            (dimensions == 3 or cells[2] == 1), "invalid topology identity")
    spacing = [(hi[d]-lo[d])/cells[d] for d in range(dimensions)]
    require(all(math.isclose(dx, spacing[0], abs_tol=1e-14, rel_tol=1e-12) for dx in spacing),
            "active lattice spacing differs")
    radius, margin = int(identity["gap_fill_radius_cells"]), int(identity["coarse_nesting_margin_cells"])
    efficiency = float(identity["grid_eff"])
    require(radius > 0 and margin > 0 and math.isfinite(efficiency) and 0 < efficiency <= 1,
            "invalid geometry controls")
    require(identity["coverage_contract"] == ("exact-repair" if efficiency == 1 else "minimum-repair"),
            "coverage contract differs from grid efficiency")
    grids = [numbers(size, int) for size in identity["max_grid_size"].split(";")]
    require(len(grids) in {1, levels} and all(len(size) == 3 and
            all(size[d] > 0 and size[d] % ratio[d] == 0 for d in range(3)) and
            (dimensions == 3 or size[2] == 1) for size in grids), "invalid max grid size")
    shapes = [tuple(cells[d]*ratio[d]**level for d in range(3)) for level in range(levels)]
    requests = [set() for _ in range(levels)]
    highest = 0
    if identity["regions"] != "none":
        for text in identity["regions"].split(";"):
            fields = text.split(",")
            require(len(fields) == 7, "region needs a level and six bounds")
            level, bounds = int(fields[0]), numbers(",".join(fields[1:]))
            require(0 < level < levels, "region level outside hierarchy")
            highest = max(highest, level)
            ranges = []
            for d in range(3):
                low, high, length = bounds[2*d], bounds[2*d+1], hi[d]-lo[d]
                require(0 < high-low <= length*(1+1e-12), "invalid periodic region extent")
                if d >= dimensions:
                    require(low == lo[d] and high == hi[d], "inactive region does not span H")
                n = shapes[level-1][d]
                a, b = (low-lo[d])*n/length, (high-lo[d])*n/length
                require(math.isclose(a, round(a), abs_tol=1e-10, rel_tol=0) and
                        math.isclose(b, round(b), abs_tol=1e-10, rel_tol=0), "region is not coarse-cell aligned")
                ranges.append({index % n for index in range(round(a), round(b))})
            requests[level] |= refine(set(product(*ranges)), ratio)
    require(highest+1 == levels, "requested highest level differs from hierarchy")
    minimum = [set(request) for request in requests]
    minimum[0] = set(product(*(range(n) for n in shapes[0])))
    for level in range(levels-1, 0, -1):
        tags = repair_tags(coarsen(minimum[level], ratio), shapes[level-1], radius, dimensions)
        minimum[level] = refine(tags, ratio)
        minimum[level-1] |= grow(tags, shapes[level-1], margin, dimensions)
    coverage = []
    for level, shape in enumerate(shapes):
        require(numbers(identity[f"domain_{level}"], int) == tuple(v for n in shape for v in (0, n)),
                "level domain differs from mesh/ratio")
        actual = set()
        grid = grids[0] if len(grids) == 1 else grids[level]
        for text in identity[f"boxes_{level}"].split(";"):
            bounds = numbers(text, int)
            require(len(bounds) == 6 and all(0 <= bounds[2*d] < bounds[2*d+1] <= shape[d] and
                    bounds[2*d+1]-bounds[2*d] <= grid[d] and
                    (level == 0 or bounds[2*d] % ratio[d] == bounds[2*d+1] % ratio[d] == 0)
                    for d in range(3)), "invalid actual valid Box")
            box = set(box_cells(bounds))
            require(actual.isdisjoint(box), "overlapping actual valid Boxes")
            actual |= box
        require(minimum[level] <= actual, f"L{level}: missing minimum/requested coverage")
        if efficiency == 1:
            require(actual == minimum[level], f"L{level}: coverage differs from independent repair")
        if level:
            tags = coarsen(actual, ratio)
            require(refine(tags, ratio) == actual, "actual coverage is not coarse-cell aligned")
            require(repair_tags(tags, shapes[level-1], radius, dimensions) == tags,
                    f"L{level}: actual coverage violates same-level repair contract")
            require(grow(tags, shapes[level-1], margin, dimensions) <= coverage[level-1],
                    f"L{level}: missing periodic coarse nesting margin")
        coverage.append(actual)
    steps = numbers(identity["steps"], int)
    require(steps and min(steps) >= 0 and tuple(sorted(set(steps))) == steps, "invalid step identity")
    times = {step: float(identity[f"time_{step}"]) for step in steps}
    require(all(math.isfinite(t) and t >= 0 for t in times.values()), "invalid physical time")
    require(levels == 1 or steps == (0,), "multilevel output unexpectedly advanced")
    timestep = 0.000625 if dimensions == 2 else 0.00125
    require(all(math.isclose(time, step*timestep, abs_tol=1e-15, rel_tol=1e-12)
                for step, time in times.items()), "physical time differs from test profile")
    return Geometry(identity, dimensions, q, ratio, shapes, coverage, steps, times)


D2_DIRECTIONS = ((0,0,0), (-1,1,0), (-1,0,0), (-1,-1,0), (0,-1,0),
                 (1,-1,0), (1,0,0), (1,1,0), (0,1,0))
D3_DIRECTIONS = ((0,0,0), (-1,0,0), (0,-1,0), (0,0,-1), (-1,-1,0), (-1,1,0),
                 (-1,0,-1), (-1,0,1), (0,-1,-1), (0,-1,1), (-1,-1,-1), (-1,-1,1),
                 (-1,1,-1), (-1,1,1), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,-1,0),
                 (1,0,1), (1,0,-1), (0,1,1), (0,1,-1), (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1))


def analytic_macros(geometry, level, cell, step):
    """None means an evolved nonuniform frame; its moments still get checked."""
    profile = geometry.identity["profile"]
    x, y, z = (2*math.pi*(cell[d]+0.5)/geometry.shapes[level][d] for d in range(3))
    if profile in {"layout-periodic-v1", "output-periodic-v1"}:
        require(step == 0 and geometry.dimensions == 2, "invalid D2 periodic profile/time")
        return (1+0.001*math.sin(x)*math.cos(y), 0.01*math.cos(x), 0.005*math.sin(y), 0.0)
    if profile == "uniform-bgk-nu0.05-u0.02-v1" or profile == "output-uniform-v1":
        require(geometry.dimensions == 2, "invalid D2 uniform profile")
        return (1.0, 0.02, 0.0, 0.0)
    if profile == "tgv-bgk-nu0.05-u0.02-v1":
        require(geometry.dimensions == 2, "invalid TGV descriptor")
        if step:
            return None
        return (1-0.75*0.02**2*(math.cos(2*x)+math.cos(2*y)),
                -0.02*math.cos(x)*math.sin(y), 0.02*math.sin(x)*math.cos(y), 0.0)
    if profile in {"D3-S0-bgk-nu0.05-uref0.02-v1", "d3-output-uniform-v1"}:
        require(geometry.dimensions == 3, "invalid D3 uniform profile")
        return (1.0, 0.01, 0.007, 0.005)
    require(profile in {"d3-periodic-xyz-v1", "D3-S1-bgk-nu0.05-uref0.02-v1"} and
            geometry.dimensions == 3, "unknown analytic profile")
    if step:
        return None
    return (1+0.001*math.sin(x)*math.cos(y)*math.cos(z),
            0.01*math.cos(y), 0.007*math.sin(z), 0.005*math.cos(x))


def equilibrium(macros, dimensions):
    rho, *velocity = macros
    speed2 = sum(u*u for u in velocity)
    directions = D2_DIRECTIONS if dimensions == 2 else D3_DIRECTIONS
    return tuple(math.prod(2/3 if c == 0 else 1/6 for c in direction[:dimensions]) * rho *
                 (1+3*(cu := sum(c*u for c, u in zip(direction, velocity)))+4.5*cu*cu-1.5*speed2)
                 for direction in directions)


def validate_state(geometry, key, state):
    level, step, *cell = key
    require(len(state) == geometry.q+4 and all(math.isfinite(value) for value in state),
            f"nonfinite/incomplete state: {key}")
    populations, macros = state[:geometry.q], state[geometry.q:]
    density = sum(populations)
    if density <= 0:
        raise NumericalMismatch(f"nonpositive density: {key}")
    directions = D2_DIRECTIONS if geometry.dimensions == 2 else D3_DIRECTIONS
    moments = (density, *(sum(f*c[d] for f, c in zip(populations, directions))/density for d in range(3)))
    for actual, expected in zip(macros, moments):
        close(actual, expected, f"population/macro moments {key}")
    initial = analytic_macros(geometry, level, cell, step)
    if initial is not None:
        for actual, expected in zip(state, equilibrium(initial, geometry.dimensions)+initial):
            close(actual, expected, f"independent analytic state {key}")
    return initial is not None


OUTPUT_VARIABLES = ("rho_norm", "ux_norm", "uy_norm", "uz_norm")


def load_plotfile_expectations(directory, source_id=None):
    geometry = validate_geometry(read_identity(directory/"identity.txt"), source_id)
    require(geometry.identity["sample"] == directory.name, "plotfile sample identity mismatch")
    require(len(geometry.steps) == 1, "plotfile oracle needs a single physical time")
    step = geometry.steps[0]
    expected = {}
    with (directory/"expected.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        require(next(reader, None) == ["level", "i", "j", "k", *OUTPUT_VARIABLES], "invalid plotfile oracle columns")
        for row in reader:
            require(len(row) == 8, "incomplete plotfile oracle row")
            level, *cell = map(int, row[:4])
            key = (level, *cell)
            require(0 <= level < len(geometry.coverage) and tuple(cell) in geometry.coverage[level] and
                    key not in expected, "duplicate/unexpected plotfile oracle cell")
            values = tuple(map(float, row[4:]))
            require(all(math.isfinite(value) for value in values), "nonfinite plotfile oracle")
            analytic = analytic_macros(geometry, level, cell, step)
            require(analytic is not None, "plotfile profile has no independent analytic field")
            for n, (actual, value) in enumerate(zip(values, analytic)):
                reference = float(geometry.identity["velocity_reference" if n else "rho_reference"])
                close(actual, value/reference, f"plotfile oracle analytic field {key}")
            expected[key] = values
    require(len(expected) == sum(map(len, geometry.coverage)), "incomplete plotfile oracle actual coverage")
    return geometry, expected
