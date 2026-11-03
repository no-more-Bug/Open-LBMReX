#!/usr/bin/env python3
"""Pair complete short AMR runs by coordinates across ranks/backends; exits 0/1/2."""
import argparse
import math
from pathlib import Path
from check_amr_evolution import DIRECTIONS, EPS, LAYOUT, MACRO, finite, moments, require, rows, within

IDENTITY = "schema sample descriptor backend real_digits ranks levels coarse_steps tau_finest uniform nx ny nz lx ly lz dt0".split()
BALANCE = "step time_phys cells volume mass px py pz reference_mass reference_px reference_py reference_pz".split()


def case_key(sample, steps=2, tau=0.65, uniform=False, split=0, sampling=1, strategy="sfc"):
    return f"{sample}-{'uniform' if uniform else 'wave'}-tau{tau:.6f}-n{steps}-split{split}-sample{sampling}-{strategy}"


def expected_cases(descriptor, group):
    d = 2 if descriptor == "D2Q9" else 3
    prefix, large = f"D{d}-", 32 if d == 2 else 16
    cases = {}

    def add(sample, steps=2, tau=0.65, uniform=False, split=0, sampling=1, strategy="sfc"):
        cases[case_key(sample, steps, tau, uniform, split, sampling, strategy)] = (sample, steps, tau, uniform)

    if group == "advance":
        add(prefix+"E0-box8")
        for tau in (0.65, 0.8):
            add(prefix+f"E1-box{large}", tau=tau, uniform=True)
            add(prefix+"E2-box8", tau=tau, uniform=True)
        add(prefix+f"E1-box{large}", steps=4)
        for split, sampling in ((0, 1), (1, 1), (0, 0), (0, 2)):
            add(prefix+"E2-box8", steps=4, split=split, sampling=sampling)
        for suffix in ("", "-half-x"):
            add(prefix+"Topo-L3Separated-L2Overlap-AutoL1"+suffix)
    else:
        through = prefix+"P-through-3-x-P" if d == 2 else "D3-T3-z-P"
        samples = [prefix+name for name in ("Topo-L2Separated-AutoL1Merged", "Topo-Nonconvex",
                                           "Topo-Through-all", "Narrow-gap", "Retained-hole")]
        samples += (["D2-P-corner-P-box8"] if d == 2 else
                    ["D3-Vertex-P-box8", "D3-Vertex-P-box16", "D3-Wrap3-P-box8"])
        samples += [through+f"-box{size}" for size in (large, 8)]
        for sample in samples:
            add(sample, strategy="rr")
        for sample in (prefix+f"E2-box{large}", prefix+"E2-box8", through+f"-box{large}", through+"-box8"):
            add(sample)
        add(prefix+"E2-box8", strategy="rr")
    return cases


def read_case(directory, descriptor, expected):
    sample, horizon, tau, uniform = expected
    identity = list(rows(directory/"identity.csv", IDENTITY))
    require(len(identity) == 1, "identity must have one record")
    meta = identity[0]
    require(meta["schema"] == "amr-advance-v1" and meta["sample"] == sample and
            meta["descriptor"] == descriptor, "wrong advance identity")
    require(meta["backend"] in ("cpu", "cuda") and int(meta["ranks"]) in (1, 2) and
            int(meta["real_digits"]) == 53, "wrong backend/ranks/precision")
    require(int(meta["coarse_steps"]) == horizon and finite(meta["tau_finest"]) == tau and
            int(meta["uniform"]) == int(uniform), "wrong scientific configuration/horizon")
    levels, ranks = int(meta["levels"]), int(meta["ranks"])
    require(1 <= levels <= 4, "wrong level count")
    ratio = (2, 2, 1 if descriptor == "D2Q9" else 2)
    shape = tuple(int(meta[k]) for k in ("nx", "ny", "nz"))
    length = tuple(finite(meta[k]) for k in ("lx", "ly", "lz"))
    dt0 = finite(meta["dt0"])
    require(all(n > 0 for n in shape) and all(x > 0 for x in length) and dt0 > 0, "invalid geometry/timestep")
    require(descriptor != "D2Q9" or shape[2] == 1, "D2 requires one z cell")
    dx = length[0]/shape[0]
    require(all(within(length[d]/shape[d], dx, dx) for d in range(2 if descriptor == "D2Q9" else 3)) and
            within(dt0, 0.02*dx, dt0), "wrong active spacing/timestep")
    cells, boxes = [set() for _ in range(levels)], [set() for _ in range(levels)]
    for box in rows(directory/"layout.csv", LAYOUT):
        lev, number, owner = (int(box[k]) for k in ("level", "box", "owner"))
        require(0 <= lev < levels and number >= 0 and number not in boxes[lev] and 0 <= owner < ranks,
                "wrong layout level/Box/owner")
        boxes[lev].add(number)
        lo = tuple(int(box[k]) for k in ("ilo", "jlo", "klo"))
        hi = tuple(int(box[k]) for k in ("ihi", "jhi", "khi"))
        require(all(0 <= lo[d] <= hi[d] < shape[d]*ratio[d]**lev for d in range(3)), "Box outside domain")
        for k in range(lo[2], hi[2]+1):
            for j in range(lo[1], hi[1]+1):
                for i in range(lo[0], hi[0]+1):
                    point = (i, j, k)
                    require(point not in cells[lev], "overlapping valid Boxes")
                    cells[lev].add(point)
    require(len(cells[0]) == math.prod(shape) and all(cells), "incomplete actual coverage")
    covered = [{tuple(p[d]//ratio[d] for d in range(3)) for p in cells[lev+1]}
               if lev+1 < levels else set() for lev in range(levels)]
    count = sum(len(cells[lev]-covered[lev]) for lev in range(levels))
    volume = math.prod(length)
    measure = [volume/(math.prod(shape)*math.prod(ratio)**lev) for lev in range(levels)]
    require(within(math.fsum(len(cells[lev]-covered[lev])*measure[lev] for lev in range(levels)), volume, volume),
            "composite volume mismatch")
    balance = list(rows(directory/"balance.csv", BALANCE))
    require(len(balance) == horizon+1, "incomplete balance horizon")
    ordered = [sorted(level, key=lambda p: (p[2], p[1], p[0])) for level in cells]
    return meta, ordered, covered, measure, count, balance


def compare_case(left, right, descriptor, expected, identities):
    a, b = (read_case(root, descriptor, expected) for root in (left, right))
    for data, identity in zip((a, b), identities):
        require((data[0]["backend"], int(data[0]["ranks"])) == identity, "wrong backend/rank pairing")
    for key in IDENTITY:
        if key not in ("backend", "ranks"):
            require(a[0][key] == b[0][key], f"paired identity differs: {key}")
    require(a[1] == b[1], "paired actual coverage differs")
    _, horizon, _, _ = expected
    directions = DIRECTIONS[descriptor]
    fields = "level step time_phys i j k".split()+[f"f{q}" for q in range(len(directions))]+MACRO
    dt0, failures, initial = finite(a[0]["dt0"]), set(), []
    eta = 4096*EPS*(1+math.ceil(math.log2(max(2, a[4]))))
    for step in range(horizon+1):
        streams = [iter(rows(root/f"step-{step}"/"states.csv", fields)) for root in (left, right)]
        sums = [[[] for _ in range(4)] for _ in range(2)]
        for lev, points in enumerate(a[1]):
            for point in points:
                values = []
                for side, stream in enumerate(streams):
                    row = next(stream, None)
                    require(row is not None, "incomplete state frame")
                    require((int(row["level"]), int(row["step"])) == (lev, step*2**lev) and
                            tuple(int(row[k]) for k in ("i", "j", "k")) == point, "wrong level/step/coordinate")
                    require(within(finite(row["time_phys"]), step*dt0, dt0), "wrong physical time")
                    f = [finite(row[f"f{q}"]) for q in range(len(directions))]
                    macro = [finite(row[k]) for k in MACRO]
                    rho, momentum = moments(f, directions)
                    if rho <= 0:
                        failures.add("positive density")
                    wanted = [rho]+[p/rho if rho else 0 for p in momentum]
                    if any(not within(macro[n], wanted[n], 1 if n == 0 else 0.02) for n in range(4)):
                        failures.add("macroscopic moments")
                    values.append(f+macro)
                    if point not in a[2][lev]:
                        for n, value in enumerate([rho]+momentum):
                            sums[side][n].append(value*a[3][lev])
                for n, (v, w) in enumerate(zip(*values)):
                    if not within(v, w, 1 if n <= len(directions) else 0.02):
                        failures.add("paired populations/macros")
        require(all(next(stream, None) is None for stream in streams), "extra state rows")
        totals = [[math.fsum(component) for component in source] for source in sums]
        if step == 0:
            initial = totals
            require(all(t[0] > 0 for t in initial), "nonpositive initial mass")
        for side, data in enumerate((a, b)):
            record = data[5][step]
            require(int(record["step"]) == step and int(record["cells"]) == a[4] and
                    within(finite(record["time_phys"]), step*dt0, dt0), "wrong balance identity")
            require(within(finite(record["volume"]), math.prod(finite(data[0][k]) for k in ("lx", "ly", "lz")),
                           finite(record["volume"])), "wrong recorded volume")
            logged = [finite(record[k]) for k in ("mass", "px", "py", "pz")]
            reference = [finite(record[k]) for k in ("reference_mass", "reference_px", "reference_py", "reference_pz")]
            for wanted in (logged, reference, initial[side], totals[1-side]):
                mass_error = abs(totals[side][0]-wanted[0])/initial[side][0]
                momentum_error = math.hypot(*(totals[side][n]-wanted[n] for n in range(1, 4)))/(initial[side][0]*0.02)
                if mass_error > eta or momentum_error > eta:
                    failures.add("composite balance/parity")
    return failures


def compare_roots(left, right, descriptor, group, identities):
    expected = expected_cases(descriptor, group)
    for root in (left, right):
        require({p.name for p in root.iterdir() if p.is_dir()} == set(expected), "missing/extra matrix case")
    failures = set()
    for key, specification in expected.items():
        failed = compare_case(left/key, right/key, descriptor, specification, identities)
        failures.update(failed)
        print(f"advance_parity case={key} result={'FAIL' if failed else 'PASS'}")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--descriptor", choices=DIRECTIONS, required=True)
    parser.add_argument("--group", choices=("advance", "matrix"), required=True)
    pair = parser.add_mutually_exclusive_group(required=True)
    pair.add_argument("--backend", choices=("cpu", "cuda"), help="compare 1 to 2 ranks on this backend")
    pair.add_argument("--ranks", type=int, choices=(1, 2), help="compare CPU to CUDA at this rank count")
    args = parser.parse_args(argv)
    identities = ((args.backend, 1), (args.backend, 2)) if args.backend else (("cpu", args.ranks), ("cuda", args.ranks))
    try:
        failed = compare_roots(args.left, args.right, args.descriptor, args.group, identities)
    except (OSError, ValueError, OverflowError) as error:
        print(f"INVALID: {error}")
        return 2
    print(f"advance_parity group={args.group} failures={sorted(failed)}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
