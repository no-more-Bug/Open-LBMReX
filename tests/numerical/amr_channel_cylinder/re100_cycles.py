"""The first qualified continuous 3+5 window, separate from literature gates."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"channel_cylinder"))
from analyze_re100 import Sample, quadratic, make_cycle


def closed_integral(times, values, begin, end):
    """Trapezoidal time integral, with linearly interpolated cycle endpoints."""
    left, right = np.searchsorted(times, (begin, end), side="right")
    inner = times[left:right]
    inner = inner[inner < end]
    x = np.concatenate(([begin], inner, [end]))
    y = np.interp(x, times, values)
    return float(np.trapz(y, x))


def cycle_records(samples):
    times = np.array([sample.t_star for sample in samples])
    cd = np.array([sample.cd for sample in samples])
    cl = np.array([sample.cl for sample in samples])
    minima = [quadratic(samples, index, "cl", True)
              for index in range(1, len(samples)-1)
              if samples[index].cl < samples[index-1].cl and samples[index].cl < samples[index+1].cl]
    cycles = []
    for start, end in zip(minima, minima[1:]):
        if start.time < 250:
            continue
        old = make_cycle(samples, start, end)
        period = end.time-start.time
        row = dict(begin_t_star=start.time, end_t_star=end.time, period_t_star=period,
                   samples=int(np.count_nonzero((times >= start.time) & (times <= end.time))),
                   Cd_min=old.cd_min, Cd_max=old.cd_max, Cl_min=old.cl_min, Cl_max=old.cl_max)
        for prefix, values in (("Cd", cd), ("Cl", cl)):
            row[prefix+"_midrange"] = (row[prefix+"_min"]+row[prefix+"_max"])/2
            row[prefix+"_peak_to_peak"] = row[prefix+"_max"]-row[prefix+"_min"]
            row[prefix+"_amplitude"] = row[prefix+"_peak_to_peak"]/2
            row[prefix+"_mean"] = closed_integral(times, values, start.time, end.time)/period
        cycles.append(row)
    return cycles


def normalized_ranges(cycles):
    ranges = {}
    for key in ("period_t_star", "Cd_amplitude", "Cl_amplitude", "Cd_mean", "Cl_mean"):
        values = [row[key] for row in cycles]
        scale = (max(row["Cl_peak_to_peak"] for row in cycles) if key == "Cl_mean"
                 else max(abs(value) for value in values))
        if not np.isfinite(scale) or scale <= 0:
            return None
        ranges[key] = (max(values)-min(values))/scale
    return ranges


def select_window(cycles):
    for start in range(len(cycles)-7):
        window = cycles[start:start+8]
        if any(row["samples"] < 200 or row["period_t_star"] <= 0 or
               row["Cd_amplitude"] <= 0 or row["Cl_amplitude"] <= 0 for row in window):
            continue
        qualified = True
        for length in range(3, 9):
            ranges = normalized_ranges(window[:length])
            if ranges is None or any(value > .01 for value in ranges.values()):
                qualified = False
                break
        if qualified:
            return start, ranges
    return None


def summarize(samples, cycles, start):
    """Extrema retain the legacy definitions; means use actual time integration."""
    used = cycles[start+3:start+8]
    begin, end = used[0]["begin_t_star"], used[-1]["end_t_star"]
    times = np.array([sample.t_star for sample in samples])
    result = dict(stable_begin_t_star=cycles[start]["begin_t_star"],
                  stable_end_t_star=begin, statistics_begin_t_star=begin,
                  statistics_end_t_star=end, St=5/(end-begin),
                  period_min=min(row["period_t_star"] for row in used),
                  period_max=max(row["period_t_star"] for row in used))
    for prefix, attr in (("Cd", "cd"), ("Cl", "cl")):
        values = np.array([getattr(sample, attr) for sample in samples])
        result[prefix+"_mean"] = closed_integral(times, values, begin, end)/(end-begin)
        for suffix in ("min", "max", "midrange", "peak_to_peak", "amplitude"):
            key = prefix+"_"+suffix
            result[key] = float(np.mean([row[key] for row in used]))
    return result


def compare_metrics(actual, reference, tolerances):
    """A zero reference is insufficient evidence, never a fabricated denominator."""
    rows = []
    for metric, tolerance in tolerances.items():
        value = reference[metric]
        error = abs(actual[metric]-value)/abs(value) if value != 0 else None
        rows.append(dict(metric=metric, value=actual[metric], reference=value,
                         relative_error=error, tolerance=tolerance,
                         status=2 if error is None else int(error > tolerance)))
    return rows
