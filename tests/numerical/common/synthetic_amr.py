"""CSV fixtures for checker unit tests; only the native binary reader is replaced."""
from pathlib import Path
import csv


def write_run(directory, metadata, plots):
    directory = Path(directory)
    directory.mkdir(parents=True)
    (directory/"identity.txt").write_text("".join(f"{key}={value}\n" for key, value in metadata.items()))
    (directory/"exit.txt").write_text("solver_exit=0\ntee_exit=0\n")
    with (directory/"layout.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("level", "box", "rank", "ilo", "jlo", "klo", "ihi", "jhi", "khi"))
        for lev, level in enumerate(plots[0].levels):
            for index, (low, high) in enumerate(level.boxes):
                writer.writerow((lev, index, 0, *low, *high))
    with (directory/"samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kind", "level", "step", "time", "coarsest_step", "phase", "path"))
        for step, plot in plots.items():
            for lev, level in enumerate(plot.levels):
                writer.writerow(("sync", lev, level.step, plot.time, step, "Sync", f"sync-{step}"))


def read_from(runs):
    def read(path):
        path = Path(path)
        return runs[path.parent][int(path.name.removeprefix("sync-"))]
    return read
