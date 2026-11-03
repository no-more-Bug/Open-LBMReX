#!/usr/bin/env python3
"""Render validated C8 comparison artifacts with pvpython/Matplotlib; no solver work."""
import argparse
from pathlib import Path
import os
import numpy as np

from compare_tandem import ComparisonRun, SNAPSHOTS, STEPS, DT, pair, fluid_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same', type=Path, required=True)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(args.output/'mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    runs = (ComparisonRun(args.same), ComparisonRun(args.cross))
    pair(*runs)
    backend = runs[0].id['backend'].upper()
    plt.rcParams.update({'font.family': ['DejaVu Sans'], 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = ('#1565b7', '#df7824')
    labels = ('Both cylinders in L2', 'A in L2, B in L1')
    time = (np.arange(STEPS)+.5)*DT
    for first, filename in ((0, 'coefficients-full.png'), (100, 'coefficients-after-startup.png'),
                            (16000, 'coefficients-tail.png')):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
        for row, name in enumerate(('cylinder_a', 'cylinder_b')):
            for col, component in enumerate(('Cd', 'Cl')):
                ax = axes[row, col]
                for run, color, label, style in zip(runs, colors, labels, ('-', '--')):
                    ax.plot(time[first:], run.coefficients[name][first:, col], color=color,
                            lw=1.1, linestyle=style, label=label)
                ax.set(xlabel='Physical time', ylabel=component, title=f'Cylinder {name[-1].upper()}')
                ax.grid(alpha=.2)
                ax.ticklabel_format(axis='y', style='plain', useOffset=False)
                ax.legend(fontsize=9)
        fig.suptitle(f'Tandem cylinder forces | {backend} / D2Q9 / YMS\n'
                     f'L0 interval averages, D2 unit span | shown t = {time[first]:.4g} to {STEPS*DT:g}', fontsize=15)
        fig.savefig(args.output/filename, dpi=170, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
    # Native cell colors; only streamlines are sampled onto a uniform display grid.
    ranges = []
    for step in SNAPSHOTS:
        plots = [run.plot(step) for run in runs]
        masks = [fluid_masks(plot) for plot in plots]
        fields = []
        for plot in plots:
            fields.append([(np.linalg.norm(data.values[1:, 0], axis=0),
                            (data.values[0, 0]-1)/(3*.035**2)) for data in plot.levels])
        limits = []
        for component in range(2):
            values = np.concatenate([values[component][mask[0]] for levels, active in zip(fields, masks)
                                     for values, mask in zip(levels, active)])
            low, high = (0., float(values.max())) if component == 0 else (float(values.min()), float(values.max()))
            if high == low:
                high = low+1e-12
            limits.append((low, high))
            ranges.append(f'{step},{component},{low:.17g},{high:.17g}\n')
        fig, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
        artists = []
        for row, (plot, active, values) in enumerate(zip(plots, masks, fields)):
            for component in range(2):
                ax = axes[row, component]
                for data, mask, shown in zip(plot.levels, active, values):
                    x = plot.low[0]+np.arange(data.values.shape[3]+1)*data.spacing[0]
                    y = plot.low[1]+np.arange(data.values.shape[2]+1)*data.spacing[1]
                    artist = ax.pcolormesh(x, y, np.ma.array(shown[component], mask=~mask[0]),
                                          cmap='viridis' if component == 0 else 'RdBu_r',
                                          vmin=limits[component][0], vmax=limits[component][1],
                                          shading='flat', rasterized=True)
                if row == 0:
                    artists.append(artist)
                if component == 0:
                    x, y = np.linspace(.001, 2.199, 660), np.linspace(.001, .409, 164)
                    velocity = np.zeros((2, len(y), len(x)))
                    for data in plot.levels:
                        i = np.floor((x-plot.low[0])/data.spacing[0]).astype(int)
                        j = np.floor((y-plot.low[1])/data.spacing[1]).astype(int)
                        valid = data.valid[0][np.ix_(j, i)]
                        for d in range(2):
                            sampled = data.values[d+1, 0][np.ix_(j, i)]
                            velocity[d][valid] = sampled[valid]
                    fluid = np.ones(velocity.shape[1:], dtype=bool)
                    for center in (.2, .4):
                        fluid &= (x[None, :]-center)**2+(y[:, None]-.2)**2 >= .05**2
                    ax.streamplot(x, y, *(np.ma.array(v, mask=~fluid) for v in velocity),
                                  color='#263849', density=(2.1, .7), linewidth=.55, arrowsize=.65)
                fine = plot.levels[2]
                x = plot.low[0]+(np.arange(fine.valid.shape[2])+.5)*fine.spacing[0]
                y = plot.low[1]+(np.arange(fine.valid.shape[1])+.5)*fine.spacing[1]
                ax.contour(x, y, fine.valid[0], levels=[.5], colors='#242f40', linewidths=.6, linestyles='--')
                for name, center in (('A', .2), ('B', .4)):
                    ax.add_patch(Circle((center, .2), .05, facecolor='#e7ecf2', edgecolor='#263849', zorder=5))
                    ax.text(center, .2, name, ha='center', va='center', fontsize=9, zorder=6)
                ax.set(xlim=(0, 2.2), ylim=(0, .41), aspect='equal', xlabel='x', ylabel='y', title=labels[row])
        for component, label in enumerate(('Speed / mean inlet speed', '(rho/rho0 - 1) / (3 U_lb^2)')):
            fig.colorbar(artists[component], ax=axes[:, component], orientation='horizontal', shrink=.85, label=label)
        fig.suptitle(f'Tandem flow | t = {step*DT:g} | L0 step {step}\n'
                     f'{backend} / D2Q9 / YMS; native composite cells, dashed L2 outline; shared scales per frame', fontsize=14)
        filename = f'flow-{step:05d}.png'
        fig.savefig(args.output/filename, dpi=170, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
        print(filename, flush=True)
    (args.output/'color-ranges.csv').write_text('step,component,min,max\n'+''.join(ranges))
    (args.output/'render.txt').write_text(f'matplotlib={matplotlib.__version__}\nsource_id={runs[0].id["source_id"]}\n'
        'field_colors=native composite fluid cells\nstreamlines=nearest composite cell sampling for display only\n'
        'color_scale=shared between layouts at each time; may differ across times\n')


if __name__ == '__main__':
    main()
