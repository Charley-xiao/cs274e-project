from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True)
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"])
    ap.add_argument("--outdir", type=str, required=True)
    return ap.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_npz(indir, method):
    return np.load(Path(indir) / f"{method}.npz", allow_pickle=True)


def violin_plot(data_list, labels, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data_list, showmeans=True, showmedians=False, showextrema=False)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelsize=20)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def line_plot(xs, ys_list, labels, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, lab in zip(ys_list, labels):
        mean = y.mean(axis=1)
        std = y.std(axis=1)
        ax.plot(xs, mean, label=lab)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.18)

    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    path_data = []
    curv_data = []
    profile_data = []
    labels = []

    for m in args.methods:
        arr = load_npz(args.indir, m)
        path_data.append(arr["path_length"])
        curv_data.append(arr["integrated_curvature"])
        profile_data.append(arr["local_angle_profile"])  # [T-2, N]
        labels.append(m)

    violin_plot(
        path_data, labels, "path length",
        Path(args.outdir) / "violin_path_length.pdf"
    )
    violin_plot(
        curv_data, labels, "integrated curvature",
        Path(args.outdir) / "violin_integrated_curvature.pdf"
    )

    # use first file to reconstruct x-axis
    ref = load_npz(args.indir, args.methods[0])
    save_times = ref["save_times"]   # [T]
    xs = save_times[1:-1]            # local angle lives between successive segments

    line_plot(
        xs,
        profile_data,
        labels,
        "local turning angle",
        Path(args.outdir) / "curvature_over_time.pdf"
    )


if __name__ == "__main__":
    main()