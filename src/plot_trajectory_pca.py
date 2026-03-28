from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True)
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"])
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--num_traj", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.05)
    return ap.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    loaded = {}
    all_feats = []

    for m in args.methods:
        arr = np.load(Path(args.indir) / f"{m}.npz", allow_pickle=True)
        snaps = arr["snapshots"]   # [T, N, C, H, W]
        loaded[m] = snaps
        T, N, C, H, W = snaps.shape
        feats = snaps.reshape(T * N, C * H * W)
        all_feats.append(feats)

    X = np.concatenate(all_feats, axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(X)

    projected = {}
    all_xy = []

    for m in args.methods:
        snaps = loaded[m]
        T, N, C, H, W = snaps.shape
        num_traj = min(args.num_traj, N)

        traj_list = []
        for i in range(num_traj):
            traj = snaps[:, i].reshape(T, C * H * W)
            z = pca.transform(traj)   # [T, 2]
            traj_list.append(z)
            all_xy.append(z)

        projected[m] = traj_list

    all_xy = np.concatenate(all_xy, axis=0)
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= args.margin * dx
    xmax += args.margin * dx
    ymin -= args.margin * dy
    ymax += args.margin * dy

    for m in args.methods:
        fig, ax = plt.subplots(figsize=(6, 6))

        for z in projected[m]:
            ax.plot(z[:, 0], z[:, 1], marker="o", markersize=3, linewidth=1.5, alpha=0.9)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(m)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(args.outdir) / f"pca_{m}_sharedaxes.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # optional: one combined panel
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, m in zip(axes, args.methods):
        for z in projected[m]:
            ax.plot(z[:, 0], z[:, 1], marker="o", markersize=2.5, linewidth=1.2, alpha=0.9)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(m)
        ax.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.tight_layout()
    fig.savefig(Path(args.outdir) / "pca_all_methods_sharedaxes.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()