from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--sens_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    return ap.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_heatmap(pivot_df: pd.DataFrame, title: str, outpath: Path, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(pivot_df.values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(title, rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)
    sens = pd.read_csv(args.sens_csv)

    # keep sampler order stable
    sampler_order = ["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"]
    class_order = list(df.sort_values("class_id")["class_name"].drop_duplicates())

    fid_pivot = df.pivot(index="sampler", columns="class_name", values="fid").reindex(index=sampler_order, columns=class_order)
    l_pivot = df.pivot(index="sampler", columns="class_name", values="path_length_mean").reindex(index=sampler_order, columns=class_order)
    c_pivot = df.pivot(index="sampler", columns="class_name", values="curvature_mean").reindex(index=sampler_order, columns=class_order)

    save_heatmap(fid_pivot, "Per-class FID", Path(args.outdir) / "per_class_fid_heatmap.png")
    save_heatmap(l_pivot, "Per-class path length", Path(args.outdir) / "per_class_path_length_heatmap.png")
    save_heatmap(c_pivot, "Per-class integrated curvature", Path(args.outdir) / "per_class_curvature_heatmap.png")

    # sensitivity bar
    sens = sens.sort_values("fid_range", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(sens["class_name"], sens["fid_range"])
    ax.set_ylabel("FID range across samplers")
    ax.set_title("Class sensitivity to sampler choice")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(Path(args.outdir) / "class_fid_sensitivity_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()