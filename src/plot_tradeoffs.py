import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(json_path: str) -> pd.DataFrame:
    """Load the 'runs' list from your sampler_metrics.json."""
    path = Path(json_path)
    with path.open("r") as f:
        data = json.load(f)
    # support both {"runs": [...]} and plain list
    runs = data["runs"] if isinstance(data, dict) and "runs" in data else data
    df = pd.DataFrame(runs)
    return df


def scatter_with_labels(
    ax,
    x,
    y,
    labels,
    xlabel,
    ylabel,
    title=None,
    annotate=True,
):
    """Scatter plot with text labels and nicer margins."""
    ax.scatter(x, y)

    if annotate:
        # use a small offset in display coords to avoid text overlapping markers
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(
                label,
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 3),
                ha="left",
                fontsize=8,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Add a bit of padding around points
    if len(x) > 0:
        x_min, x_max = min(x), max(x)
        x_pad = (x_max - x_min) * 0.08 if x_max != x_min else 1.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if len(y) > 0:
        y_min, y_max = min(y), max(y)
        y_pad = (y_max - y_min) * 0.08 if y_max != y_min else 1.0
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.grid(True, linestyle="--", alpha=0.3)


def main():
    # path to your JSON file
    json_path = "samples_cmp/sampler_metrics.json"  # change if needed
    df = load_runs(json_path)

    # Ensure output dir exists
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # For your example, we only have one NFE per method, but this works in general.
    # Set an order if you want consistent plotting:
    method_order = ["euler", "ab2", "heun", "midpoint", "rk4", "rk23-adaptive"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values("method")

    methods = df["method"].astype(str).tolist()
    fid = df["fid"].values
    eff_nfe = df["effective_nfe"].values
    time_sec = df["wall_time_sec"].values
    L = df["path_length_mean"].values
    C = df["integrated_curvature_mean"].values
    time_per_eff_nfe = time_sec / eff_nfe

    # ========== Individual PNGs (if you still want them) ==========
    # 1) FID vs time per effective NFE
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter_with_labels(
        ax,
        time_per_eff_nfe,
        fid,
        methods,
        xlabel="Time (s) per effective NFE",
        ylabel="FID (↓)",
        title=None,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fid_vs_t_per_nfe.png", dpi=200)
    plt.close(fig)

    # 2) FID vs effective NFE
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter_with_labels(
        ax,
        eff_nfe,
        fid,
        methods,
        xlabel="Effective NFE",
        ylabel="FID (↓)",
        title="FID vs Effective NFE",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fid_vs_eff_nfe.png", dpi=200)
    plt.close(fig)

    # 3) FID vs path length
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter_with_labels(
        ax,
        L,
        fid,
        methods,
        xlabel="Path length $L$ (mean)",
        ylabel="FID (↓)",
        title="FID vs Path Length",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fid_vs_path_length.png", dpi=200)
    plt.close(fig)

    # 4) FID vs integrated curvature
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter_with_labels(
        ax,
        C,
        fid,
        methods,
        xlabel="Integrated curvature $\\mathcal{C}$ (mean)",
        ylabel="FID (↓)",
        title="FID vs Curvature",
    )
    plt.tight_layout()
    fig.savefig(out_dir / "fid_vs_curvature.png", dpi=200)
    plt.close(fig)

    # Optional: print the table nicely
    print(
        df[
            [
                "method",
                "effective_nfe",
                "wall_time_sec",
                "fid",
                "path_length_mean",
                "integrated_curvature_mean",
            ]
        ]
    )

    # ========== Combined 2×2 PDF ==========
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    scatter_with_labels(
        ax1,
        time_per_eff_nfe,
        fid,
        methods,
        xlabel="Time (s) per effective NFE",
        ylabel="FID (↓)",
        title=None,
    )

    scatter_with_labels(
        ax2,
        eff_nfe,
        fid,
        methods,
        xlabel="Effective NFE",
        ylabel="FID (↓)",
        title=None,
    )

    scatter_with_labels(
        ax3,
        L,
        fid,
        methods,
        xlabel="Path length $L$ (mean)",
        ylabel="FID (↓)",
        title=None,
    )

    scatter_with_labels(
        ax4,
        C,
        fid,
        methods,
        xlabel="Integrated curvature $\\mathcal{C}$ (mean)",
        ylabel="FID (↓)",
        title=None,
    )

    plt.tight_layout()
    fig.savefig(out_dir / "tradeoffs.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
