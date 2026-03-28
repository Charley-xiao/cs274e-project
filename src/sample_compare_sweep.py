from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics

from .model import create_model
from .samplers import integrate
from .util import unnormalize_to01


# ============================================================
# Metrics
# ============================================================

def _compute_fid_is(real_dir: str, fake_dir: str, use_cuda: bool, batch_size: int = 64) -> Dict[str, Any]:
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=use_cuda,
        batch_size=batch_size,
        isc=True,
        fid=True,
        verbose=False,
        samples_find_deep=True,
    )
    return {
        "fid": float(metrics["frechet_inception_distance"]),
        "isc_mean": float(metrics["inception_score_mean"]),
        "isc_std": float(metrics["inception_score_std"]),
    }


# ============================================================
# Plotting
# ============================================================

def _plot_fid_vs_nfe(results, out_dir, annotate=False, zoom_xmax=6000):

    if len(results) == 0:
        return

    df = pd.DataFrame(results)
    if "fid" not in df.columns:
        print("[plot] No FID found in results; skip plotting.")
        return

    df = df.dropna(subset=["effective_nfe", "fid", "method"]).copy()
    if df.empty:
        print("[plot] No valid rows for plotting.")
        return

    method_order = ["euler", "ab2", "heun", "midpoint", "rk4", "rk23-adaptive"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["method", "effective_nfe"])

    def draw_one(ax, data, use_logx=True, xlim=None, title=None):
        grouped = data.groupby("method", observed=True)
        for method, g in grouped:
            g = g.sort_values("effective_nfe")
            x = g["effective_nfe"].values / 32
            y = g["fid"].values
            ax.plot(x, y, marker="o", linewidth=1.8, label=str(method))

            if annotate:
                # only annotate first and last point to reduce clutter
                idxs = [0]
                if len(g) > 1:
                    idxs.append(len(g) - 1)
                for idx in sorted(set(idxs)):
                    row = g.iloc[idx]
                    xi = row["effective_nfe"]
                    yi = row["fid"]
                    if row.get("is_adaptive", False):
                        label = f"{row['rtol']:.0e}"
                    else:
                        label = f"{int(row['steps'])}"
                    ax.annotate(
                        label,
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(4, 3),
                        ha="left",
                        fontsize=8,
                    )

        if use_logx:
            ax.set_xscale("log")

        ax.set_xlabel("Avg NFE")
        ax.set_ylabel("FID (↓)")
        if title is not None:
            ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()

        if xlim is not None:
            ax.set_xlim(*xlim)

    # ---------- Main log-x figure ----------
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    draw_one(ax, df, use_logx=True, title=None)
    plt.tight_layout()
    fig.savefig(out_dir / "fid_vs_nfe_curves_logx.png", dpi=220)
    fig.savefig(out_dir / "fid_vs_nfe_curves_logx.pdf")
    plt.close(fig)

    # ---------- Zoomed low-budget figure ----------
    df_zoom = df[df["effective_nfe"] <= zoom_xmax].copy()
    if not df_zoom.empty:
        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        draw_one(
            ax,
            df_zoom,
            use_logx=False,
            xlim=(0, zoom_xmax / 32),
            title=None,
        )
        plt.tight_layout()
        fig.savefig(out_dir / "fid_vs_nfe_curves_zoom.png", dpi=220)
        fig.savefig(out_dir / "fid_vs_nfe_curves_zoom.pdf")
        plt.close(fig)

    print(f"[plot] Saved: {out_dir / 'fid_vs_nfe_curves_logx.pdf'}")
    if not df_zoom.empty:
        print(f"[plot] Saved: {out_dir / 'fid_vs_nfe_curves_zoom.pdf'}")


# ============================================================
# Sampling helpers
# ============================================================

def _make_labels(num: int, classes: List[int] | None, device: torch.device) -> torch.Tensor:
    if not classes:
        return torch.full((num,), -1, device=device, dtype=torch.long)
    cls = torch.tensor(classes, device=device, dtype=torch.long)
    return cls.repeat((num + len(cls) - 1) // len(cls))[:num]


def _save_fake_images(X: torch.Tensor, fake_dir: Path) -> None:
    fake_dir.mkdir(parents=True, exist_ok=True)
    for i, x in enumerate(X):
        save_image(unnormalize_to01(x.unsqueeze(0)), fake_dir / f"{i:05d}.png", nrow=1)


def _run_one_config(
    *,
    model,
    device: torch.device,
    method: str,
    z0: torch.Tensor,
    y: torch.Tensor,
    steps: int | None,
    guidance_scale: float,
    adaptive: bool,
    rtol: float,
    atol: float,
    max_nfe: int,
    out_dir: Path,
    real_dir: str | None,
    fid_batch: int,
    no_cuda_fid: bool,
) -> Dict[str, Any]:
    C, H, W = z0.shape[1], z0.shape[2], z0.shape[3]
    num = z0.shape[0]

    X_chunks = []
    pl_all = []
    ic_all = []

    total_fe = 0
    bs = min(num, 64)

    t_start = time.time()

    for i in range(0, num, bs):
        z = z0[i:i + bs]
        yb = y[i:i + bs]

        x1, metrics, perf = integrate(
            lambda X, T, Y: model(X, T, Y),
            z,
            method=method,
            nfe=0 if adaptive else int(steps),
            y=yb,
            guidance_scale=guidance_scale,
            adaptive=adaptive,
            rtol=rtol,
            atol=atol,
            max_nfe=max_nfe,
            return_metrics=True,
        )

        X_chunks.append(x1.cpu())
        pl_all.append(metrics["path_length"].cpu())
        ic_all.append(metrics["integrated_curvature"].cpu())
        total_fe += int(perf["nfe"])

    wall = time.time() - t_start

    X = torch.cat(X_chunks, dim=0)
    PL = torch.cat(pl_all, dim=0)
    IC = torch.cat(ic_all, dim=0)

    if adaptive:
        tag = f"{method}_rtol{rtol:.0e}_atol{atol:.0e}"
    else:
        tag = f"{method}_steps{steps}"

    grid_path = out_dir / f"grid_{tag}.png"
    save_image(unnormalize_to01(X), grid_path, nrow=10)

    fake_dir = out_dir / tag
    _save_fake_images(X, fake_dir)

    stats: Dict[str, Any] = {
        "method": method,
        "is_adaptive": bool(adaptive),
        "steps": None if adaptive else int(steps),
        "rtol": float(rtol) if adaptive else None,
        "atol": float(atol) if adaptive else None,
        "max_nfe_arg": int(max_nfe) if adaptive else None,
        "effective_nfe": int(total_fe),
        "wall_time_sec": float(wall),
        "num_samples": int(num),
        "guidance_scale": float(guidance_scale),
        "path_length_mean": float(PL.mean()),
        "path_length_std": float(PL.std(unbiased=False)),
        "integrated_curvature_mean": float(IC.mean()),
        "integrated_curvature_std": float(IC.std(unbiased=False)),
        "grid_path": str(grid_path),
        "fake_dir": str(fake_dir),
    }

    if real_dir:
        use_cuda_fid = torch.cuda.is_available() and (not no_cuda_fid)
        fidis = _compute_fid_is(real_dir, str(fake_dir), use_cuda_fid, batch_size=fid_batch)
        stats.update(fidis)
        print(
            f"[{tag}] wall={wall:.2f}s FE={total_fe} "
            f"FID={fidis['fid']:.3f} IS={fidis['isc_mean']:.3f}±{fidis['isc_std']:.3f} "
            f"PLμ={stats['path_length_mean']:.4f} ICμ={stats['integrated_curvature_mean']:.4f}"
        )
    else:
        print(
            f"[{tag}] wall={wall:.2f}s FE={total_fe} "
            f"PLμ={stats['path_length_mean']:.4f} ICμ={stats['integrated_curvature_mean']:.4f}"
        )

    return stats


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="samples_sweep")

    # non-adaptive methods
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["euler", "heun", "midpoint", "rk4", "ab2", "rk23-adaptive"],
    )
    ap.add_argument(
        "--steps_list",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="For non-adaptive samplers.",
    )

    # adaptive RK23 sweep
    ap.add_argument(
        "--adaptive_rtol_list",
        type=float,
        nargs="+",
        default=[1e-2, 5e-3, 2e-3, 1e-3, 5e-4],
        help="For rk23-adaptive.",
    )
    ap.add_argument("--adaptive_atol", type=float, default=1e-4)
    ap.add_argument("--max_nfe", type=int, default=4000)

    ap.add_argument("--num", type=int, default=2048)
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--guidance_scale", type=float, default=0.0)

    ap.add_argument("--metrics_out", type=str, default="sampler_sweep_metrics.json")

    # FID / IS
    ap.add_argument("--real_dir", type=str, default=None)
    ap.add_argument("--fid_batch", type=int, default=64)
    ap.add_argument("--no_cuda_fid", action="store_true")

    # reproducibility
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--read_from", type=str, default=None, help="Path to existing metrics JSON to read and append to (instead of starting fresh).")

    args = ap.parse_args()

    if args.read_from:
        with open(args.read_from, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing.get("runs", [])

        _plot_fid_vs_nfe(results, Path(args.out))
        print(f"[main] Loaded {len(results)} existing runs from {args.read_from}.")
        return

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    C = cfg.get("model", {}).get("in_channels", 3)
    H = W = cfg["data"]["image_size"]
    num_classes = cfg["cond"]["num_classes"]

    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    state = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict(state, strict=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    y = _make_labels(args.num, args.classes, device)
    z0 = torch.randn(args.num, C, H, W, device=device)

    results: List[Dict[str, Any]] = []

    # -----------------------------
    # non-adaptive sweep
    # -----------------------------
    for method in args.methods:
        if method == "rk23-adaptive":
            continue

        for steps in args.steps_list:
            stats = _run_one_config(
                model=model,
                device=device,
                method=method,
                z0=z0,
                y=y,
                steps=steps,
                guidance_scale=args.guidance_scale,
                adaptive=False,
                rtol=1e-3,
                atol=1e-4,
                max_nfe=args.max_nfe,
                out_dir=out_dir,
                real_dir=args.real_dir,
                fid_batch=args.fid_batch,
                no_cuda_fid=args.no_cuda_fid,
            )
            results.append(stats)

            with open(out_dir / args.metrics_out, "w", encoding="utf-8") as f:
                json.dump({"runs": results}, f, indent=2)

    # -----------------------------
    # adaptive sweep
    # -----------------------------
    if "rk23-adaptive" in args.methods:
        for rtol in args.adaptive_rtol_list:
            stats = _run_one_config(
                model=model,
                device=device,
                method="rk23-adaptive",
                z0=z0,
                y=y,
                steps=None,
                guidance_scale=args.guidance_scale,
                adaptive=True,
                rtol=float(rtol),
                atol=float(args.adaptive_atol),
                max_nfe=args.max_nfe,
                out_dir=out_dir,
                real_dir=args.real_dir,
                fid_batch=args.fid_batch,
                no_cuda_fid=args.no_cuda_fid,
            )
            results.append(stats)

            with open(out_dir / args.metrics_out, "w", encoding="utf-8") as f:
                json.dump({"runs": results}, f, indent=2)

    # final save
    with open(out_dir / args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"runs": results}, f, indent=2)

    # plot
    _plot_fid_vs_nfe(results, out_dir)


if __name__ == "__main__":
    main()