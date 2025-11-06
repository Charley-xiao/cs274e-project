# src/sample.py
# Conditional sampling with optional classifier-free guidance,
# plus reporting path length and integrated curvature, and simple visualizations.

from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from .ode import integrate
from .model import create_model

EUROSAT_MEAN_DEFAULT = [0.5, 0.5, 0.5]  # fallback if config doesn't provide
EUROSAT_STD_DEFAULT  = [0.5, 0.5, 0.5]

def unnormalize_to01(x: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    s = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    x01 = x * s + m
    return x01.clamp(0.0, 1.0)

def plot_hist(data, title, out_path, bins=40):
    plt.figure(figsize=(5, 4), dpi=150)
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_traj_pca(traj: torch.Tensor, out_path: str, max_curves: int = 8):
    """
    traj: (S+1, N, D) CPU float tensor; project to 2D with SVD (PCA) and plot curves.
    """
    S1, N, D = traj.shape
    Np = min(N, max_curves)
    X = traj[:, :Np, :].reshape(-1, D)  # ((S+1)*Np, D)
    X = X - X.mean(dim=0, keepdim=True)
    # SVD for PCA
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    W = Vh[:2, :].T  # D x 2
    Y = (traj[:, :Np, :] - traj[:, :Np, :].mean(dim=0, keepdim=True)) @ W  # (S+1, Np, 2)

    plt.figure(figsize=(5, 5), dpi=150)
    for i in range(Np):
        yi = Y[:, i, :].numpy()
        plt.plot(yi[:, 0], yi[:, 1], marker='o', markersize=2, linewidth=1, alpha=0.9)
    plt.title("Trajectory PCA (first 2 PCs)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="samples_out")
    ap.add_argument("--nfe", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--solver", type=str, default=None)               # euler|heun
    ap.add_argument("--classes", type=int, nargs="*", default=None)   # e.g., --classes 0 1 2
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--viz_traces", type=int, default=8, help="store & plot PCA for first N traces per NFE (0=off)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    solver = args.solver or cfg.get("sample", {}).get("solver", "heun")
    img_size = cfg["data"]["image_size"]
    C = cfg.get("model", {}).get("in_channels", 3)
    num_classes = cfg["cond"]["num_classes"]

    mean = cfg.get("data", {}).get("mean", EUROSAT_MEAN_DEFAULT)
    std  = cfg.get("data", {}).get("std",  EUROSAT_STD_DEFAULT)

    # model (EMA if present)
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    state = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict(state, strict=False)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    def make_labels(n: int) -> torch.Tensor | None:
        if args.classes is None or len(args.classes) == 0:
            return torch.full((n,), -1, device=device, dtype=torch.long)
        cls = torch.tensor(args.classes, device=device, dtype=torch.long)
        y = cls.repeat((n + len(cls) - 1) // len(cls))[:n]
        return y

    # JSON report for metrics across NFEs
    report = {}

    for nfe in args.nfe:
        # generate samples + metrics in mini-batches
        total = args.num
        bs = min(64, total)
        imgs = []
        pathlens = []
        curvats = []
        pca_traj_captured = None

        for start in range(0, total, bs):
            b = min(bs, total - start)
            z0 = torch.randn(b, C, img_size, img_size, device=device)
            y  = make_labels(b)

            x1, metrics = integrate(
                lambda X, T, Y: model(X, T, Y),
                z0,
                nfe=nfe,
                solver=solver,
                y=y,
                guidance_scale=args.guidance_scale,
                return_metrics=True,
                traj_for_viz=(args.viz_traces if pca_traj_captured is None else 0),
            )
            imgs.append(x1.cpu())
            pathlens.append(metrics["path_length"])
            curvats.append(metrics["integrated_curvature"])
            if pca_traj_captured is None and "traj" in metrics:
                pca_traj_captured = metrics["traj"]  # (S+1,N,D) on CPU

        X = torch.cat(imgs, dim=0)[:args.num]
        pl = torch.cat(pathlens, dim=0)[:args.num].numpy().tolist()
        kc = torch.cat(curvats, dim=0)[:args.num].numpy().tolist()

        # save grids & individual images (unnormalized to [0,1])
        grid_path = out_dir / f"grid_nfe{nfe}.png"
        save_image(unnormalize_to01(X, mean, std), grid_path, nrow=int(args.num ** 0.5) or 8)
        nfe_dir = out_dir / f"nfe{nfe}"
        nfe_dir.mkdir(exist_ok=True)
        for i, x in enumerate(X):
            save_image(unnormalize_to01(x.unsqueeze(0), mean, std), nfe_dir / f"{i:05d}.png")

        # write metrics JSON
        summary = {
            "nfe": nfe,
            "solver": solver,
            "guidance_scale": args.guidance_scale,
            "num_samples": args.num,
            "path_length": {
                "mean": float(torch.tensor(pl).mean()),
                "std":  float(torch.tensor(pl).std(unbiased=False)),
            },
            "integrated_curvature": {
                "mean": float(torch.tensor(kc).mean()),
                "std":  float(torch.tensor(kc).std(unbiased=False)),
            },
            "per_sample": {
                "path_length": pl,
                "integrated_curvature": kc,
            },
        }
        report[f"nfe_{nfe}"] = summary
        with open(out_dir / f"metrics_nfe{nfe}.json", "w") as f:
            json.dump(summary, f, indent=2)

        # histograms
        plot_hist(pl, f"Path length (nfe={nfe})", out_dir / f"pathlen_hist_nfe{nfe}.png")
        plot_hist(kc, f"Integrated curvature (nfe={nfe})", out_dir / f"curv_hist_nfe{nfe}.png")

        # PCA trajectory viz (only if captured)
        if pca_traj_captured is not None:
            plot_traj_pca(pca_traj_captured, out_dir / f"traj_pca_nfe{nfe}.png", max_curves=min(args.viz_traces, 8))

    # aggregate report across NFEs
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
