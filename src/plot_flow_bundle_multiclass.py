from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import gaussian_kde
from matplotlib.colors import to_rgba
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.feature_extractors import build_feature_extractor
from src.model import create_model

CLASS_TO_LABEL = {
    0: "Annual Crop",
    1: "Forest",
    2: "Herbaceous Vegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "Permanent Crop",
    7: "Residential",
    8: "River",
    9: "Sea/Lake",
}


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cfg", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--classes", type=int, nargs="+", required=True,
                    help="Class IDs to visualize, e.g. --classes 0 1 2 3")
    ap.add_argument("--samples_per_class", type=int, default=80)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dpi", type=int, default=220)

    # plotting
    ap.add_argument("--line_alpha", type=float, default=0.05)
    ap.add_argument("--point_alpha", type=float, default=0.65)
    ap.add_argument("--point_size", type=float, default=8.0)
    ap.add_argument("--line_width", type=float, default=0.8)
    ap.add_argument("--show_legend", action="store_true")

    # features
    ap.add_argument("--feature_extractor", type=str, default="dinov2",
                    choices=["inception", "dinov2"])
    ap.add_argument("--feature_batch_size", type=int, default=64)

    # cache
    ap.add_argument("--save_cached_features", action="store_true")
    ap.add_argument("--load_cached_features", type=str, default=None,
                    help="Path to cached .npz with keys X and class_ids")

    # unified reducer config
    ap.add_argument("--reducer", type=str, default="pca", choices=["pca", "tsne"])
    ap.add_argument("--fit_on", type=str, default="endpoints",
                    choices=["all", "endpoints", "final_only"],
                    help="Which states are used to fit/embed the reducer")

    # t-SNE options
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_learning_rate", type=float, default=200.0)
    ap.add_argument("--tsne_max_iter", type=int, default=1000)

    return ap.parse_args()


def load_cfg(args) -> Dict:
    if args.cfg is not None:
        with open(args.cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        return ckpt["cfg"]

    raise ValueError("Could not find cfg in checkpoint. Please pass --cfg explicitly.")


@torch.no_grad()
def model_velocity(model, x, t_scalar: float, y: Optional[torch.Tensor]):
    b = x.shape[0]
    t = torch.full((b,), float(t_scalar), device=x.device, dtype=x.dtype)
    return model(x, t, y)


@torch.no_grad()
def integrate_trajectories(
    model,
    z0: torch.Tensor,
    y: Optional[torch.Tensor],
    steps: int,
    solver: str,
):
    assert solver in {"euler", "heun"}
    x = z0.clone()
    states = [x.detach().cpu()]

    h = 1.0 / steps
    for k in range(steps):
        t = k / steps

        if solver == "euler":
            v = model_velocity(model, x, t, y)
            x = x + h * v
        else:
            v1 = model_velocity(model, x, t, y)
            x_pred = x + h * v1
            v2 = model_velocity(model, x_pred, t + h, y)
            x = x + 0.5 * h * (v1 + v2)

        states.append(x.detach().cpu())

    return torch.stack(states, dim=0)  # [T, B, C, H, W]


@torch.no_grad()
def extract_state_features(
    states_np: np.ndarray,   # [T, B, C, H, W]
    feature_extractor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    T, B, C, H, W = states_np.shape
    feats_all = []

    for t_idx in trange(T, desc="Extracting features from states"):
        x = torch.from_numpy(states_np[t_idx]).to(device=device, dtype=torch.float32)
        feats_t = []
        for i in range(0, B, batch_size):
            xb = x[i:i + batch_size]
            fb = feature_extractor(xb)
            feats_t.append(fb.detach().cpu().numpy())
        feats_t = np.concatenate(feats_t, axis=0)
        feats_all.append(feats_t)

    return np.stack(feats_all, axis=0)  # [T, B, D]


def get_colors(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_multiclass_bundle(
    traj_2d: np.ndarray,
    class_ids: np.ndarray,
    unique_classes: List[int],
    outpath: Path,
    dpi: int,
    line_alpha: float,
    point_alpha: float,
    point_size: float,
    line_width: float,
    show_legend: bool,
):
    T, B, _ = traj_2d.shape
    start = traj_2d[0]
    end = traj_2d[-1]

    colors = get_colors(len(unique_classes))
    cls_to_color = {c: colors[i] for i, c in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    for i in range(B):
        c = int(class_ids[i])
        color = cls_to_color[c]
        ax.plot(
            traj_2d[:, i, 0],
            traj_2d[:, i, 1],
            alpha=line_alpha,
            linewidth=line_width,
            color=color,
            zorder=1,
        )

    for c in unique_classes:
        mask = class_ids == c
        color = cls_to_color[c]
        ax.scatter(
            start[mask, 0], start[mask, 1],
            s=point_size * 0.35,
            alpha=0.12,
            color=color,
            zorder=2,
        )

    all_pts = np.concatenate([start, end], axis=0)
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.08 * dx
    xmax += 0.08 * dx
    ymin -= 0.08 * dy
    ymax += 0.08 * dy

    xx, yy = np.mgrid[xmin:xmax:240j, ymin:ymax:240j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for c in unique_classes:
        mask = class_ids == c
        pts = end[mask]
        color = cls_to_color[c]

        if pts.shape[0] >= 5:
            try:
                kde = gaussian_kde(pts.T)
                z = kde(grid).reshape(xx.shape)
                z = z / (z.max() + 1e-12)

                ax.contourf(
                    xx, yy, z,
                    levels=[0.10, 0.22, 0.38, 0.58, 0.80, 1.01],
                    colors=[to_rgba(color, a) for a in [0.04, 0.06, 0.08, 0.11, 0.14]],
                    zorder=2.2,
                    antialiased=True,
                )

                ax.contour(
                    xx, yy, z,
                    levels=[0.22, 0.45, 0.70],
                    colors=[to_rgba(color, 0.45)],
                    linewidths=0.9,
                    zorder=2.4,
                )
            except Exception:
                pass

    for c in unique_classes:
        mask = class_ids == c
        color = cls_to_color[c]
        ax.scatter(
            end[mask, 0], end[mask, 1],
            s=point_size,
            alpha=point_alpha,
            color=color,
            edgecolors="white",
            linewidths=0.25,
            label=CLASS_TO_LABEL[c] if show_legend else None,
            zorder=3,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.12)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_h.append(h)
                uniq_l.append(l)
                seen.add(l)
        ax.legend(uniq_h, uniq_l, frameon=True, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_model_from_ckpt(cfg: Dict, ckpt_path: str, device: torch.device):
    num_classes = cfg["cond"]["num_classes"]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]), strict=False)
    return model


def maybe_cache_features(X: np.ndarray, class_ids: np.ndarray, args):
    if not args.save_cached_features:
        return None

    class_str = "-".join(str(c) for c in args.classes)
    cache_path = Path(args.outdir) / (
        f"cached_features_{args.feature_extractor}_"
        f"cls{class_str}_steps{args.steps}.npz"
    )
    np.savez_compressed(cache_path, X=X, class_ids=class_ids)
    print(f"[cached features] {cache_path}")
    return cache_path


def load_cached_features(cache_path: str):
    arr = np.load(cache_path, allow_pickle=False)
    return arr["X"], arr["class_ids"]

def subset_cached_features_by_classes(
    X: np.ndarray,            # [T, B, D]
    class_ids: np.ndarray,    # [B]
    selected_classes: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    selected_classes = [int(c) for c in selected_classes]
    mask = np.isin(class_ids, selected_classes)

    if not np.any(mask):
        raise ValueError(
            f"No cached samples match requested classes: {selected_classes}. "
            f"Available classes are: {sorted(np.unique(class_ids).tolist())}"
        )

    X_sub = X[:, mask, :]
    class_ids_sub = class_ids[mask]

    # preserve the order requested by the user, but only keep classes that exist
    present = set(class_ids_sub.tolist())
    unique_classes = [c for c in selected_classes if c in present]

    return X_sub, class_ids_sub, unique_classes


def reduce_with_pca(X: np.ndarray, fit_on: str, seed: int) -> np.ndarray:
    T, B, D = X.shape

    if fit_on == "all":
        fit_X = X.reshape(T * B, D)
    elif fit_on == "endpoints":
        fit_X = np.concatenate([X[0], X[-1]], axis=0)
    else:  # final_only
        fit_X = X[-1]

    reducer = PCA(n_components=2, random_state=seed)
    reducer.fit(fit_X)
    traj_2d = reducer.transform(X.reshape(T * B, D)).reshape(T, B, 2)
    return traj_2d


def reduce_with_tsne(X: np.ndarray, fit_on: str, seed: int,
                     perplexity: float, learning_rate: float, max_iter: int) -> np.ndarray:
    T, B, D = X.shape

    if fit_on == "all":
        tsne_input = X.reshape(T * B, D)
        mode = "all"
    elif fit_on == "endpoints":
        X_sel = np.stack([X[0], X[-1]], axis=0)   # [2, B, D]
        tsne_input = X_sel.reshape(2 * B, D)
        mode = "endpoints"
    else:
        tsne_input = X[-1]
        mode = "final_only"

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        init="pca",
        random_state=seed,
    )
    Y = reducer.fit_transform(tsne_input)

    if mode == "all":
        traj_2d = Y.reshape(T, B, 2)
    elif mode == "endpoints":
        Y2 = Y.reshape(2, B, 2)
        start_2d = Y2[0]
        end_2d = Y2[1]
        alphas = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None, None]
        traj_2d = (1.0 - alphas) * start_2d[None, :, :] + alphas * end_2d[None, :, :]
    else:
        end_2d = Y
        center = end_2d.mean(axis=0, keepdims=True)
        rng = np.random.default_rng(seed)
        scale = 0.03 * np.maximum(np.std(end_2d, axis=0, keepdims=True), 1e-6)
        start_2d = center + scale * rng.standard_normal(size=(B, 2))
        alphas = np.linspace(0.0, 1.0, T, dtype=np.float32)[:, None, None]
        traj_2d = (1.0 - alphas) * start_2d[None, :, :] + alphas * end_2d[None, :, :]

    return traj_2d


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.load_cached_features is not None:
        print(f"[loading cached features] {args.load_cached_features}")
        X, class_ids = load_cached_features(args.load_cached_features)

        # if user requested a subset of classes, filter cached features here
        X, class_ids, unique_classes = subset_cached_features_by_classes(
            X=X,
            class_ids=class_ids,
            selected_classes=args.classes,
        )

        print(f"[cached subset] using classes: {unique_classes}, total samples: {class_ids.shape[0]}")
    else:
        cfg = load_cfg(args)
        model = build_model_from_ckpt(cfg, args.ckpt, device)

        image_size = cfg["data"]["image_size"]
        in_channels = cfg["model"].get("in_channels", 3)

        all_states = []
        all_class_ids = []

        g = torch.Generator(device=device)
        g.manual_seed(args.seed)

        for class_id in tqdm(args.classes, desc="Integrating trajectories for classes"):
            z0 = torch.randn(
                args.samples_per_class,
                in_channels,
                image_size,
                image_size,
                generator=g,
                device=device,
            )
            y = torch.full(
                (args.samples_per_class,),
                int(class_id),
                dtype=torch.long,
                device=device,
            )

            states = integrate_trajectories(
                model=model,
                z0=z0,
                y=y,
                steps=args.steps,
                solver=args.solver,
            )

            all_states.append(states.numpy())
            all_class_ids.extend([int(class_id)] * args.samples_per_class)

        states_np = np.concatenate(all_states, axis=1)
        class_ids = np.array(all_class_ids, dtype=np.int64)

        feature_extractor = build_feature_extractor(args.feature_extractor).to(device).eval()
        X = extract_state_features(
            states_np=states_np,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=args.feature_batch_size,
        )
        maybe_cache_features(X, class_ids, args)
        unique_classes = list(args.classes)

    if args.reducer == "pca":
        traj_2d = reduce_with_pca(X, args.fit_on, args.seed)
    else:
        traj_2d = reduce_with_tsne(
            X=X,
            fit_on=args.fit_on,
            seed=args.seed,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_learning_rate,
            max_iter=args.tsne_max_iter,
        )

    class_str = "-".join(str(c) for c in unique_classes)
    outpath = Path(args.outdir) / (
        f"bundle_multiclass_{args.feature_extractor}_{args.reducer}_{args.solver}_"
        f"cls{class_str}_steps{args.steps}.pdf"
    )

    plot_multiclass_bundle(
        traj_2d=traj_2d,
        class_ids=class_ids,
        unique_classes=unique_classes,
        outpath=outpath,
        dpi=args.dpi,
        line_alpha=args.line_alpha,
        point_alpha=args.point_alpha,
        point_size=args.point_size,
        line_width=args.line_width,
        show_legend=args.show_legend,
    )

    meta = {
        "ckpt": args.ckpt,
        "cfg": args.cfg,
        "classes": unique_classes,
        "samples_per_class": args.samples_per_class,
        "steps": args.steps,
        "solver": args.solver,
        "seed": args.seed,
        "feature_extractor": args.feature_extractor,
        "reducer": args.reducer,
        "fit_on": args.fit_on,
        "tsne_perplexity": args.tsne_perplexity,
        "tsne_learning_rate": args.tsne_learning_rate,
        "tsne_max_iter": args.tsne_max_iter,
        "load_cached_features": args.load_cached_features,
        "output": str(outpath),
    }
    with open(Path(args.outdir) / "bundle_multiclass_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {outpath}")


if __name__ == "__main__":
    main()