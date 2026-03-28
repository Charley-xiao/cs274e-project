from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .model import create_model
from .samplers import integrate
from .util import unnormalize_to01


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="solver_trajectory_vis")
    ap.add_argument("--methods", nargs="+", default=["euler", "ab2", "heun", "midpoint", "rk4", "rk23-adaptive"])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--max_nfe", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

    z0 = torch.randn(1, C, H, W, device=device)
    y = torch.tensor([args.class_id], device=device, dtype=torch.long)

    save_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    row_images = []

    for method in args.methods:
        if method == "rk23-adaptive":
            x1, snapshots = integrate(
                lambda X, T, Y: model(X, T, Y),
                z0,
                method=method,
                nfe=0,
                y=y,
                guidance_scale=args.guidance_scale,
                adaptive=True,
                rtol=args.rtol,
                atol=args.atol,
                max_nfe=args.max_nfe,
                return_snapshots=True,
                save_times=save_times,
            )
        else:
            x1, snapshots = integrate(
                lambda X, T, Y: model(X, T, Y),
                z0,
                method=method,
                nfe=args.steps,
                y=y,
                guidance_scale=args.guidance_scale,
                adaptive=False,
                return_snapshots=True,
                save_times=save_times,
            )

        imgs = [unnormalize_to01(snapshots[t]) for t in save_times]
        row = torch.cat(imgs, dim=0)  # 5 x C x H x W
        row_images.append(row)

        save_image(
            row,
            out_dir / f"{method}_trajectory.png",
            nrow=len(save_times),
        )

    full = torch.cat(row_images, dim=0)
    save_image(
        full,
        out_dir / "all_methods_trajectory_grid.png",
        nrow=len(save_times),
    )

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()