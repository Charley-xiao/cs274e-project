from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image

from src.model import create_model
from src.samplers import integrate


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"],
    )
    ap.add_argument("--nfe", type=int, default=64)
    ap.add_argument("--num_per_class", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--classes", type=int, nargs="*", default=list(range(10)))

    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")

    # adaptive RK23
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--max_nfe", type=int, default=2000)

    return ap.parse_args()


def load_cfg(cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if len(state_dict) > 0 and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_from_ckpt(ckpt_path: str, cfg: Dict, device: torch.device):
    num_classes = cfg["cond"]["num_classes"]
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        sd = ckpt["ema_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    sd = strip_prefix_if_needed(sd, "model.")
    sd = strip_prefix_if_needed(sd, "module.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_model] missing keys: {len(missing)}")
    print(f"[load_model] unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    # x: [C,H,W], assumed roughly in [-1,1] or similar normalized range
    x = x.detach().float().cpu()

    # robust min-max per image for saving class-conditional galleries
    x = x - x.min()
    x = x / (x.max() + 1e-8)

    x = (x * 255.0).clamp(0, 255).byte()
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)

@torch.no_grad()
def main():
    args = parse_args()
    ensure_dir(args.outdir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_cfg(args.cfg)
    num_classes = cfg["cond"]["num_classes"]
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]), strict=False)

    image_size = cfg["data"]["image_size"]
    in_channels = cfg["model"].get("in_channels", 3)

    for method in args.methods:
        print(f"\n=== sampler: {method} ===")
        for cls in args.classes:
            class_dir = Path(args.outdir) / method / f"class_{cls}"
            ensure_dir(class_dir)

            done = 0
            batch_idx = 0

            while done < args.num_per_class:
                bs = min(args.batch_size, args.num_per_class - done)

                z0 = torch.randn(bs, in_channels, image_size, image_size, device=device)
                y = torch.full((bs,), cls, dtype=torch.long, device=device)

                kwargs = dict(
                    v_theta=model,
                    z0=z0,
                    method=method,
                    nfe=args.nfe,
                    y=y,
                    guidance_scale=args.guidance_scale,
                    return_metrics=False,
                    return_snapshots=False,
                )

                if method == "rk23-adaptive":
                    kwargs.update(
                        adaptive=True,
                        rtol=args.rtol,
                        atol=args.atol,
                        max_nfe=args.max_nfe,
                    )

                out = integrate(**kwargs)

                # tolerate either x_final-only or tuple-first-return
                if isinstance(out, tuple):
                    x_final = out[0]
                else:
                    x_final = out

                for i in range(bs):
                    img = tensor_to_pil(x_final[i])
                    img.save(class_dir / f"{done + i:05d}.png")

                done += bs
                batch_idx += 1
                print(f"[{method}] class={cls} {done}/{args.num_per_class}")

    print("[done] class-conditional samples saved.")


if __name__ == "__main__":
    main()