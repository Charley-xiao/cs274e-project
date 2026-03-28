from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from src.model import create_model
from src.samplers import integrate


# --------------------------
# utils
# --------------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_fixed_inputs(cfg: Dict, args, device: torch.device):
    image_size = cfg["data"]["image_size"]
    in_channels = cfg["model"].get("in_channels", 3)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    z_all = torch.randn(
        args.num_samples,
        in_channels,
        image_size,
        image_size,
        generator=g,
        device=device,
    )

    if args.classes is None or len(args.classes) == 0:
        y_all = None
    else:
        ys = [args.classes[i % len(args.classes)] for i in range(args.num_samples)]
        y_all = torch.tensor(ys, dtype=torch.long, device=device)

    return z_all, y_all

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--cfg", type=str, default=None, help="Optional YAML config if not stored in ckpt")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["euler", "midpoint", "heun", "rk4", "ab2", "rk23-adaptive"])
    ap.add_argument("--nfe", type=int, default=16, help="Fixed-step budget for non-adaptive samplers")
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="If given, cycle through these classes")
    ap.add_argument("--guidance_scale", type=float, default=0.0)

    # adaptive RK23 args
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--max_nfe", type=int, default=2000)

    # saved snapshots for PCA trajectory plots
    ap.add_argument("--save_times", type=float, nargs="*",
                    default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def load_cfg(args) -> Dict:
    if args.cfg is not None:
        with open(args.cfg, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        return ckpt["cfg"]

    raise ValueError(
        "Could not find cfg in checkpoint. Please pass --cfg configs/rf.yaml explicitly."
    )


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if all(k.startswith(prefix) for k in state_dict.keys()):
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

    # common cases
    sd = strip_prefix_if_needed(sd, "model.")
    sd = strip_prefix_if_needed(sd, "module.")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load_model] missing keys:", len(missing))
    print("[load_model] unexpected keys:", len(unexpected))

    model.eval()
    return model


def make_labels(num: int, classes: Optional[List[int]], device: torch.device):
    if classes is None or len(classes) == 0:
        return None
    ys = [classes[i % len(classes)] for i in range(num)]
    return torch.tensor(ys, dtype=torch.long, device=device)


def compute_local_angle_profile(states: torch.Tensor) -> np.ndarray:
    """
    states: [T, B, C, H, W]
    returns: [T-2, B] local turning angles
    """
    dx = states[1:] - states[:-1]                 # [T-1, B, C, H, W]
    flat = dx.flatten(2)                          # [T-1, B, D]
    norms = flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = flat / norms                              # [T-1, B, D]

    dots = (u[:-1] * u[1:]).sum(dim=-1).clamp(-1.0, 1.0)  # [T-2, B]
    ang = torch.acos(dots)
    return ang.cpu().numpy()


@torch.no_grad()
def run_one_method(
    model,
    method: str,
    args,
    cfg: Dict,
    device: torch.device,
    z_all: torch.Tensor,
    y_all: Optional[torch.Tensor],
):
    all_path = []
    all_curv = []
    all_labels = []
    all_profiles = []
    snapshot_bank = []
    perf_bank = []

    num_done = 0

    while num_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - num_done)

        z0 = z_all[num_done:num_done + bs]
        y = None if y_all is None else y_all[num_done:num_done + bs]

        kwargs = dict(
            v_theta=model,
            z0=z0,
            method=method,
            nfe=args.nfe,
            y=y,
            guidance_scale=args.guidance_scale,
            return_metrics=True,
            save_times=args.save_times,
            return_snapshots=True,
        )

        if method == "rk23-adaptive":
            kwargs.update(
                adaptive=True,
                rtol=args.rtol,
                atol=args.atol,
                max_nfe=args.max_nfe,
            )

        out = integrate(**kwargs)
        x_final, metrics, perf, snapshots = out

        ordered_times = sorted(snapshots.keys())
        ordered_states = [snapshots[t] for t in ordered_times]
        states = torch.stack(ordered_states, dim=0)

        prof = compute_local_angle_profile(states)

        all_path.append(metrics["path_length"].cpu().numpy())
        all_curv.append(metrics["integrated_curvature"].cpu().numpy())
        all_profiles.append(prof)

        if y is None:
            all_labels.append(np.full((bs,), -1, dtype=np.int64))
        else:
            all_labels.append(y.detach().cpu().numpy())

        snapshot_bank.append(states.cpu().numpy())
        perf_bank.append(perf)

        num_done += bs
        print(f"[{method}] {num_done}/{args.num_samples} done, nfe={perf.get('nfe', 'NA')}")

    path_arr = np.concatenate(all_path, axis=0)
    curv_arr = np.concatenate(all_curv, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)
    profile_arr = np.concatenate(all_profiles, axis=1)
    snaps = np.concatenate(snapshot_bank, axis=1)

    return {
        "path_length": path_arr,
        "integrated_curvature": curv_arr,
        "labels": label_arr,
        "local_angle_profile": profile_arr,
        "snapshots": snaps,
        "save_times": np.array(sorted(args.save_times), dtype=np.float32),
        "perf": perf_bank,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(args)
    num_classes = cfg["cond"]["num_classes"]
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]), strict=False)

    ensure_dir(args.outdir)

    meta = {
        "ckpt": args.ckpt,
        "cfg": args.cfg,
        "methods": args.methods,
        "nfe": args.nfe,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "classes": args.classes,
        "guidance_scale": args.guidance_scale,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_nfe": args.max_nfe,
        "save_times": args.save_times,
        "seed": args.seed,
    }
    with open(Path(args.outdir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    z_all, y_all = build_fixed_inputs(cfg, args, device)

    for method in args.methods:
        print(f"\n=== Running {method} ===")
        res = run_one_method(model, method, args, cfg, device, z_all, y_all)
        out_path = Path(args.outdir) / f"{method}.npz"
        np.savez_compressed(
            out_path,
            path_length=res["path_length"],
            integrated_curvature=res["integrated_curvature"],
            labels=res["labels"],
            local_angle_profile=res["local_angle_profile"],
            snapshots=res["snapshots"],
            save_times=res["save_times"],
        )
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()