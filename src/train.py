from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, Tuple
import time
from tqdm import trange

import yaml
import torch
from torch import optim
from torchvision.utils import save_image

from .loss import rf_loss, rf_div_loss
from .ode import integrate
from .data import eurosat_dataloaders
from .model import create_model, create_model_using_diffusers
from .util import unnormalize_to01

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def sample_t(batch_size: int, device: torch.device, mode: str = "uniform") -> torch.Tensor:
    if mode == "beta_half":
        t = torch.distributions.Beta(0.5, 0.5).sample((batch_size, 1, 1, 1)).to(device)
    else:
        t = torch.rand(batch_size, 1, 1, 1, device=device)
    return t

def maybe_drop_labels(y: torch.Tensor, p_uncond: float) -> torch.Tensor:
    """
    Classifier-free dropout: with prob p_uncond, set label to -1 (unconditional token).
    y: (B,) int64 class indices
    """
    if p_uncond <= 0:
        return y
    mask = (torch.rand_like(y.float()) < p_uncond)
    y_dropped = y.clone()
    y_dropped[mask] = -1
    return y_dropped

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def train(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device: {device}")
    set_seed(cfg.get("seed", 42))

    # Data
    t0 = time.time()
    train_loader, val_loader = eurosat_dataloaders(
        root=cfg["data"]["root"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    num_classes = cfg["cond"]["num_classes"]

    # Model (+ num_classes passed in cfg['cond'])
    if cfg.get("model", {}).get("use_diffusers", False):
        print("[INFO] using Diffusers UNet model architecture")
        model = create_model_using_diffusers(cfg.get("model", {}), num_classes=cfg["cond"]["num_classes"]).to(device)
    else:
        model = create_model(cfg.get("model", {}), num_classes=cfg["cond"]["num_classes"]).to(device)
    print(model)
    param_stats = count_parameters(model)

    opt = optim.AdamW(
        model.parameters(),
        lr=float(cfg["opt"].get("lr", 2e-4)),
        betas=cfg["opt"].get("betas", (0.9, 0.999)),
        weight_decay=float(cfg["opt"].get("wd", 2e-2)),
    )

    # Output dirs
    name = cfg.get("name", "rf_cond")
    out_dir = Path("runs") / name
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    meta = {
        "name": name,
        "data_root": cfg["data"]["root"],
        "train_size": train_size,
        "val_size": val_size,
        "num_classes": num_classes,
        "image_size": cfg["data"]["image_size"],
        "batch_size": cfg["data"]["batch_size"],
        "model_params": param_stats,
        "cfg": cfg,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # also log to console
    print(f"[INFO] run: {name}")
    print(f"[INFO] train size: {train_size} | val size: {val_size} | classes: {num_classes}")
    print(f"[INFO] model params: {param_stats['trainable']:,} trainable / {param_stats['total']:,} total")
    print(f"[INFO] data/model setup took {time.time() - t0:.2f}s")

    # Hyperparams
    epochs = cfg["train"].get("epochs", 120)
    t_mode = cfg["train"].get("t_sampling", "uniform")
    log_every = cfg["train"].get("log_every", 100)
    grad_clip = cfg["train"].get("grad_clip", 1.0)
    p_uncond = cfg["cond"].get("p_uncond", 0.1)
    lambda_div = cfg.get("loss", {}).get("lambda_div", 0.0)
    hutch_probes = cfg.get("loss", {}).get("hutch_probes", 1)
    estimator = cfg.get("loss", {}).get("estimator", "fd")
    delta_fd = cfg.get("loss", {}).get("delta_fd", 1e-3)
    div_batch_frac = cfg.get("loss", {}).get("div_batch_frac", 1.0)
    div_every = cfg.get("loss", {}).get("div_every", 1)

    step = 0
    for ep in trange(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)          # (B,C,H,W)
            y = y.to(device, non_blocking=True).long()   # (B,)
            z = torch.randn_like(x)
            t = sample_t(x.size(0), device, mode=t_mode)

            y_cf = maybe_drop_labels(y, p_uncond)

            opt.zero_grad(set_to_none=True)
            use_div = (lambda_div > 0.0) and ( (step % div_every) == 0 )
            if use_div:
                # losses = rf_div_loss(model, x, z, t, y_cf, lambda_div=lambda_div, hutch_probes=hutch_probes)
                # loss = losses["loss"]
                lam_eff = lambda_div * div_every
                losses = rf_div_loss(
                    model, x, z, t, y_cf,
                    lambda_div=lam_eff,
                    hutch_probes=hutch_probes,
                    estimator=estimator,
                    div_batch_frac=div_batch_frac,
                    delta_fd=delta_fd,
                )
                loss = losses["loss"]
            else:
                loss = rf_loss(model, x, z, t, y_cf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            step += 1

            if step % log_every == 0:
                lr = next(iter(opt.param_groups))["lr"]
                if lambda_div > 0:
                    print(f"[ep {ep:03d} | step {step:06d}] loss={loss.item():.4f} "
                          f"rf={losses['loss_rf'].item():.4f} div={losses['loss_div'].item():.6f} lr={lr:.2e}")
                else:
                    print(f"[ep {ep:03d} | step {step:06d}] loss={loss.item():.4f} lr={lr:.2e}")

        if ep % cfg["train"].get("save_every", 10) == 0:
            model.eval()
            with torch.no_grad():
                C = cfg.get("model", {}).get("in_channels", 3)
                H = W = cfg["data"]["image_size"]
                n_show = min(cfg["cond"]["num_classes"], 10)
                per_class = 4
                imgs = []
                for cls in range(n_show):
                    z0 = torch.randn(per_class, C, H, W, device=device)
                    y_grid = torch.full((per_class,), cls, device=device, dtype=torch.long)
                    x1 = integrate(lambda X, T, Y: model(X, T, Y),
                                z0, nfe=4, solver=cfg.get("sample", {}).get("solver", "heun"),
                                y=y_grid, guidance_scale=cfg.get("sample", {}).get("guidance_scale", 0.0))
                    imgs.append(x1)
                X = torch.cat(imgs, dim=0)
                save_image(unnormalize_to01(X), out_dir / f"samples/ep{ep:03d}_grid.png", nrow=per_class)

            ckpt = {
                "model": model.state_dict(),
                "config": cfg,
                "step": step,
                "epoch": ep,
            }
            torch.save(ckpt, out_dir / "last.ckpt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
