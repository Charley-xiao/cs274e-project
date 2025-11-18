from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics

from .model import create_model
from .samplers import integrate
from .util import unnormalize_to01


def _compute_fid_is(real_dir: str, fake_dir: str, use_cuda: bool, batch_size: int = 64) -> Dict[str, Any]:
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=use_cuda,
        batch_size=batch_size,
        isc=True, fid=True, verbose=False, samples_find_deep=True
    )
    # torch-fidelity returns a dict with keys like 'frechet_inception_distance', 'inception_score_mean', 'inception_score_std'
    return {
        "fid": float(metrics["frechet_inception_distance"]),
        "isc_mean": float(metrics["inception_score_mean"]),
        "isc_std": float(metrics["inception_score_std"]),
    }

# ------------ main ------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="samples_cmp")
    ap.add_argument("--methods", nargs="+", default=["euler","heun","midpoint","rk4","ab2","rk23-adaptive"])
    ap.add_argument("--nfe", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--max_nfe", type=int, default=2000)
    ap.add_argument("--metrics_out", type=str, default="sampler_metrics.json")

    # NEW: FID/IS args
    ap.add_argument("--real_dir", type=str, default=None, help="Folder of real validation images (PNG/JPG). If set, FID/IS will be computed.")
    ap.add_argument("--fid_batch", type=int, default=64)
    ap.add_argument("--no_cuda_fid", action="store_true")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    C = cfg.get("model", {}).get("in_channels", 3)
    H = W = cfg["data"]["image_size"]
    num_classes = cfg["cond"]["num_classes"]
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    state = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict(state, strict=False)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    def make_labels(n: int):
        if not args.classes:
            return torch.full((n,), -1, device=device, dtype=torch.long)
        cls = torch.tensor(args.classes, device=device, dtype=torch.long)
        return cls.repeat((n + len(cls) - 1) // len(cls))[:n]

    results = []

    for method in args.methods:
        nfe_list = args.nfe if (method != "rk23-adaptive" and not args.adaptive) else [0]  # NFE not fixed for adaptive
        for k in nfe_list:
            z0 = torch.randn(args.num, C, H, W, device=device)
            y  = make_labels(args.num)

            X_chunks, pl_all, ic_all = [], [], []
            t0 = time.time()
            bs = min(args.num, 64)
            total_fe = 0
            for i in range(0, args.num, bs):
                z = z0[i:i+bs]; yb = y[i:i+bs]
                x1, metrics, perf = integrate(
                    lambda X,T,Y: model(X,T,Y),
                    z,
                    method=method,
                    nfe=k,
                    y=yb,
                    guidance_scale=args.guidance_scale,
                    adaptive=args.adaptive or (method == "rk23-adaptive"),
                    rtol=args.rtol, atol=args.atol, max_nfe=args.max_nfe,
                    return_metrics=True,
                )
                X_chunks.append(x1.cpu())
                pl_all.append(metrics["path_length"])
                ic_all.append(metrics["integrated_curvature"])
                total_fe += perf["nfe"]
            wall = time.time() - t0

            X = torch.cat(X_chunks, dim=0)
            PL = torch.cat(pl_all, dim=0); IC = torch.cat(ic_all, dim=0)

            tag = f"{method}_nfe{k}" if k else f"{method}_adaptive"
            grid_path = out_dir / f"grid_{tag}.png"
            save_image(unnormalize_to01(X), grid_path, nrow=int(args.num**0.5) or 8)

            # also save per-image PNGs for FID/IS
            fake_dir = out_dir / tag
            fake_dir.mkdir(exist_ok=True)
            print(f"Saving individual images to {fake_dir}")
            for i, x in enumerate(X):
                save_image(unnormalize_to01(x.unsqueeze(0)), fake_dir / f"{i:05d}.png", nrow=1)

            # aggregate stats
            stats = {
                "method": method,
                "nfe_arg": int(k),
                "effective_nfe": int(total_fe),
                "wall_time_sec": wall,
                "num_samples": int(args.num),
                "guidance_scale": float(args.guidance_scale),
                "path_length_mean": float(PL.mean()), "path_length_std": float(PL.std(unbiased=False)),
                "integrated_curvature_mean": float(IC.mean()), "integrated_curvature_std": float(IC.std(unbiased=False)),
            }

            # FID/IS if real_dir provided
            if args.real_dir:
                use_cuda_fid = (torch.cuda.is_available() and not args.no_cuda_fid)
                fidis = _compute_fid_is(args.real_dir, str(fake_dir), use_cuda_fid, batch_size=args.fid_batch)
                stats.update(fidis)
                print(f"[{tag}] wall={wall:.2f}s FE={total_fe}  FID={fidis['fid']:.3f}  "
                      f"IS={fidis['isc_mean']:.3f}±{fidis['isc_std']:.3f}  "
                      f"PLμ={stats['path_length_mean']:.4f} ICμ={stats['integrated_curvature_mean']:.4f}")
            else:
                print(f"[{tag}] wall={wall:.2f}s FE={total_fe}  "
                      f"PLμ={stats['path_length_mean']:.4f} ICμ={stats['integrated_curvature_mean']:.4f}")

            results.append(stats)

    with open(out_dir / args.metrics_out, "w") as f:
        json.dump({"runs": results}, f, indent=2)

if __name__ == "__main__":
    main()
