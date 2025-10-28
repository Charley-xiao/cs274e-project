from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

from .ode import integrate
from .model import create_model
from .util import unnormalize_to01

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="samples_out")
    ap.add_argument("--nfe", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--solver", type=str, default=None)
    ap.add_argument("--classes", type=int, nargs="*", default=None)
    ap.add_argument("--guidance_scale", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]
    solver = args.solver or cfg.get("sample", {}).get("solver", "heun")
    img_size = cfg["data"]["image_size"]
    C = cfg.get("model", {}).get("in_channels", 3)
    num_classes = cfg["cond"]["num_classes"]

    # model (EMA if present)
    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device).eval()
    model.load_state_dict(ckpt.get("model_ema", ckpt["model"]), strict=False)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    def make_labels(n: int) -> torch.Tensor | None:
        if args.classes is None or len(args.classes) == 0:
            return torch.full((n,), -1, device=device, dtype=torch.long)  # unconditional token
        # cycle through requested classes
        cls = torch.tensor(args.classes, device=device, dtype=torch.long)
        y = cls.repeat((n + len(cls) - 1) // len(cls))[:n]
        return y

    for nfe in args.nfe:
        z0 = torch.randn(args.num, C, img_size, img_size, device=device)
        y  = make_labels(args.num)

        X = []
        bs = min(args.num, 64)
        for i in range(0, args.num, bs):
            x_chunk = integrate(lambda X_, T_, Y_: model(X_, T_, Y_),
                                z0[i:i+bs], nfe=nfe, solver=solver,
                                y=None if y is None else y[i:i+bs],
                                guidance_scale=args.guidance_scale)
            X.append(x_chunk.cpu())
        X = torch.cat(X, dim=0)

        grid_path = out_dir / f"grid_nfe{nfe}.png"
        save_image(unnormalize_to01(X), grid_path, nrow=int(args.num**0.5) or 8)

        nfe_dir = out_dir / f"nfe{nfe}"
        nfe_dir.mkdir(exist_ok=True)
        for i, x in enumerate(X):
            save_image(unnormalize_to01(x), nfe_dir / f"{i:05d}.png")

if __name__ == "__main__":
    main()
