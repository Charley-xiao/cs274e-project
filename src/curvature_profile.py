import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.model import create_model
from src.ode import integrate  # your integrate with record_states=True


@torch.no_grad()
def compute_curvature_profile(states: torch.Tensor):
    """
    states: (T+1, B, C, H, W) tensor of trajectory.
    Returns:
        kappa_mean: (T-1,) tensor -> mean curvature at interior steps
        kappa_std:  (T-1,) tensor -> std across batch
    """
    # Flatten spatial dims
    T_plus_1, B, C, H, W = states.shape
    T = T_plus_1 - 1  # number of segments

    x = states.view(T_plus_1, B, -1)  # (T+1, B, D)

    v = x[1:] - x[:-1]                # (T, B, D)
    seg_len = v.norm(dim=-1)         # (T, B)

    # avoid zero-length segments
    eps = 1e-8
    u = v / (seg_len.unsqueeze(-1) + eps)  # (T, B, D)

    # turning angles between consecutive segments: T-1 entries
    dot = (u[1:] * u[:-1]).sum(dim=-1)  # (T-1, B)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)             # (T-1, B)

    # average segment length around each interior point
    seg_len_mid = 0.5 * (seg_len[1:] + seg_len[:-1])  # (T-1, B)

    kappa = theta / (seg_len_mid + eps)  # (T-1, B) local curvature estimate

    # mean/std over batch
    kappa_mean = kappa.mean(dim=1)       # (T-1,)
    kappa_std  = kappa.std(dim=1)        # (T-1,)

    return kappa_mean.cpu(), kappa_std.cpu()


@torch.no_grad()
def main():
    ckpt_path = "600rf/runs/rf_cond/last.ckpt"  # adjust as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model + config ---
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    num_classes = cfg["cond"]["num_classes"]

    model = create_model(cfg.get("model", {}), num_classes=num_classes).to(device)
    state = ckpt.get("model_ema", ckpt["model"])
    model.load_state_dict(state, strict=False)
    model.eval()

    C = cfg.get("model", {}).get("in_channels", 3)
    H = W = cfg["data"]["image_size"]

    # --- sample initial noise and labels ---
    batch_size = 64
    z0 = torch.randn(batch_size, C, H, W, device=device)

    # e.g., random labels or fixed class
    y = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device)

    # --- choose high-accuracy sampler for "ground truth" trajectory ---
    def v_fn(x, t, y_cond):
        return model(x, t, y_cond)

    x1, states = integrate(
        v_fn,
        z0,
        nfe=512,              # large NFE for fine resolution
        solver="heun",        # or "rk23-adaptive" with adaptive=True
        y=y,
        guidance_scale=0.0,
        record_states=True,
    )

    kappa_mean, kappa_std = compute_curvature_profile(states)

    # --- plot curvature vs (normalized) time ---
    T_minus_1 = kappa_mean.shape[0]
    t = torch.linspace(0.0, 1.0, T_minus_1)  # interior points

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_np = t.numpy()
    kappa_mean_np = kappa_mean.numpy()
    kappa_std_np = kappa_std.numpy()

    plt.figure(figsize=(9, 3.5))

    # main curve + band
    plt.plot(t_np, kappa_mean_np, label="mean curvature")
    plt.fill_between(
        t_np,
        kappa_mean_np - kappa_std_np,
        kappa_mean_np + kappa_std_np,
        alpha=0.2,
        label="±1 std",
    )

    # --- 64 vertical dotted lines for a 64-step Euler solver ---
    n_euler = 64
    # positions of Euler steps in normalized time
    # you can choose endpoints; here we use centers of 64 equal bins
    t_euler = np.linspace(1.0/64, 1.0, 63, endpoint=False) + 0.5 / n_euler

    # interpolate kappa_mean at those positions
    kappa_euler = np.interp(t_euler, t_np, kappa_mean_np)

    for x_step, k_step in zip(t_euler, kappa_euler):
        # vertical dotted line from y=0 to y=kappa_mean at that time
        plt.plot(
            [x_step, x_step],
            [0.0, float(k_step)],
            linestyle=":",
            linewidth=0.7,
            color="0.5",
            alpha=0.7,
        )

    plt.xlabel("normalized time $t$")
    plt.ylabel("estimated curvature $\\kappa(t)$")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.ylim((0, 0.2))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "curvature_profile_euler_512.pdf", dpi=200)
    plt.close()

    print("Saved curvature profile to", out_dir / "curvature_profile_euler_512.pdf")


if __name__ == "__main__":
    main()
