# src/ode.py
# Deterministic ODE integration with optional classifier-free guidance,
# plus *path length* and *integrated curvature* metrics.

from __future__ import annotations
import torch
from typing import Callable, Optional, Tuple, Dict

Tensor = torch.Tensor
VField = Callable[[Tensor, Tensor, Optional[torch.Tensor]], Tensor]  # v_theta(x, t, y)

def _guided_v(
    v_theta: VField,
    x: Tensor,
    t: Tensor,
    y: Optional[torch.Tensor],
    guidance_scale: float = 0.0,
) -> Tensor:
    if guidance_scale <= 0.0 or y is None:
        return v_theta(x, t, y)
    y_uncond = torch.full_like(y, -1)
    v_cond = v_theta(x, t, y)
    v_uncond = v_theta(x, t, y_uncond)
    return (1.0 + guidance_scale) * v_cond - guidance_scale * v_uncond

@torch.no_grad()
def integrate(
    v_theta: VField,
    z0: Tensor,
    nfe: int = 4,
    solver: str = "heun",
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
    return_metrics: bool = False,
    traj_for_viz: int = 0,  # store trajectories for first N samples (0 = off)
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """
    Integrate dx/dt = v_theta(x,t,y) from t=0->1. Supports CFG guidance.
    Also computes:
      - path length: sum_k ||x_{k+1}-x_k||_2
      - integrated curvature: sum_k arccos( <Δx_k, Δx_{k+1}> / (||Δx_k||*||Δx_{k+1}||) )
        (discrete turning-angle measure in state-space)

    Args:
        z0: (B,C,H,W)
        y:  (B,) class indices; -1 = unconditional; None → unconditional batch
        traj_for_viz: if >0, stores flattened states for first N samples (S+1, N, D)

    Returns:
        x1  or  (x1, metrics_dict)
        metrics_dict keys:
          - "path_length": (B,) tensor
          - "integrated_curvature": (B,) tensor
          - optionally "traj": (steps+1, N, D) float32 CPU tensor for viz
    """
    x = z0.clone()
    B = x.shape[0]
    steps = max(1, nfe)
    h = 1.0 / steps

    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unknown solver '{solver}'")

    eps = 1e-12
    path_len = x.new_zeros(B)
    integ_curv = x.new_zeros(B)
    prev_dx_flat = None

    # optional trajectory capture (keep small N)
    Nviz = int(min(max(traj_for_viz, 0), B))
    traj_list = []
    if Nviz > 0:
        traj_list.append(x[:Nviz].reshape(Nviz, -1).detach().cpu())

    for k in range(steps):
        t_k = torch.full((B, 1, 1, 1), k * h, device=x.device, dtype=x.dtype)
        if solver == "euler":
            f_k = _guided_v(v_theta, x, t_k, y, guidance_scale)
            x_new = x + h * f_k
        else:  # Heun
            f_k = _guided_v(v_theta, x, t_k, y, guidance_scale)
            t_k1 = torch.full((B, 1, 1, 1), (k + 1) * h, device=x.device, dtype=x.dtype)
            x_pred = x + h * f_k
            f_k1 = _guided_v(v_theta, x_pred, t_k1, y, guidance_scale)
            x_new = x + 0.5 * h * (f_k + f_k1)

        # --- metrics: path length & integrated curvature (turning angle) ---
        dx = x_new - x
        dx_flat = dx.view(B, -1)
        step_len = torch.linalg.norm(dx_flat, dim=1)  # (B,)
        path_len += step_len

        if prev_dx_flat is not None:
            dot = (prev_dx_flat * dx_flat).sum(dim=1)
            denom = torch.clamp(
                torch.linalg.norm(prev_dx_flat, dim=1) * step_len, min=eps
            )
            cos_theta = torch.clamp(dot / denom, -1.0 + 1e-6, 1.0 - 1e-6)
            theta = torch.arccos(cos_theta)  # radians
            integ_curv += theta

        prev_dx_flat = dx_flat

        x = x_new

        if Nviz > 0:
            traj_list.append(x[:Nviz].reshape(Nviz, -1).detach().cpu())

    if not return_metrics:
        return x

    metrics: Dict[str, Tensor] = {
        "path_length": path_len.detach().cpu(),
        "integrated_curvature": integ_curv.detach().cpu(),
    }
    if Nviz > 0:
        metrics["traj"] = torch.stack(traj_list, dim=0).float()  # (steps+1, Nviz, D) on CPU

    return x, metrics
