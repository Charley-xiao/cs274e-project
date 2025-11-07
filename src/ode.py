from __future__ import annotations
import torch
from typing import Callable, Optional

Tensor = torch.Tensor
VField = Callable[[Tensor, Tensor, Optional[torch.Tensor]], Tensor]  # v_theta(x, t, y)

def _guided_v(
    v_theta: VField,
    x: Tensor,
    t: Tensor,
    y: Optional[torch.Tensor],
    guidance_scale: float = 0.0,
) -> Tensor:
    """
    Classifier-free guidance for velocity fields:
        v_guided = (1+s) * v(x,t,y) - s * v(x,t,uncond)
    y should be LongTensor (B,) with -1 for unconditional token. For the uncond eval, we force -1.
    """
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
    report_traj_len: bool = False,
) -> Tensor |tuple[Tensor, float]:
    """
    Integrate dx/dt = v_theta(x,t,y) from t=0->1. Supports CFG guidance. Reports average trajectory length if requested.
    Args:
        z0: (B,C,H,W)
        y:  (B,) class indices; use -1 for unconditional; None --> unconditional for all.
    """
    x = z0.clone()
    B = x.shape[0]
    steps = max(1, nfe)
    h = 1.0 / steps
    traj_len = 0.0

    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unknown solver '{solver}'")

    for k in range(steps):
        t_k = torch.full((B, 1, 1, 1), k * h, device=x.device, dtype=x.dtype)
        if solver == "euler":
            f_k = _guided_v(v_theta, x, t_k, y, guidance_scale)
            x = x + h * f_k
            traj_len += h * f_k.flatten(1).norm(dim=1).mean().item() # avg over batch
        else:  # Heun
            f_k = _guided_v(v_theta, x, t_k, y, guidance_scale)
            t_k1 = torch.full((B, 1, 1, 1), (k + 1) * h, device=x.device, dtype=x.dtype)
            x_pred = x + h * f_k
            f_k1 = _guided_v(v_theta, x_pred, t_k1, y, guidance_scale)
            x = x + 0.5 * h * (f_k + f_k1)
            traj_len += 0.5 * h * (f_k.flatten(1).norm(dim=1) + f_k1.flatten(1).norm(dim=1)).mean().item()

    if report_traj_len:
        return x, traj_len
    return x
