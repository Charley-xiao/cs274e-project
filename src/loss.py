from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Callable, Dict

Tensor = torch.Tensor
VField = Callable[[Tensor, Tensor, torch.Tensor | None], Tensor]

def rf_loss(v_theta: VField, x: Tensor, z: Tensor, t: Tensor, y: torch.Tensor | None) -> Tensor:
    """
    Conditional RF loss: regress to straight-line target (x - z), conditioned on label y.
    x,z: (B,C,H,W); t: (B,1,1,1) in [0,1]; y: (B,) int64 or None; use -1 for unconditional token.
    """
    x_t = (1 - t) * z + t * x
    target = x - z
    pred = v_theta(x_t, t, y)
    return F.mse_loss(pred, target)

def divergence_hutchinson(v_theta: VField, x: Tensor, t: Tensor, y: torch.Tensor | None, n_probe: int = 1) -> Tensor:
    """
    Hutchinson estimator of divergence_x v_theta(x,t,y).
    Returns (B,)
    """
    B = x.shape[0]
    div = x.new_zeros(B)
    x = x.detach().requires_grad_(True)
    for _ in range(n_probe):
        eps = torch.randn_like(x)
        v = v_theta(x, t, y)
        dot = (v * eps).sum()
        (grad_x,) = torch.autograd.grad(dot, x, create_graph=True)
        div = div + (grad_x * eps).flatten(1).sum(dim=1)
    return div / float(n_probe)

def rf_div_loss(
    v_theta: VField,
    x: Tensor,
    z: Tensor,
    t: Tensor,
    y: torch.Tensor | None,
    lambda_div: float = 1e-3,
    hutch_probes: int = 1,
) -> Dict[str, Tensor]:
    """
    RF-Div objective = RF MSE + lambda_div * (div_x v_theta)^2.
    For straight-line target (x - z), target divergence wrt x is 0.
    """
    x_t = (1 - t) * z + t * x
    target = x - z
    pred = v_theta(x_t, t, y)
    loss_rf = F.mse_loss(pred, target)

    div_pred = divergence_hutchinson(lambda X, T, Y: v_theta(X, T, Y), x_t, t, y, n_probe=hutch_probes)  # (B,)
    loss_div = (div_pred ** 2).mean()

    loss = loss_rf + float(lambda_div) * loss_div
    return {"loss": loss, "loss_rf": loss_rf, "loss_div": loss_div}
