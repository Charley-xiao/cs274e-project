from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from enum import Enum
from typing import Callable, Dict
from torch.autograd.functional import jvp as autograd_jvp

Tensor = torch.Tensor
VField = Callable[[Tensor, Tensor, torch.Tensor | None], Tensor]

class DivEstimator(str, Enum):
    FD   = "fd"    # finite-difference
    GRAD = "grad"  # reverse-mode grad(dot(v, eps), x)
    JVP  = "jvp"   # autograd.functional.jvp

def _rademacher_like(x: Tensor) -> Tensor:
    return torch.randint_like(x, low=0, high=2).mul_(2).sub_(1).to(x.dtype)

def rf_loss(v_theta: VField, x: Tensor, z: Tensor, t: Tensor, y: torch.Tensor | None) -> Tensor:
    x_t = (1 - t) * z + t * x
    target = x - z
    pred = v_theta(x_t, t, y)
    return F.mse_loss(pred, target)

def divergence_hutchinson(
    v_theta: VField,
    x: Tensor,
    t: Tensor,
    y: torch.Tensor | None,
    n_probe: int = 1,
    estimator: DivEstimator = DivEstimator.FD,
    delta_fd: float = 1e-3,
) -> Tensor:
    """
    Hutchinson estimator of div_x v(x,t,y). Returns (B,).

    estimators:
      - FD  : εᵀ (J ε) ≈ [(v(x+δ ε) - v(x-δ ε)) · ε] / (2δ). 2 fwd / probe, first-order accurate.
      - GRAD: εᵀ (∂/∂x (v·ε)) via reverse-mode; requires create_graph=True (slow/heavy).
      - JVP : εᵀ (J ε) via autograd.functional.jvp (still 2nd-order graph).

    Notes for FD:
      * x is treated as a constant input (no requires_grad); gradients flow to θ through v(·; θ).
      * We scale δ by 1/√N (N = C*H*W) so the per-pixel perturbation stays small.
    """
    B = x.shape[0]
    div = x.new_zeros(B)

    if estimator == DivEstimator.FD:
        N = x[0].numel()
        step = (delta_fd / math.sqrt(N))
        for _ in range(n_probe):
            eps = _rademacher_like(x)
            x_plus  = x + step * eps
            x_minus = x - step * eps
            v_plus  = v_theta(x_plus,  t, y)
            v_minus = v_theta(x_minus, t, y)
            j_eps   = (v_plus - v_minus) / (2.0 * step)          # ≈ (J ε)
            div     = div + (j_eps * eps).flatten(1).sum(dim=1)  # εᵀ (J ε)
        return div / float(n_probe)

    if estimator == DivEstimator.GRAD:
        x_req = x.detach().requires_grad_(True)
        for _ in range(n_probe):
            eps = _rademacher_like(x_req)
            v = v_theta(x_req, t, y)
            dot = (v * eps).sum()
            (grad_x,) = torch.autograd.grad(dot, x_req, create_graph=True)
            div = div + (grad_x * eps).flatten(1).sum(dim=1)
        return div / float(n_probe)

    # JVP branch
    x_req = x.detach().requires_grad_(True)
    def _v_only(x_): return v_theta(x_, t, y)
    for _ in range(n_probe):
        eps = _rademacher_like(x_req)
        _, j = autograd_jvp(_v_only, (x_req,), (eps,), create_graph=True, strict=False)
        div = div + (j * eps).flatten(1).sum(dim=1)
    return div / float(n_probe)

def rf_div_loss(
    v_theta: VField,
    x: Tensor,
    z: Tensor,
    t: Tensor,
    y: torch.Tensor | None,
    lambda_div: float = 1e-3,
    hutch_probes: int = 1,
    estimator: str = "fd",
    div_batch_frac: float = 1.0,
    delta_fd: float = 1e-3,
) -> Dict[str, Tensor]:
    """
    RF-Div objective = RF MSE + λ * (div_x v)^2.
    """
    x_t = (1 - t) * z + t * x
    target = x - z
    pred = v_theta(x_t, t, y)
    loss_rf = F.mse_loss(pred, target)

    if lambda_div <= 0.0:
        return {"loss": loss_rf, "loss_rf": loss_rf, "loss_div": x.new_tensor(0.0)}

    B = x_t.size(0)
    m = max(1, int(B * float(div_batch_frac)))
    xb, tb = x_t[:m], t[:m]
    yb = None if y is None else y[:m]

    div_est = divergence_hutchinson(
        v_theta, xb, tb, yb,
        n_probe=hutch_probes,
        estimator=DivEstimator(estimator),
        delta_fd=delta_fd,
    )
    loss_div = (div_est ** 2).mean()
    loss = loss_rf + float(lambda_div) * loss_div
    return {"loss": loss, "loss_rf": loss_rf, "loss_div": loss_div}
