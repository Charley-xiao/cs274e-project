from __future__ import annotations
from typing import Optional, Dict, Tuple, Callable
import math
import time
import torch

Tensor = torch.Tensor
VField = Callable[[Tensor, Tensor, Optional[torch.Tensor]], Tensor]  # v_theta(x,t,y)

# ---- guidance wrapper (same convention as your ode.py) -----------------------
def _guided_v(
    v_theta: VField, x: Tensor, t: Tensor, y: Optional[torch.Tensor], guidance_scale: float
) -> Tensor:
    if guidance_scale <= 0.0 or (y is None):
        return v_theta(x, t, y)
    y_uncond = torch.full_like(y, -1)
    vc = v_theta(x, t, y)
    vu = v_theta(x, t, y_uncond)
    return (1.0 + guidance_scale) * vc - guidance_scale * vu

# ---- trajectory metrics (path length & integrated curvature) ----------------
class TrajMeter:
    def __init__(self, B: int, device, dtype):
        self.path = torch.zeros(B, device=device, dtype=dtype)
        self.curv = torch.zeros(B, device=device, dtype=dtype)
        self.prev_u = None
        self.prev_valid = None
        self.eps = 1e-12

    def update(self, x_old: Tensor, x_new: Tensor):
        dx = x_new - x_old
        step_len = dx.flatten(1).norm(dim=1)
        self.path += step_len
        denom = (step_len + self.eps).view(-1, 1, 1, 1)
        u = dx / denom
        curr_valid = (step_len > self.eps)
        if self.prev_u is not None:
            dot = (self.prev_u * u).flatten(1).sum(dim=1).clamp(-1.0, 1.0)
            angle = torch.acos(dot)
            mask = self.prev_valid & curr_valid
            if mask.any():
                self.curv[mask] += angle[mask]
        self.prev_u = u
        self.prev_valid = curr_valid

# ---- fixed-step steppers -----------------------------------------------------
def _step_euler(f, x, t, h):                # 1 FE
    k1 = f(x, t)
    return x + h * k1, 1

def _step_midpoint(f, x, t, h):             # 2 FE
    k1 = f(x, t)
    k2 = f(x + 0.5*h*k1, t + 0.5*h)
    return x + h * k2, 2

def _step_heun(f, x, t, h):                 # 2 FE (explicit trapezoid)
    k1 = f(x, t)
    k2 = f(x + h*k1, t + h)
    return x + 0.5*h*(k1 + k2), 2

def _step_rk4(f, x, t, h):                  # 4 FE
    k1 = f(x, t)
    k2 = f(x + 0.5*h*k1, t + 0.5*h)
    k3 = f(x + 0.5*h*k2, t + 0.5*h)
    k4 = f(x + h*k3, t + h)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4), 4

def _init_snapshot_buffer(x0: Tensor, save_times):
    if not save_times:
        return None, None
    ts = sorted(float(t) for t in save_times)
    snaps = {t: None for t in ts}
    return ts, snaps


def _maybe_store_snapshot(
    x: Tensor,
    t_prev: float,
    t_curr: float,
    save_times,
    snapshots,
):
    """
    Store x at target save_times when the integration interval [t_prev, t_curr]
    reaches or passes them. For now we store the current state x as an approximation.
    """
    if save_times is None or snapshots is None:
        return

    for tau in save_times:
        if snapshots[tau] is None and (t_prev <= tau <= t_curr):
            snapshots[tau] = x.detach().cpu().clone()


def _finalize_snapshots(x_final: Tensor, save_times, snapshots):
    if save_times is None or snapshots is None:
        return None
    for tau in save_times:
        if snapshots[tau] is None:
            snapshots[tau] = x_final.detach().cpu().clone()
    return snapshots

# Two-step Adams–Bashforth (AB2). Bootstrap first step with Heun.
def _integrate_ab2(
    f, x0, nfe, y, guidance_scale,
    return_metrics=False,
    save_times=None,
    return_snapshots=False,
):
    B = x0.shape[0]
    device, dtype = x0.device, x0.dtype
    steps = max(1, nfe)
    h = 1.0 / steps
    t = torch.full((B,1,1,1), 0.0, device=device, dtype=dtype)
    meter = TrajMeter(B, device, dtype) if return_metrics else None
    fe = 0

    save_times_sorted, snapshots = _init_snapshot_buffer(x0, save_times)
    if snapshots is not None:
        snapshots[0.0] = x0.detach().cpu().clone()

    def fv(x, tt):
        return _guided_v(f, x, tt, y, guidance_scale)

    x = x0

    # first step: Heun
    x1, used = _step_heun(fv, x, t, h)
    fe += used
    if meter:
        meter.update(x, x1)

    if snapshots is not None:
        _maybe_store_snapshot(x1, 0.0, h, save_times_sorted, snapshots)

    k_prev = fv(x1, t + h)
    fe += 1

    x = x1
    for s in range(1, steps):
        t_prev = s * h
        t_curr = (s + 1) * h
        t_s = torch.full_like(t, s*h)

        f_n = fv(x, t_s)
        fe += 1
        x_new = x + h * (1.5 * f_n - 0.5 * k_prev)

        if meter:
            meter.update(x, x_new)

        if snapshots is not None:
            _maybe_store_snapshot(x_new, t_prev, t_curr, save_times_sorted, snapshots)

        x, k_prev = x_new, f_n

    metrics = None
    if return_metrics:
        metrics = {
            "path_length": meter.path.cpu(),
            "integrated_curvature": meter.curv.cpu(),
        }

    snapshots = _finalize_snapshots(x, save_times_sorted, snapshots) if return_snapshots else None

    if not return_metrics and not return_snapshots:
        return x, fe, None

    return x, fe, metrics, snapshots

# ---- adaptive RK23 (Bogacki–Shampine) ---------------------------------------
def _integrate_rk23_adaptive(
    f, x0, y, guidance_scale,
    rtol=1e-3, atol=1e-4,
    h_init=1/8, h_min=1/512, h_max=1/4, max_nfe=2000,
    return_metrics=False,
    save_times=None,
    return_snapshots=False,
):
    B = x0.shape[0]
    device, dtype = x0.device, x0.dtype
    meter = TrajMeter(B, device, dtype) if return_metrics else None

    save_times_sorted, snapshots = _init_snapshot_buffer(x0, save_times)
    if snapshots is not None:
        snapshots[0.0] = x0.detach().cpu().clone()

    def fv(x, tt):
        return _guided_v(f, x, tt, y, guidance_scale)

    t0 = 0.0
    x = x0
    h = float(h_init)
    nfe = 0
    wall = 0.0
    hist_steps = 0

    safety, facmin, facmax = 0.9, 0.2, 5.0

    while t0 < 1.0:
        h = min(h, 1.0 - t0)
        t = torch.full((B,1,1,1), t0, device=device, dtype=dtype)

        tic = time.time()
        k1 = fv(x, t);                                nfe += 1
        k2 = fv(x + 0.5*h*k1, t + 0.5*h);            nfe += 1
        k3 = fv(x + 0.75*h*k2, t + 0.75*h);          nfe += 1

        x3 = x + h * (2/9*k1 + 1/3*k2 + 4/9*k3)

        k4 = fv(x3, t + h);                          nfe += 1
        x2 = x + h * (7/24*k1 + 0.25*k2 + 1/3*k3 + 1/8*k4)

        err = (x3 - x2).flatten(1)
        sc = atol + rtol * torch.max(x.abs().flatten(1), x3.abs().flatten(1))
        e = (err / sc).norm(dim=1) / math.sqrt(err.shape[1])
        e_max = float(e.max().item())
        wall += (time.time() - tic)

        if e_max <= 1.0 or h <= h_min * 1.01:
            t_prev = t0
            t_curr = t0 + h

            if meter:
                meter.update(x, x3)

            x = x3
            t0 += h
            hist_steps += 1

            if snapshots is not None:
                _maybe_store_snapshot(x, t_prev, t_curr, save_times_sorted, snapshots)

        if e_max == 0.0:
            fac = facmax
        else:
            fac = min(facmax, max(facmin, safety * (1.0 / e_max) ** 0.25))
        h = max(h_min, min(h_max, h * fac))

        if nfe > max_nfe:
            break

    metrics = None
    if return_metrics:
        metrics = {
            "path_length": meter.path.cpu(),
            "integrated_curvature": meter.curv.cpu(),
        }

    history = {
        "accepted_steps": hist_steps,
        "total_fe": nfe,
        "wall_time_sec": wall,
    }

    snapshots = _finalize_snapshots(x, save_times_sorted, snapshots) if return_snapshots else None

    return x, nfe, metrics, history, snapshots

# ---- public integrate() ------------------------------------------------------
def integrate(
    v_theta: VField,
    z0: Tensor,
    method: str = "heun",
    nfe: int = 4,
    y: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
    adaptive: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    max_nfe: int = 2000,
    return_metrics: bool = False,
    save_times = None,
    return_snapshots: bool = False,
):
    """
    Unified integration API with multiple samplers.
    method ∈ {"euler","midpoint","heun","rk4","ab2","rk23-adaptive"}
    If adaptive=True or method == "rk23-adaptive", the RK23 controller is used.
    Returns:
        x1 (Tensor) or (x1, metrics_dict, perf_dict) if return_metrics=True
    """
    B = z0.shape[0]
    device, dtype = z0.device, z0.dtype

    if adaptive or method == "rk23-adaptive":
        x1, fe, metrics, hist, snapshots = _integrate_rk23_adaptive(
            v_theta,
            z0,
            y,
            guidance_scale,
            rtol,
            atol,
            max_nfe=max_nfe,
            return_metrics=return_metrics,
            save_times=save_times,
            return_snapshots=return_snapshots,
        )
        perf = {"nfe": fe, **hist}

        if return_metrics and return_snapshots:
            return x1, metrics, perf, snapshots
        if return_metrics:
            return x1, metrics, perf
        if return_snapshots:
            return x1, snapshots
        return x1

    # fixed-step methods
    STEPS = {
        "euler": _step_euler,
        "midpoint": _step_midpoint,
        "heun": _step_heun,
        "rk4": _step_rk4,
    }
    if method not in (*STEPS.keys(), "ab2"):
        raise ValueError(f"Unknown method '{method}'")

    def fv(x, tt):
        return _guided_v(v_theta, x, tt, y, guidance_scale)

    if method == "ab2":
        x1, fe, metrics, snapshots = _integrate_ab2(
            v_theta,
            z0,
            nfe,
            y,
            guidance_scale,
            return_metrics=return_metrics,
            save_times=save_times,
            return_snapshots=return_snapshots,
        )
        perf = {"nfe": fe}

        if return_metrics and return_snapshots:
            return x1, metrics, perf, snapshots
        if return_metrics:
            return x1, metrics, perf
        if return_snapshots:
            return x1, snapshots
        return x1

    steps = max(1, nfe)
    h = 1.0 / steps
    t0 = torch.full((B,1,1,1), 0.0, device=device, dtype=dtype)
    x = z0.clone()
    fe = 0
    meter = TrajMeter(B, device, dtype) if return_metrics else None

    save_times_sorted, snapshots = _init_snapshot_buffer(z0, save_times)
    if snapshots is not None:
        snapshots[0.0] = z0.detach().cpu().clone()

    step_fn = STEPS[method]

    for s in range(steps):
        t_prev = s * h
        t_curr = (s + 1) * h
        t_s = t0 + s * h

        x_new, used = step_fn(lambda X, T: fv(X, T), x, t_s, h)
        fe += used

        if meter:
            meter.update(x, x_new)

        if snapshots is not None:
            _maybe_store_snapshot(x_new, t_prev, t_curr, save_times_sorted, snapshots)

        x = x_new

    perf = {"nfe": fe}
    metrics = None
    if return_metrics:
        metrics = {
            "path_length": meter.path.cpu(),
            "integrated_curvature": meter.curv.cpu(),
        }

    snapshots = _finalize_snapshots(x, save_times_sorted, snapshots) if return_snapshots else None

    if return_metrics and return_snapshots:
        return x, metrics, perf, snapshots
    if return_metrics:
        return x, metrics, perf
    if return_snapshots:
        return x, snapshots
    return x
