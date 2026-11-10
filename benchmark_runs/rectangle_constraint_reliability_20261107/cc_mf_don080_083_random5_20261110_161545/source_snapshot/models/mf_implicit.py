"""Eager, stage-batched implicit multi-feature clearance margins.

Geometry and feature selection belong to the caller.  This module evaluates
the frozen Gaussian feature model in float64, without autograd, variance
floors, feature pruning, or a fallback root solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Optional

import torch
from torch import Tensor


class MFImplicitRootStatus(IntEnum):
    NOT_REQUESTED = 0
    DOMAIN_INACTIVE = 1
    NO_OBSTACLES = 2
    SUCCESS = 3
    NO_VALID_FEATURES = 4
    INVALID_FEATURE = 5
    INVALID_VARIANCE = 6
    INVALID_BRACKET = 7
    INVALID_DERIVATIVE = 8
    NONFINITE_STEP = 9
    BRACKET_ESCAPE = 10
    STAGNATION = 11
    MAX_ITERATIONS = 12
    INVALID_COEFFICIENTS = 13
    INVALID_GEOMETRY = 14
    INVALID_EPSILON = 15
    INVALID_NOMINAL = 16
    LOCAL_BRACKET_FAILED = 17
    PROBABILITY_RESIDUAL = 18


ROOT_STATUS_NAMES = {int(status): status.name.lower() for status in MFImplicitRootStatus}


def root_status_name(value: int) -> str:
    return ROOT_STATUS_NAMES.get(int(value), f"unknown_status_{value}")


@dataclass
class MFImplicitResult:
    phi: Tensor
    gradient: Tensor
    A: Tensor
    c: Tensor
    margin: Tensor
    probability_sum: Tensor
    root_residual: Tensor
    root_bracket_width: Tensor
    root_error_bound: Tensor
    root_iterations: Tensor
    feature_count: Tensor
    root_valid: Tensor
    root_status: Tensor
    mf_required: Tensor
    root_candidate: Tensor
    root_value: Tensor
    root_derivative: Tensor
    root_lower: Tensor
    root_upper: Tensor
    frozen_features: Optional[dict] = None
    failure_features: Optional[dict] = None


def validate_root_settings(
    root_max_iter=12,
    root_atol_m=1e-8,
    root_rtol=1e-10,
    root_prob_atol=1e-10,
) -> None:
    if isinstance(root_max_iter, bool) or not isinstance(root_max_iter, int) or root_max_iter <= 0:
        raise ValueError("mf_imp_root_max_iter must be a positive integer")
    for name, value, strictly_positive in (
        ("mf_imp_root_atol_m", root_atol_m, True),
        ("mf_imp_root_rtol", root_rtol, False),
        ("mf_imp_root_prob_atol", root_prob_atol, True),
    ):
        if not math.isfinite(float(value)) or (value <= 0 if strictly_positive else value < 0):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {qualifier}")


def normalize_implicit_epsilon(epsilon, z_bar: Tensor) -> Tensor:
    # Validate ordinary Python/config values before transferring them.  Tensor
    # value checks on CUDA remain stagewise GPU status checks below.
    if isinstance(epsilon, (float, int)):
        if not math.isfinite(float(epsilon)) or not 0.0 < epsilon < 0.5:
            raise ValueError("implicit MF epsilon must satisfy 0 < epsilon < 0.5")
    result = torch.as_tensor(epsilon, dtype=torch.float64, device=z_bar.device)
    if result.ndim == 0:
        result = result.expand(z_bar.shape[0])
    if result.ndim != 1 or result.shape[0] != z_bar.shape[0]:
        raise ValueError("epsilon must be a scalar or have shape (H,)")
    if result.device.type == "cpu":
        if not torch.all(torch.isfinite(result) & (result > 0) & (result < 0.5)):
            raise ValueError("implicit MF epsilon must satisfy 0 < epsilon < 0.5")
    return result


def projected_variance_and_gradient(q: Tensor, z_bar: Tensor) -> tuple[Tensor, Tensor]:
    """Return the unfloored Log-Cholesky projected variance and its xi gradient."""
    q_f, q_l, q_theta = q.unbind(-1)
    forward = z_bar[:, 4, None].exp() * q_f
    lateral = z_bar[:, 5, None].exp() * q_l
    heading = z_bar[:, 6, None].exp() * q_theta
    coupled = lateral + z_bar[:, 7, None] * q_theta
    variance = forward.square() + coupled.square() + heading.square()
    gradient = torch.stack(
        (2.0 * forward.square(), 2.0 * lateral * coupled,
         2.0 * heading.square(), 2.0 * q_theta * coupled), dim=-1,
    )
    return variance, gradient


def _feature_world_gradient(q: Tensor, theta: Tensor) -> Tensor:
    cos_theta, sin_theta = theta.cos()[:, None], theta.sin()[:, None]
    return torch.stack(
        (cos_theta * q[..., 0] - sin_theta * q[..., 1],
         sin_theta * q[..., 0] + cos_theta * q[..., 1], q[..., 2]), dim=-1,
    )


def _probability_and_derivative(b: Tensor, sigma: Tensor, mask: Tensor, r: Tensor):
    zeta = (b - r[:, None]) / sigma
    probability = torch.where(mask, 0.5 * torch.special.erfc(zeta / math.sqrt(2)), 0.0).sum(-1)
    derivative = torch.where(
        mask, torch.exp(-0.5 * zeta.square()) / (math.sqrt(2 * math.pi) * sigma), 0.0,
    ).sum(-1)
    return probability, derivative


@torch.no_grad()
def compute_mf_imp(
    phi: Tensor,
    gradient: Tensor,
    d: Tensor,
    q: Tensor,
    feature_mask: Tensor,
    z_bar: Tensor,
    epsilon,
    d_min: float,
    *,
    stage_requested=None,
    d_col: float = 0.0002,
    root_max_iter: int = 12,
    root_atol_m: float = 1e-8,
    root_rtol: float = 1e-10,
    root_prob_atol: float = 1e-10,
    return_frozen_features: bool = False,
    required_override=None,
    pose_feature_gradient: Optional[Tensor] = None,
    no_obstacles: bool = False,
) -> MFImplicitResult:
    """Solve and linearize one implicit margin per requested stage.

    All returned risk quantities are float64 tensors on ``z_bar.device``.
    Status codes distinguish ordinary inactive rows from required failures.
    A failed required row has NaN coefficients and must never reach a solver.
    ``failure_features`` contains device references solely for failure dumps;
    callers should discard it after a successful evaluation.
    """
    validate_root_settings(root_max_iter, root_atol_m, root_rtol, root_prob_atol)
    if not math.isfinite(float(d_min)) or not math.isfinite(float(d_col)):
        raise ValueError("implicit MF thresholds must be finite")
    if z_bar.ndim != 2 or z_bar.shape[1] != 8:
        raise ValueError("z_bar must have shape (H, 8)")
    z_bar = z_bar.to(dtype=torch.float64)
    H = z_bar.shape[0]
    if d.ndim != 2 or d.shape[0] != H or q.shape != (*d.shape, 3):
        raise ValueError("feature d and q must have shapes (H, M) and (H, M, 3)")
    if feature_mask.shape != d.shape or feature_mask.dtype != torch.bool:
        raise ValueError("feature_mask must be bool with shape (H, M)")
    if phi.shape != (H,) or gradient.shape != (H, 3):
        raise ValueError("phi and gradient must have shapes (H,) and (H, 3)")
    if any(t.device != z_bar.device for t in (phi, gradient, d, q, feature_mask)):
        raise ValueError("implicit MF inputs must use the same device")
    epsilon_t = normalize_implicit_epsilon(epsilon, z_bar)
    requested = (
        torch.ones(H, dtype=torch.bool, device=z_bar.device)
        if stage_requested is None else
        torch.as_tensor(stage_requested, dtype=torch.bool, device=z_bar.device)
    )
    if requested.shape != (H,):
        raise ValueError("stage_requested must have shape (H,)")
    geometry_ok = torch.isfinite(phi) & torch.isfinite(gradient).all(-1)
    required = requested & ((phi > d_col) | ~geometry_ok)
    if required_override is not None:
        required = torch.as_tensor(required_override, dtype=torch.bool, device=z_bar.device)
        if required.shape != (H,):
            raise ValueError("required_override must have shape (H,)")
    if no_obstacles:
        required = torch.zeros_like(required)
    status = torch.full(
        (H,), int(MFImplicitRootStatus.NOT_REQUESTED),
        dtype=torch.int64, device=z_bar.device,
    )
    status = torch.where(requested & ~required, int(MFImplicitRootStatus.DOMAIN_INACTIVE), status)
    status = torch.where(required, int(MFImplicitRootStatus.SUCCESS), status)
    if no_obstacles:
        status.fill_(int(MFImplicitRootStatus.NO_OBSTACLES))
    count = feature_mask.sum(-1)
    nan = torch.full((H,), float("nan"), dtype=torch.float64, device=z_bar.device)
    zeros = torch.zeros_like(nan)
    iterations = torch.zeros(H, dtype=torch.int64, device=z_bar.device)
    A_out = torch.zeros_like(z_bar)
    c_out = torch.ones_like(nan)  # Inactive distance residual is one metre.
    if d.shape[1] == 0:
        status = torch.where(required, int(MFImplicitRootStatus.NO_VALID_FEATURES), status)
        A_out = torch.where(required[:, None], float("nan"), A_out)
        c_out = torch.where(required, nan, c_out)
        failure_features = {
            "d": d.to(torch.float64), "q": q.to(torch.float64),
            "mask": feature_mask, "sigma": d.to(torch.float64),
            "z_bar": z_bar, "epsilon": epsilon_t,
            "settings": {"d_min": float(d_min), "d_col": float(d_col),
                         "root_max_iter": root_max_iter, "root_atol_m": root_atol_m,
                         "root_rtol": root_rtol, "root_prob_atol": root_prob_atol},
        }
        return MFImplicitResult(phi, gradient, A_out, c_out, nan.clone(), zeros,
                                nan.clone(), nan.clone(), nan.clone(), iterations,
                                count, torch.zeros_like(required), status, required,
                                nan.clone(), nan.clone(), nan.clone(), nan.clone(), nan.clone(),
                                failure_features=failure_features)

    d64, q64 = d.to(torch.float64), q.to(torch.float64)
    # Invalid padding is replaced before arithmetic, not multiplied by zero.
    safe_q = torch.where(feature_mask[..., None], q64, torch.zeros_like(q64))
    safe_z = torch.where(torch.isfinite(z_bar), z_bar, torch.zeros_like(z_bar))
    variance_raw, variance_gradient_raw = projected_variance_and_gradient(safe_q, safe_z)
    feature_ok = ((torch.isfinite(d64) & torch.isfinite(q64).all(-1)) | ~feature_mask).all(-1)
    variance_ok_slots = (
        torch.isfinite(variance_raw) & (variance_raw > 0)
        & torch.isfinite(variance_gradient_raw).all(-1)
    )
    variance_ok = (variance_ok_slots | ~feature_mask).all(-1)
    nominal_ok = torch.isfinite(z_bar).all(-1)
    epsilon_ok = torch.isfinite(epsilon_t) & (epsilon_t > 0) & (epsilon_t < 0.5)
    eligible = required.clone()
    for ok, code in (
        (geometry_ok, MFImplicitRootStatus.INVALID_GEOMETRY),
        (nominal_ok, MFImplicitRootStatus.INVALID_NOMINAL),
        (epsilon_ok, MFImplicitRootStatus.INVALID_EPSILON),
        (count > 0, MFImplicitRootStatus.NO_VALID_FEATURES),
        (feature_ok, MFImplicitRootStatus.INVALID_FEATURE),
        (variance_ok, MFImplicitRootStatus.INVALID_VARIANCE),
    ):
        status = torch.where(eligible & ~ok, int(code), status)
        eligible = eligible & ok
    good_slots = (
        feature_mask & torch.isfinite(d64)
        & torch.isfinite(q64).all(-1) & variance_ok_slots
    )
    sigma = torch.where(good_slots, variance_raw, torch.ones_like(variance_raw)).sqrt()
    variance_gradient = torch.where(
        good_slots[..., None], variance_gradient_raw,
        torch.zeros_like(variance_gradient_raw),
    )
    safe_d = torch.where(good_slots, d64, torch.full_like(d64, float(d_min)))
    safe_q = torch.where(good_slots[..., None], q64, torch.zeros_like(q64))
    b = safe_d - d_min
    F0, _ = _probability_and_derivative(b, sigma, feature_mask, zeros)
    probability_sum = torch.where(feature_ok & variance_ok & nominal_ok, F0, nan)
    if no_obstacles:
        probability_sum = zeros.clone()
    safe_epsilon = torch.where(epsilon_ok, epsilon_t, torch.full_like(epsilon_t, 0.2))
    safe_count = count.clamp_min(1).to(torch.float64)
    lower_shift = sigma * torch.special.ndtri(safe_epsilon / safe_count)[:, None]
    upper_shift = sigma * torch.special.ndtri(safe_epsilon)[:, None]
    lower_features = b + lower_shift
    upper_features = b + upper_shift
    inf = torch.full_like(d64, float("inf"))
    lo = torch.where(feature_mask, lower_features, inf).amin(-1)
    hi = torch.where(feature_mask, upper_features, inf).amin(-1)
    risk_ready = eligible.clone()
    lo = torch.where(eligible, lo, zeros)
    hi = torch.where(eligible, hi, zeros)
    root_mask = feature_mask & eligible[:, None]
    machine_eps = torch.finfo(torch.float64).eps
    probability_roundoff = 64 * machine_eps * safe_epsilon
    # Near a cancelling sum b + sigma*Q, an ulp of r can be arbitrarily
    # smaller than the rounding in b-r used by the CDF.  Outward correction
    # must use the arithmetic scale of the selected endpoint's summands.
    lo_index = torch.where(feature_mask, lower_features, inf).argmin(-1, keepdim=True)
    hi_index = torch.where(feature_mask, upper_features, inf).argmin(-1, keepdim=True)
    lo_scale = torch.gather(b.abs() + lower_shift.abs(), 1, lo_index).squeeze(-1) + lo.abs()
    hi_scale = torch.gather(b.abs() + upper_shift.abs(), 1, hi_index).squeeze(-1) + hi.abs()
    lo_roundoff = 4 * machine_eps * lo_scale
    hi_roundoff = 4 * machine_eps * hi_scale
    neg_inf, pos_inf = torch.full_like(lo, -float("inf")), torch.full_like(hi, float("inf"))
    # Only roundoff-sized endpoint sign errors permit outward rounding.  A
    # genuinely invalid analytic bracket is a failure, never a search step.
    for _ in range(4):
        Flo, _ = _probability_and_derivative(b, sigma, root_mask, lo)
        Fhi, _ = _probability_and_derivative(b, sigma, root_mask, hi)
        lo = torch.where(
            eligible & (Flo > safe_epsilon) & (Flo - safe_epsilon <= probability_roundoff),
            torch.nextafter(lo - lo_roundoff, neg_inf), lo,
        )
        hi = torch.where(
            eligible & (Fhi < safe_epsilon) & (safe_epsilon - Fhi <= probability_roundoff),
            torch.nextafter(hi + hi_roundoff, pos_inf), hi,
        )
    Flo, _ = _probability_and_derivative(b, sigma, root_mask, lo)
    Fhi, derivative_hi = _probability_and_derivative(b, sigma, root_mask, hi)
    bracket_ok = (
        torch.isfinite(lo) & torch.isfinite(hi) & (lo <= hi)
        & torch.isfinite(Flo) & torch.isfinite(Fhi)
        & (Flo <= safe_epsilon) & (Fhi >= safe_epsilon)
    )
    status = torch.where(eligible & ~bracket_ok, int(MFImplicitRootStatus.INVALID_BRACKET), status)
    eligible = eligible & bracket_ok
    r = torch.where(eligible, hi, zeros)
    running = eligible & (count > 1)
    # The Python loop has a fixed length.  It contains no tensor-dependent
    # control flow or host/device transfers, including for converged stages.
    for _ in range(root_max_iter):
        value, derivative = _probability_and_derivative(b, sigma, root_mask, r)
        derivative_ok = torch.isfinite(value) & torch.isfinite(derivative) & (derivative > 0)
        status = torch.where(
            running & ~derivative_ok, int(MFImplicitRootStatus.INVALID_DERIVATIVE), status,
        )
        running = running & derivative_ok
        residual = value - safe_epsilon
        lo = torch.where(running & (residual < 0), torch.maximum(lo, r), lo)
        hi = torch.where(running & (residual > 0), torch.minimum(hi, r), hi)
        safe_derivative = torch.where(running, derivative, torch.ones_like(derivative))
        correction = torch.where(running, residual, zeros) / safe_derivative
        candidate = r - correction
        fixed = running & (
            (residual == 0) | ((candidate == r) & (residual.abs() <= root_prob_atol))
        )
        active_update = running & ~fixed
        finite_step = torch.isfinite(correction) & torch.isfinite(candidate)
        status = torch.where(
            active_update & ~finite_step, int(MFImplicitRootStatus.NONFINITE_STEP), status,
        )
        active_update = active_update & finite_step
        distance_roundoff = 32 * machine_eps * (
            r.abs() + lo.abs() + hi.abs() + correction.abs() + lo_scale + hi_scale
        )
        inside = (candidate >= lo - distance_roundoff) & (candidate <= hi + distance_roundoff)
        status = torch.where(
            active_update & ~inside, int(MFImplicitRootStatus.BRACKET_ESCAPE), status,
        )
        active_update = active_update & inside
        # This correction handles endpoint rounding only; larger escapes have
        # already failed.  There is no general step clipping or bisection.
        candidate = torch.minimum(torch.maximum(candidate, lo), hi)
        stagnated = active_update & (candidate == r) & (residual.abs() > root_prob_atol)
        status = torch.where(stagnated, int(MFImplicitRootStatus.STAGNATION), status)
        active_update = active_update & ~stagnated
        r = torch.where(active_update, candidate, r)
        iterations += active_update.to(torch.int64)
        running = active_update

    # Evaluate the last Newton candidate, including after the last permitted
    # update.  Global bracket width is only a diagnostic, not convergence.
    value, derivative = _probability_and_derivative(b, sigma, root_mask, r)
    candidate_ok = eligible & (status == int(MFImplicitRootStatus.SUCCESS))
    final_derivative_ok = torch.isfinite(value) & torch.isfinite(derivative) & (derivative > 0)
    status = torch.where(
        candidate_ok & ~final_derivative_ok,
        int(MFImplicitRootStatus.INVALID_DERIVATIVE), status,
    )
    candidate_ok = candidate_ok & final_derivative_ok
    residual = value - safe_epsilon
    probability_ok = residual.abs() <= root_prob_atol
    status = torch.where(
        candidate_ok & ~probability_ok, int(MFImplicitRootStatus.MAX_ITERATIONS), status,
    )
    tau = root_atol_m + root_rtol * r.abs()
    local_lo = torch.nextafter(r - tau, neg_inf)
    local_hi = torch.nextafter(r + tau, pos_inf)
    F_local_lo, _ = _probability_and_derivative(b, sigma, root_mask, local_lo)
    F_local_hi, _ = _probability_and_derivative(b, sigma, root_mask, local_hi)
    local_ok = (
        torch.isfinite(local_lo) & torch.isfinite(local_hi)
        & (local_lo < r) & (local_hi > r)
        & torch.isfinite(F_local_lo) & torch.isfinite(F_local_hi)
        & (F_local_lo <= safe_epsilon) & (F_local_hi >= safe_epsilon)
    )
    status = torch.where(
        candidate_ok & probability_ok & ~local_ok,
        int(MFImplicitRootStatus.LOCAL_BRACKET_FAILED), status,
    )
    root_valid = candidate_ok & probability_ok & local_ok

    zeta = (b - r[:, None]) / sigma
    logits = -0.5 * zeta.square() - sigma.log() - 0.5 * math.log(2 * math.pi)
    weight_mask = feature_mask & root_valid[:, None]
    logits = torch.where(weight_mask, logits, torch.full_like(logits, -float("inf")))
    # Explicit finite dummy for rows with no valid root avoids all--inf
    # softmax.  Those rows never contribute coefficients or an accepted root.
    dummy = torch.zeros_like(weight_mask)
    dummy[:, 0] = ~root_valid
    logits = torch.where(dummy, torch.zeros_like(logits), logits)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(weight_mask, weights, torch.zeros_like(weights))
    world_q = (
        _feature_world_gradient(safe_q, safe_z[:, 2])
        if pose_feature_gradient is None else pose_feature_gradient.to(torch.float64)
    )
    if world_q.shape != q.shape:
        raise ValueError("pose_feature_gradient must have shape (H, M, 3)")
    world_q = torch.where(good_slots[..., None], world_q, torch.zeros_like(world_q))
    A = torch.zeros_like(z_bar)
    A[:, :3] = (weights[..., None] * world_q).sum(1)
    A[:, 4:] = -(
        weights[..., None] * zeta[..., None] * variance_gradient
        / (2 * sigma[..., None])
    ).sum(1)
    c = r - (A * z_bar).sum(-1)
    coefficient_ok = torch.isfinite(A).all(-1) & torch.isfinite(c)
    status = torch.where(
        root_valid & ~coefficient_ok,
        int(MFImplicitRootStatus.INVALID_COEFFICIENTS), status,
    )
    root_valid = root_valid & coefficient_ok
    A_out = torch.where(root_valid[:, None], A, A_out)
    c_out = torch.where(root_valid, c, c_out)
    A_out = torch.where((required & ~root_valid)[:, None], float("nan"), A_out)
    c_out = torch.where(required & ~root_valid, nan, c_out)
    failure_features = {
        "d": d64, "q": q64, "mask": feature_mask,
        "sigma": torch.where(feature_mask, variance_raw.sqrt(), torch.ones_like(variance_raw)),
        "z_bar": z_bar, "epsilon": epsilon_t,
        "settings": {"d_min": float(d_min), "d_col": float(d_col),
                     "root_max_iter": root_max_iter, "root_atol_m": root_atol_m,
                     "root_rtol": root_rtol, "root_prob_atol": root_prob_atol},
    }
    frozen = None
    if return_frozen_features:
        frozen = {"d": d64, "q": q64, "mask": feature_mask,
                  "z_bar": z_bar, "pose_feature_gradient": world_q,
                  "phi": phi, "gradient": gradient, "mf_required": required}
    # Before a usable bracket exists there is no Newton candidate.  Preserve
    # the analytic upper endpoint for bracket failures, and leave candidate
    # metrics missing for invalid feature/variance inputs.
    reported_candidate = torch.where(bracket_ok, r, hi)
    reported_value = torch.where(bracket_ok, value, Fhi)
    reported_derivative = torch.where(bracket_ok, derivative, derivative_hi)
    return MFImplicitResult(
        phi=phi, gradient=gradient, A=A_out, c=c_out,
        margin=torch.where(root_valid, r, nan), probability_sum=probability_sum,
        root_residual=torch.where(risk_ready, reported_value - safe_epsilon, nan),
        root_bracket_width=torch.where(risk_ready, hi - lo, nan),
        root_error_bound=torch.where(root_valid, torch.maximum(r - local_lo, local_hi - r), nan),
        root_iterations=iterations, feature_count=count, root_valid=root_valid,
        root_status=status, mf_required=required,
        root_candidate=torch.where(risk_ready, reported_candidate, nan),
        root_value=torch.where(risk_ready, reported_value, nan),
        root_derivative=torch.where(risk_ready, reported_derivative, nan),
        root_lower=torch.where(risk_ready, lo, nan), root_upper=torch.where(risk_ready, hi, nan),
        frozen_features=frozen, failure_features=failure_features,
    )


@torch.no_grad()
def evaluate_mf_imp_frozen(
    snapshot: dict, z_star: Tensor, epsilon, d_min: float, **kwargs,
) -> MFImplicitResult:
    """Re-evaluate nominal distance affines and covariance at returned states.

    Feature q, world pose gradients, masks, original frame, and requested
    domain stay fixed.  This function never replaces a nominal cache.
    """
    nominal = snapshot["z_bar"]
    z_star = torch.as_tensor(z_star, dtype=torch.float64, device=nominal.device)
    if z_star.shape != nominal.shape:
        raise ValueError("z_star must have the snapshot nominal shape")
    displacement = z_star[:, :3] - nominal[:, :3]
    d = snapshot["d"] + (snapshot["pose_feature_gradient"] * displacement[:, None, :]).sum(-1)
    phi = snapshot["phi"].to(torch.float64) + (
        snapshot["gradient"].to(torch.float64) * displacement
    ).sum(-1)
    kwargs.pop("stage_requested", None)
    kwargs.pop("required_override", None)
    return compute_mf_imp(
        phi, snapshot["gradient"], d, snapshot["q"], snapshot["mask"], z_star,
        epsilon, d_min, stage_requested=snapshot["mf_required"],
        required_override=snapshot["mf_required"],
        pose_feature_gradient=snapshot["pose_feature_gradient"], **kwargs,
    )
