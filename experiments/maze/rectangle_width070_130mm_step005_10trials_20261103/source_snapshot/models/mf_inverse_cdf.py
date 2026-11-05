"""Analytic inverse-CDF MF rows for a frozen nominal geometry model.

The risk arithmetic is float64.  No feature selection, nonlinear root solve,
or covariance flooring is performed here.  The extremely small lower tail is
explicitly clipped, with a zero derivative consistent with that clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Optional

import torch
from torch import Tensor

from models.mf_implicit import projected_variance_and_gradient


class MFInverseCDFStatus(IntEnum):
    NOT_REQUESTED = 0
    DOMAIN_INACTIVE = 1
    NO_OBSTACLES = 2
    SUCCESS = 3
    NO_VALID_FEATURES = 4
    INVALID_GEOMETRY = 5
    INVALID_NOMINAL = 6
    INVALID_EPSILON = 7
    INVALID_FEATURE = 8
    INVALID_VARIANCE = 9
    INVALID_PROBABILITY = 10
    INVALID_COEFFICIENTS = 11


class MFInverseCDFBranch(IntEnum):
    INACTIVE = 0
    INVERSE_CDF = 1
    LINEAR_EXTENSION = 2
    TAIL_CLIPPED = 3
    INVALID = 4


ICDF_STATUS_NAMES = {int(item): item.name.lower() for item in MFInverseCDFStatus}
ICDF_BRANCH_NAMES = {int(item): item.name.lower() for item in MFInverseCDFBranch}
ICDF_PROBABILITY_FLOOR = torch.finfo(torch.float64).tiny


@dataclass
class MFInverseCDFResult:
    phi: Tensor
    gradient: Tensor
    A: Tensor
    c: Tensor
    residual: Tensor
    probability_sum: Tensor
    feature_count: Tensor
    valid: Tensor
    required: Tensor
    status: Tensor
    branch: Tensor
    tail_clipped: Tensor
    frozen_features: Optional[dict] = None


def _normalize_epsilon(epsilon, z_bar: Tensor) -> Tensor:
    if isinstance(epsilon, (float, int)):
        if not math.isfinite(float(epsilon)) or not ICDF_PROBABILITY_FLOOR < epsilon < 0.5:
            raise ValueError("inverse-CDF MF epsilon must satisfy float64.tiny < epsilon < 0.5")
    result = torch.as_tensor(epsilon, dtype=torch.float64, device=z_bar.device)
    if result.ndim == 0:
        result = result.expand(z_bar.shape[0])
    if result.shape != (z_bar.shape[0],):
        raise ValueError("epsilon must be a scalar or have shape (H,)")
    return result


def _world_feature_gradient(q: Tensor, theta: Tensor) -> Tensor:
    cos_theta, sin_theta = theta.cos()[:, None], theta.sin()[:, None]
    return torch.stack(
        (cos_theta * q[..., 0] - sin_theta * q[..., 1],
         sin_theta * q[..., 0] + cos_theta * q[..., 1], q[..., 2]), dim=-1,
    )


@torch.no_grad()
def compute_mf_icdf(
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
    no_obstacles: bool = False,
    return_frozen_features: bool = False,
    required_override=None,
    pose_feature_gradient: Optional[Tensor] = None,
) -> MFInverseCDFResult:
    """Return one affine row per stage, with explicit validity metadata.

    States are ``[x, y, theta, s, log(L_ff), log(L_ll),
    log(L_theta,theta), L_theta,l]``.  The strict signed-PSDF gate ``phi > d_min``
    and requested stage window determine ``required``.  Invalid required rows
    have NaN coefficients and must never reach the solver.  Normal inactive
    rows use ``A=0, c=1``; their missing residual is not an evaluated safety margin.

    Geometry and covariance inputs are validated with batched tensor masks;
    no per-stage device synchronization is needed on CUDA.
    """
    if not math.isfinite(float(d_min)):
        raise ValueError("inverse-CDF MF d_min must be finite")
    if z_bar.ndim != 2 or z_bar.shape[1] != 8:
        raise ValueError("z_bar must have shape (H, 8)")
    if not z_bar.is_floating_point():
        raise TypeError("z_bar must be a floating-point tensor")
    z_bar = z_bar.to(dtype=torch.float64)
    H = z_bar.shape[0]
    if d.ndim != 2 or d.shape[0] != H or q.shape != (*d.shape, 3):
        raise ValueError("feature d and q must have shapes (H, M) and (H, M, 3)")
    if feature_mask.shape != d.shape or feature_mask.dtype != torch.bool:
        raise ValueError("feature_mask must be bool with shape (H, M)")
    if phi.shape != (H,) or gradient.shape != (H, 3):
        raise ValueError("phi and gradient must have shapes (H,) and (H, 3)")
    if any(t.device != z_bar.device for t in (phi, gradient, d, q, feature_mask)):
        raise ValueError("inverse-CDF MF inputs must use the same device")
    epsilon_t = _normalize_epsilon(epsilon, z_bar)
    requested = (
        torch.ones(H, dtype=torch.bool, device=z_bar.device)
        if stage_requested is None else
        torch.as_tensor(stage_requested, dtype=torch.bool, device=z_bar.device)
    )
    if requested.shape != (H,):
        raise ValueError("stage_requested must have shape (H,)")
    geometry_ok = torch.isfinite(phi) & torch.isfinite(gradient).all(-1)
    required = requested & ((phi > d_min) | ~geometry_ok)
    if required_override is not None:
        required = torch.as_tensor(required_override, dtype=torch.bool, device=z_bar.device)
        if required.shape != (H,):
            raise ValueError("required_override must have shape (H,)")
    if no_obstacles:
        required = torch.zeros_like(required)

    status = torch.full((H,), int(MFInverseCDFStatus.NOT_REQUESTED),
                        dtype=torch.int64, device=z_bar.device)
    status = torch.where(requested & ~required, int(MFInverseCDFStatus.DOMAIN_INACTIVE), status)
    status = torch.where(required, int(MFInverseCDFStatus.SUCCESS), status)
    if no_obstacles:
        status.fill_(int(MFInverseCDFStatus.NO_OBSTACLES))

    count = feature_mask.sum(-1)
    d64, q64 = d.to(torch.float64), q.to(torch.float64)
    finite_features = torch.isfinite(d64) & torch.isfinite(q64).all(-1)
    feature_ok = (finite_features | ~feature_mask).all(-1)
    safe_q = torch.where((feature_mask & finite_features)[..., None], q64, 0.0)
    safe_z = torch.where(torch.isfinite(z_bar), z_bar, 0.0)
    variance, variance_gradient = projected_variance_and_gradient(safe_q, safe_z)
    variance_ok_slots = (
        torch.isfinite(variance) & (variance > 0)
        & torch.isfinite(variance_gradient).all(-1)
    )
    variance_ok = (variance_ok_slots | ~feature_mask).all(-1)
    nominal_ok = torch.isfinite(z_bar).all(-1)
    epsilon_ok = (
        torch.isfinite(epsilon_t) & (epsilon_t > ICDF_PROBABILITY_FLOOR)
        & (epsilon_t < 0.5)
    )
    good_slots = feature_mask & finite_features & variance_ok_slots
    safe_variance = torch.where(good_slots, variance, 1.0)
    sigma = safe_variance.sqrt()
    safe_d = torch.where(good_slots, d64, float(d_min))
    zeta = (float(d_min) - safe_d) / sigma
    standardization_ok = (torch.isfinite(zeta) | ~feature_mask).all(-1)
    safe_zeta = torch.where(torch.isfinite(zeta), zeta, 0.0)
    probabilities = torch.where(
        good_slots, 0.5 * torch.special.erfc(-safe_zeta / math.sqrt(2.0)), 0.0,
    )
    R = probabilities.sum(-1)
    probability_ok = standardization_ok & torch.isfinite(R) & (R >= 0)

    eligible = required.clone()
    for ok, code in (
        (geometry_ok, MFInverseCDFStatus.INVALID_GEOMETRY),
        (nominal_ok, MFInverseCDFStatus.INVALID_NOMINAL),
        (epsilon_ok, MFInverseCDFStatus.INVALID_EPSILON),
        (count > 0, MFInverseCDFStatus.NO_VALID_FEATURES),
        (feature_ok, MFInverseCDFStatus.INVALID_FEATURE),
        (variance_ok, MFInverseCDFStatus.INVALID_VARIANCE),
        (probability_ok, MFInverseCDFStatus.INVALID_PROBABILITY),
    ):
        status = torch.where(eligible & ~ok, int(code), status)
        eligible = eligible & ok

    nan = torch.full((H,), float("nan"), dtype=torch.float64, device=z_bar.device)
    risk_ok = feature_ok & variance_ok & nominal_ok & probability_ok
    probability_sum = torch.where(risk_ok, R, nan)
    if no_obstacles:
        probability_sum = torch.zeros_like(R)
    safe_epsilon = torch.where(epsilon_ok, epsilon_t, 0.2)
    # -ndtri(epsilon) avoids rounding 1-epsilon to one for small budgets.
    gamma = -torch.special.ndtri(safe_epsilon)
    quantile = torch.special.ndtri(R.clamp(min=ICDF_PROBABILITY_FLOOR, max=0.5))
    extended = R > 0.5
    clipped = R <= ICDF_PROBABILITY_FLOOR
    psi = torch.where(
        extended, -math.sqrt(2.0 * math.pi) * (R - 0.5) - gamma,
        -quantile - gamma,
    )
    # Each feature probability is <= R.  On R<=1/2, zeta<=quantile<=0,
    # so this PDF ratio is <=1 and needs no division by an underflowing PDF.
    log_weights = torch.where(
        extended[:, None], -0.5 * safe_zeta.square(),
        0.5 * (quantile[:, None].square() - safe_zeta.square()),
    )
    derivative_slots = good_slots & eligible[:, None] & ~clipped[:, None]
    # Mask before exp to avoid overflow in unused branches and padding.
    weights = torch.exp(torch.where(derivative_slots, log_weights, -float("inf")))
    world_q = (
        _world_feature_gradient(safe_q, safe_z[:, 2])
        if pose_feature_gradient is None else
        torch.as_tensor(pose_feature_gradient, dtype=torch.float64, device=z_bar.device)
    )
    if world_q.shape != q.shape:
        raise ValueError("pose_feature_gradient must have shape (H, M, 3)")
    world_q = torch.where(derivative_slots[..., None], world_q, 0.0)
    variance_gradient = torch.where(derivative_slots[..., None], variance_gradient, 0.0)
    gradient_zeta = torch.where(derivative_slots, safe_zeta, 0.0)
    A = torch.zeros_like(z_bar)
    A[:, :3] = ((weights / sigma)[..., None] * world_q).sum(1)
    A[:, 4:] = (
        (weights * gradient_zeta)[..., None]
        * (variance_gradient / (2.0 * safe_variance[..., None]))
    ).sum(1)
    c = psi - (A * safe_z).sum(-1)
    coefficient_ok = torch.isfinite(A).all(-1) & torch.isfinite(c) & torch.isfinite(psi)
    status = torch.where(eligible & ~coefficient_ok,
                         int(MFInverseCDFStatus.INVALID_COEFFICIENTS), status)
    valid = eligible & coefficient_ok
    A_out = torch.where(valid[:, None], A, torch.zeros_like(A))
    c_out = torch.where(valid, c, torch.ones_like(c))
    A_out = torch.where((required & ~valid)[:, None], float("nan"), A_out)
    c_out = torch.where(required & ~valid, nan, c_out)
    branch = torch.full_like(status, int(MFInverseCDFBranch.INACTIVE))
    branch = torch.where(valid, int(MFInverseCDFBranch.INVERSE_CDF), branch)
    branch = torch.where(valid & extended, int(MFInverseCDFBranch.LINEAR_EXTENSION), branch)
    branch = torch.where(valid & clipped, int(MFInverseCDFBranch.TAIL_CLIPPED), branch)
    branch = torch.where(required & ~valid, int(MFInverseCDFBranch.INVALID), branch)
    frozen = None
    if return_frozen_features:
        # Keep geometric derivatives even for clipped rows: a candidate can
        # leave that tail while retaining this frozen geometry.
        frozen_world_q = (
            _world_feature_gradient(safe_q, safe_z[:, 2])
            if pose_feature_gradient is None else
            torch.as_tensor(pose_feature_gradient, dtype=torch.float64, device=z_bar.device)
        )
        frozen = {
            "d": d64.clone(), "q": q64.clone(), "mask": feature_mask.clone(),
            "z_bar": z_bar.clone(), "pose_feature_gradient": frozen_world_q.clone(),
            "phi": phi.clone(), "gradient": gradient.clone(),
            "mf_required": required.clone(), "no_obstacles": bool(no_obstacles),
        }
    return MFInverseCDFResult(
        phi=phi, gradient=gradient, A=A_out, c=c_out,
        residual=torch.where(valid, psi, nan), probability_sum=probability_sum,
        feature_count=count, valid=valid, required=required, status=status,
        branch=branch, tail_clipped=valid & clipped, frozen_features=frozen,
    )


@torch.no_grad()
def evaluate_mf_icdf_frozen(
    snapshot: dict, z_star: Tensor, epsilon, d_min: float, **kwargs,
) -> MFInverseCDFResult:
    """Evaluate candidate states without changing nominal feature geometry.

    Distances follow their nominal affines.  Feature gradients, original
    body frame, masks, and the required-stage gate remain fixed.
    """
    nominal = snapshot["z_bar"]
    z_star = torch.as_tensor(z_star, dtype=torch.float64, device=nominal.device)
    if z_star.shape != nominal.shape:
        raise ValueError("z_star must have the snapshot nominal shape")
    displacement = z_star[:, :3] - nominal[:, :3]
    d = snapshot["d"] + (
        snapshot["pose_feature_gradient"] * displacement[:, None, :]
    ).sum(-1)
    phi = snapshot["phi"].to(torch.float64) + (
        snapshot["gradient"].to(torch.float64) * displacement
    ).sum(-1)
    for key in ("stage_requested", "required_override", "pose_feature_gradient", "no_obstacles"):
        kwargs.pop(key, None)
    return compute_mf_icdf(
        phi, snapshot["gradient"], d, snapshot["q"], snapshot["mask"], z_star,
        epsilon, d_min, stage_requested=snapshot["mf_required"],
        required_override=snapshot["mf_required"],
        pose_feature_gradient=snapshot["pose_feature_gradient"],
        no_obstacles=snapshot.get("no_obstacles", False), **kwargs,
    )
