"""Shared reduced log-Cholesky covariance model for the CC controllers.

The state is ``[log(L_ff), log(L_ll), log(L_thetatheta), L_thetal]``.
Covariance is retained only in the forward and lateral/heading blocks; dropping
the other cross terms is a model reduction, not a numerical PSD repair.
"""

from numbers import Integral

import numpy as np

from models.dd_dynamics import symbolic_dd_step as _symbolic_dd_step


def _finite_array(value, trailing_shape, name):
    array = np.asarray(value, dtype=float)
    if (
        array.ndim < len(trailing_shape)
        or array.shape[-len(trailing_shape):] != trailing_shape
    ):
        raise ValueError(f"{name} must have trailing shape {trailing_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def covariance_from_log_cholesky(xi):
    """Reconstruct reduced covariance from a single or batched ``(..., 4)`` state."""
    xi = _finite_array(xi, (4,), "log-Cholesky state")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        diagonal_variances = np.exp(2.0 * xi[..., :3])
        lateral_factor = np.exp(xi[..., 1])
        cross_factor = xi[..., 3]
        covariance = np.zeros(xi.shape[:-1] + (3, 3), dtype=float)
        covariance[..., 0, 0] = diagonal_variances[..., 0]
        covariance[..., 1, 1] = diagonal_variances[..., 1]
        covariance[..., 2, 2] = cross_factor**2 + diagonal_variances[..., 2]
        covariance[..., 1, 2] = covariance[..., 2, 1] = cross_factor * lateral_factor
    if not np.all(np.isfinite(covariance)) or np.any(diagonal_variances <= 0.0):
        raise ValueError("log-Cholesky covariance exceeds the finite positive numeric range")
    return covariance


def log_cholesky_from_covariance(covariance):
    """Invert the retained positive-definite blocks of ``(..., 3, 3)`` covariance.

    Forward/lateral and forward/heading cross-covariances are intentionally
    discarded. No clipping, variance floor, or eigenvalue repair is applied.
    """
    covariance = _finite_array(covariance, (3, 3), "covariance")
    if not np.allclose(
        covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-15
    ):
        raise ValueError("covariance must be symmetric")
    forward_variance = covariance[..., 0, 0]
    lateral_variance = covariance[..., 1, 1]
    if np.any(forward_variance <= 0.0) or np.any(lateral_variance <= 0.0):
        raise ValueError("retained covariance blocks must be positive definite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cross_factor = covariance[..., 1, 2] / np.sqrt(lateral_variance)
        conditional_heading_variance = covariance[..., 2, 2] - cross_factor**2
    if (
        not np.all(np.isfinite(conditional_heading_variance))
        or np.any(conditional_heading_variance <= 0.0)
    ):
        raise ValueError("retained lateral/heading covariance block must be positive definite")
    return np.stack(
        (
            0.5 * np.log(forward_variance),
            0.5 * np.log(lateral_variance),
            0.5 * np.log(conditional_heading_variance),
            cross_factor,
        ),
        axis=-1,
    )


def initial_local_covariance(param, state=None):
    """Build a reduced body-frame covariance from configured initial uncertainty.

    ``initial_pose_covariance``, when supplied, is an estimator covariance in
    world coordinates and requires the current state's heading. Otherwise the
    existing positive ``sigma_f0``, ``sigma_l0`` and ``sigma_psi0`` are used.
    """
    world_covariance = getattr(param, "initial_pose_covariance", None)
    if world_covariance is None:
        sigmas = np.asarray(
            [
                getattr(param, name, 0.0)
                for name in ("sigma_f0", "sigma_l0", "sigma_psi0")
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("initial covariance standard deviations must be finite and positive")
        with np.errstate(over="ignore", under="ignore"):
            covariance = np.diag(sigmas**2)
    else:
        covariance = _finite_array(world_covariance, (3, 3), "initial_pose_covariance")
        if covariance.shape != (3, 3):
            raise ValueError("initial_pose_covariance must have shape (3, 3)")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-15):
            raise ValueError("initial_pose_covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("initial_pose_covariance must be positive definite") from exc
        if state is None:
            raise ValueError("initial_pose_covariance requires the current pose for frame conversion")
        theta = float(np.asarray(state._x, dtype=float).reshape(-1)[2])
        if not np.isfinite(theta):
            raise ValueError("current heading must be finite")
        cosine, sine = np.cos(theta), np.sin(theta)
        transform = np.array([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        covariance = transform.T @ covariance @ transform
        covariance[0, 1] = covariance[1, 0] = 0.0
        covariance[0, 2] = covariance[2, 0] = 0.0
    log_cholesky_from_covariance(covariance)
    return covariance


def resolve_robust_horizon(param):
    """Return the covariance propagation horizon, validated in ``[1, N]``."""
    horizon = param.horizon
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    robust_horizon = getattr(param, "robust_horizon", None)
    if robust_horizon is None:
        return min(10, int(horizon))
    if (
        isinstance(robust_horizon, bool)
        or not isinstance(robust_horizon, Integral)
        or not 1 <= robust_horizon <= horizon
    ):
        raise ValueError("robust_horizon must be an integer between 1 and horizon")
    return int(robust_horizon)


def _noise_coefficients(param):
    names = (
        "q_f0", "q_l0", "q_psi0", "alpha_f", "alpha_v", "alpha_kappa",
        "beta_v", "beta_kappa", "beta_omega",
    )
    values = tuple(float(getattr(param, name, 0.0)) for name in names)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("process noise coefficients must be finite and nonnegative")
    return values


def _step_duration(dt):
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("covariance step duration must be finite and positive")
    return dt


def _transition_components(xi, v, omega, kappa, dt, param, exp, fabs, log, sqrt):
    """Common algebra keeps NumPy rollouts and CasADi model transitions identical."""
    (
        q_f0, q_l0, q_psi0, alpha_f, alpha_v, alpha_kappa,
        beta_v, beta_kappa, beta_omega,
    ) = _noise_coefficients(param)
    dt = _step_duration(dt)
    speed_squared = v * v
    curvature_speed_squared = speed_squared * fabs(kappa)
    noise_f = dt * (q_f0 + alpha_f * speed_squared)
    noise_l = dt * (q_l0 + alpha_v * speed_squared + alpha_kappa * curvature_speed_squared)
    noise_heading = dt * (
        q_psi0 + beta_v * speed_squared + beta_kappa * curvature_speed_squared
        + beta_omega * omega * omega
    )

    variance_f = exp(2.0 * xi[0])
    factor_l = exp(xi[1])
    variance_l = exp(2.0 * xi[1])
    conditional_heading_variance = exp(2.0 * xi[2])
    cross_factor = xi[3]
    variance_heading = cross_factor * cross_factor + conditional_heading_variance
    rotation = dt * omega
    translation = dt * v

    # Retained entries of A Sigma A.T + dt diag(q), where
    # A = [[1, dt*omega, 0], [-dt*omega, 1, dt*v], [0, 0, 1]].
    next_variance_f = variance_f + rotation * rotation * variance_l + noise_f
    lateral_extra = rotation * rotation * variance_f + noise_l
    translated_factor_l = factor_l + translation * cross_factor
    next_variance_l = (
        translated_factor_l * translated_factor_l
        + translation * translation * conditional_heading_variance
        + lateral_extra
    )
    next_cross_covariance = (
        cross_factor * translated_factor_l
        + translation * conditional_heading_variance
    )
    # det(lateral/heading block) / next_variance_l, factored to avoid
    # subtracting almost equal marginal and cross-covariance terms.
    next_conditional_heading_variance = (
        variance_l * conditional_heading_variance + lateral_extra * variance_heading
    ) / next_variance_l + noise_heading
    return (
        0.5 * log(next_variance_f),
        0.5 * log(next_variance_l),
        0.5 * log(next_conditional_heading_variance),
        next_cross_covariance / sqrt(next_variance_l),
    )


def propagate_log_cholesky(xi, v, omega, kappa, dt, param):
    """Predict one reduced covariance step; the caller applies horizon freezing."""
    xi = _finite_array(xi, (4,), "log-Cholesky state")
    inputs = tuple(np.asarray(value, dtype=float) for value in (v, omega, kappa))
    if any(not np.all(np.isfinite(value)) for value in inputs):
        raise ValueError("covariance prediction inputs must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        components = _transition_components(
            np.moveaxis(xi, -1, 0), *inputs, dt, param, np.exp, np.abs, np.log, np.sqrt
        )
        result = np.stack(np.broadcast_arrays(*components), axis=-1)
    covariance_from_log_cholesky(result)
    return result


def symbolic_covariance_from_log_cholesky(xi):
    """CasADi reconstruction of the same positive-definite factor product."""
    import casadi as ca

    forward_variance = ca.exp(2.0 * xi[0])
    lateral_variance = ca.exp(2.0 * xi[1])
    cross_covariance = xi[3] * ca.exp(xi[1])
    heading_variance = xi[3] * xi[3] + ca.exp(2.0 * xi[2])
    return ca.vertcat(
        ca.horzcat(forward_variance, 0.0, 0.0),
        ca.horzcat(0.0, lateral_variance, cross_covariance),
        ca.horzcat(0.0, cross_covariance, heading_variance),
    )


def symbolic_log_cholesky_transition(xi, v, omega, kappa, dt, param):
    """CasADi one-step prediction matching :func:`propagate_log_cholesky`."""
    import casadi as ca

    return ca.vertcat(*_transition_components(
        xi, v, omega, kappa, dt, param, ca.exp, ca.fabs, ca.log, ca.sqrt
    ))


def symbolic_dd_step(pose, v, omega, dt):
    """One RK4 differential-drive pose step with constant physical inputs."""
    return _symbolic_dd_step(pose, v, omega, _step_duration(dt))
