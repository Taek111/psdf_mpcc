"""Pure cubic-spline path helpers shared by planners and MPCC controllers."""

from collections.abc import Mapping

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = [
    "build_cubic_spline_path_data",
    "clamp_progress",
    "closest_progress",
    "curvature",
    "evaluate",
    "evaluate_spline_path",
    "normalize_path_data",
    "progress_bounds",
    "project_progress",
    "slice_path_data",
    "slice_reference_path_data",
    "stage_parameter",
    "tangent",
]


def _sanitize_polyline(polyline, min_point_distance):
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("global_path must be Nx2 or Nx3 array.")

    points = points[:, :2]
    if points.shape[0] == 0:
        raise ValueError("global_path must contain at least one point.")

    min_distance = max(float(min_point_distance), 1e-8)
    filtered = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - filtered[-1]) >= min_distance:
            filtered.append(point)

    if len(filtered) == 1:
        filtered.append(filtered[0] + np.array([min_distance, 0.0]))
    return np.asarray(filtered, dtype=float)


def _fit_cubic_coefficients(points, s_breaks, boundary_condition):
    spline_x = CubicSpline(
        s_breaks,
        points[:, 0],
        bc_type=boundary_condition,
        extrapolate=True,
    )
    spline_y = CubicSpline(
        s_breaks,
        points[:, 1],
        bc_type=boundary_condition,
        extrapolate=True,
    )
    # SciPy uses descending powers; controllers use [a0, a1, a2, a3].
    segments_x = np.stack(
        [spline_x.c[3], spline_x.c[2], spline_x.c[1], spline_x.c[0]],
        axis=1,
    )
    segments_y = np.stack(
        [spline_y.c[3], spline_y.c[2], spline_y.c[1], spline_y.c[0]],
        axis=1,
    )
    return segments_x, segments_y


def build_cubic_spline_path_data(
    polyline,
    min_point_distance=1e-4,
    bc_type="natural",
):
    """Fit a cubic spline and return the canonical segment representation."""
    points = _sanitize_polyline(polyline, min_point_distance)
    segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    minimum = max(float(min_point_distance), 1e-8)
    segment_lengths = np.maximum(segment_lengths, minimum)
    s_breaks = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segments_x, segments_y = _fit_cubic_coefficients(
        points,
        s_breaks,
        bc_type,
    )
    return {
        "segments_x": np.asarray(segments_x, dtype=float),
        "segments_y": np.asarray(segments_y, dtype=float),
        "s_breaks": np.asarray(s_breaks, dtype=float),
        "n_segments": int(segments_x.shape[0]),
        "points": points,
    }


def evaluate_spline_path(path_data, s_samples):
    """Evaluate one or more progress samples on a canonical spline path."""
    scalar_input = np.isscalar(s_samples)
    samples = (
        np.asarray([s_samples], dtype=float)
        if scalar_input
        else np.asarray(s_samples, dtype=float).reshape(-1)
    )
    n_segments = int(path_data["n_segments"])
    if n_segments <= 0:
        result = np.zeros((samples.size, 2), dtype=float)
        return result[0] if scalar_input else result

    s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
    clamped = np.clip(samples, s_breaks[0], s_breaks[n_segments])
    indices = np.searchsorted(
        s_breaks[1 : n_segments + 1],
        clamped,
        side="right",
    )
    indices = np.clip(indices, 0, n_segments - 1).astype(int)
    tau = clamped - s_breaks[indices]
    coefficients_x = np.asarray(path_data["segments_x"], dtype=float)[indices]
    coefficients_y = np.asarray(path_data["segments_y"], dtype=float)[indices]
    powers = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3))
    result = np.column_stack(
        (
            np.einsum("ij,ij->i", coefficients_x, powers),
            np.einsum("ij,ij->i", coefficients_y, powers),
        )
    )
    return result[0] if scalar_input else result


def slice_path_data(path_data, s_start, window_length, max_segments=None):
    """Return a segment-aligned local window without changing global progress."""
    n_segments = int(path_data["n_segments"])
    if n_segments <= 0:
        return {
            "segments_x": np.zeros((0, 4), dtype=float),
            "segments_y": np.zeros((0, 4), dtype=float),
            "s_breaks": np.array([0.0]),
            "n_segments": 0,
            "slice_start_idx": 0,
            "slice_end_idx": 0,
            "slice_request_s": 0.0,
            "slice_window_end_s": 0.0,
            "slice_start_s": 0.0,
            "slice_end_s": 0.0,
        }

    s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
    s_min, s_max = float(s_breaks[0]), float(s_breaks[n_segments])
    start = float(np.clip(s_start, s_min, s_max))
    end = float(
        np.clip(start + max(float(window_length), 1e-6), s_min, s_max)
    )
    start_index = int(
        np.clip(np.searchsorted(s_breaks, start, side="right") - 1, 0, n_segments - 1)
    )
    end_index = int(
        np.clip(
            np.searchsorted(s_breaks, end, side="right") - 1,
            start_index,
            n_segments - 1,
        )
    )
    if max_segments is not None:
        end_index = min(end_index, start_index + max(int(max_segments), 1) - 1)

    segment_slice = slice(start_index, end_index + 1)
    local_breaks = s_breaks[start_index : end_index + 2]
    return {
        "segments_x": np.asarray(path_data["segments_x"][segment_slice], dtype=float),
        "segments_y": np.asarray(path_data["segments_y"][segment_slice], dtype=float),
        "s_breaks": local_breaks.copy(),
        "n_segments": int(end_index - start_index + 1),
        "slice_start_idx": start_index,
        "slice_end_idx": end_index,
        "slice_request_s": start,
        "slice_window_end_s": end,
        "slice_start_s": float(local_breaks[0]),
        "slice_end_s": float(local_breaks[-1]),
    }


# Compatibility name used by the trajectory generator.
slice_reference_path_data = slice_path_data


def normalize_path_data(reference_path_data):
    """Return the canonical cubic-spline path representation."""
    if isinstance(reference_path_data, np.ndarray):
        return build_cubic_spline_path_data(reference_path_data)
    if not isinstance(reference_path_data, Mapping):
        raise ValueError("reference_path_data must be ndarray or dict.")

    def pick(*keys):
        return next(
            (
                reference_path_data[key]
                for key in keys
                if reference_path_data.get(key) is not None
            ),
            None,
        )

    segments_x = pick("segments_x", "coeff_x", "cx")
    segments_y = pick("segments_y", "coeff_y", "cy")
    s_breaks = pick("s_breaks", "S", "s_nodes", "segment_bounds")
    if segments_x is None or segments_y is None or s_breaks is None:
        raise ValueError(
            "reference_path_data must contain spline coefficients and s-breaks."
        )

    segments_x = np.asarray(segments_x, dtype=float)
    segments_y = np.asarray(segments_y, dtype=float)
    s_breaks = np.asarray(s_breaks, dtype=float).reshape(-1)
    n_segments = int(reference_path_data.get("n_segments", segments_x.shape[0]))
    if (
        segments_x.ndim != 2
        or segments_x.shape != segments_y.shape
        or segments_x.shape[1] != 4
    ):
        raise ValueError("segments_x/segments_y must have shape [n_segments, 4].")
    if n_segments < 0 or n_segments > segments_x.shape[0]:
        raise ValueError("n_segments must match the available spline coefficients.")
    if s_breaks.size < n_segments + 1:
        raise ValueError("s_breaks must have length >= n_segments + 1.")
    return {
        "segments_x": segments_x,
        "segments_y": segments_y,
        "s_breaks": s_breaks,
        "n_segments": n_segments,
    }


def _segment(path_data, s_value):
    n_segments = int(path_data["n_segments"])
    if n_segments <= 0:
        return None

    s_breaks = np.asarray(path_data["s_breaks"], dtype=float)
    s_min, s_max = float(s_breaks[0]), float(s_breaks[n_segments])
    s_value = float(np.clip(s_value, s_min, s_max))
    segment_index = int(
        np.searchsorted(s_breaks[1 : n_segments + 1], s_value, side="right")
    )
    segment_index = int(np.clip(segment_index, 0, n_segments - 1))
    tau = s_value - float(s_breaks[segment_index])
    cx = np.asarray(path_data["segments_x"][segment_index], dtype=float)
    cy = np.asarray(path_data["segments_y"][segment_index], dtype=float)
    return s_min, s_max, s_value, tau, cx, cy


def evaluate(path_data, s_value):
    segment = _segment(path_data, s_value)
    if segment is None:
        return np.zeros(2, dtype=float)
    _, _, _, tau, cx, cy = segment
    powers = np.array([1.0, tau, tau * tau, tau * tau * tau])
    return np.array([cx @ powers, cy @ powers], dtype=float)


def tangent(path_data, s_value, tangent_regularizer=1e-6, fd_epsilon=1e-3):
    segment = _segment(path_data, s_value)
    if segment is None:
        return np.array([1.0, 0.0], dtype=float)

    s_min, s_max, s_value, tau, cx, cy = segment
    derivative = np.array(
        [
            cx[1] + 2.0 * cx[2] * tau + 3.0 * cx[3] * tau * tau,
            cy[1] + 2.0 * cy[2] * tau + 3.0 * cy[3] * tau * tau,
        ],
        dtype=float,
    )
    tangent_regularizer = max(float(tangent_regularizer), 1e-9)
    norm = float(np.linalg.norm(derivative))
    if norm <= tangent_regularizer:
        fd_epsilon = max(float(fd_epsilon), 1e-4)
        s_previous = float(np.clip(s_value - fd_epsilon, s_min, s_max))
        s_next = float(np.clip(s_value + fd_epsilon, s_min, s_max))
        if s_next > s_previous + 1e-10:
            derivative = evaluate(path_data, s_next) - evaluate(
                path_data,
                s_previous,
            )
            norm = float(np.linalg.norm(derivative))
    if norm <= tangent_regularizer:
        return np.array([1.0, 0.0], dtype=float)
    return derivative / (norm + tangent_regularizer)


def curvature(path_data, s_value):
    segment = _segment(path_data, s_value)
    if segment is None:
        return 0.0
    _, _, _, tau, cx, cy = segment
    dx = cx[1] + 2.0 * cx[2] * tau + 3.0 * cx[3] * tau * tau
    dy = cy[1] + 2.0 * cy[2] * tau + 3.0 * cy[3] * tau * tau
    ddx = 2.0 * cx[2] + 6.0 * cx[3] * tau
    ddy = 2.0 * cy[2] + 6.0 * cy[3] * tau
    speed_squared = dx * dx + dy * dy
    if speed_squared <= 1e-12:
        return 0.0
    return float((dx * ddy - dy * ddx) / speed_squared**1.5)


def stage_parameter(
    path_data,
    s_value,
    tangent_regularizer=1e-6,
    fd_epsilon=1e-3,
    *,
    include_curvature=True,
):
    """Pack the common MPCC path frame, optionally including curvature."""
    if path_data is None:
        values = [0.0, 0.0, 1.0, 0.0, float(s_value)]
        if include_curvature:
            values.append(0.0)
        return np.asarray(values, dtype=float)
    position = evaluate(path_data, s_value)
    direction = tangent(
        path_data,
        s_value,
        tangent_regularizer,
        fd_epsilon,
    )
    values = [*position, *direction, float(s_value)]
    if include_curvature:
        values.append(curvature(path_data, s_value))
    return np.asarray(values, dtype=float)


def clamp_progress(path_data, s_value):
    if path_data is None:
        return max(0.0, float(s_value))
    n_segments = int(path_data["n_segments"])
    return float(
        np.clip(
            s_value,
            path_data["s_breaks"][0],
            path_data["s_breaks"][n_segments],
        )
    )


def closest_progress(path_data, position_xy):
    if path_data is None or int(path_data["n_segments"]) <= 0:
        return 0.0

    position = np.asarray(position_xy, dtype=float).reshape(2)
    best_distance = np.inf
    best_s = float(path_data["s_breaks"][0])
    for index in range(int(path_data["n_segments"])):
        s_left = float(path_data["s_breaks"][index])
        s_right = float(path_data["s_breaks"][index + 1])
        segment_length = max(s_right - s_left, 1e-6)
        cx, cy = path_data["segments_x"][index], path_data["segments_y"][index]
        start = np.array([cx[0], cy[0]], dtype=float)
        tau = segment_length
        end = np.array(
            [
                cx[0] + cx[1] * tau + cx[2] * tau**2 + cx[3] * tau**3,
                cy[0] + cy[1] * tau + cy[2] * tau**2 + cy[3] * tau**3,
            ]
        )
        chord = end - start
        denominator = float(chord @ chord)
        fraction = 0.0 if denominator <= 1e-12 else float(
            np.clip((position - start) @ chord / denominator, 0.0, 1.0)
        )
        projection = start + fraction * chord
        distance = float(np.linalg.norm(position - projection))
        if distance < best_distance:
            best_distance = distance
            best_s = s_left + fraction * segment_length
    return clamp_progress(path_data, best_s)


def project_progress(
    path_data,
    position_xy,
    previous_s,
    line_search_window,
    line_search_samples,
):
    if path_data is None:
        return 0.0

    closest_s = closest_progress(path_data, position_xy)
    center = closest_s if previous_s is None else clamp_progress(path_data, previous_s)
    n_segments = int(path_data["n_segments"])
    s_min = float(path_data["s_breaks"][0])
    s_max = float(path_data["s_breaks"][n_segments])
    half_window = max(float(line_search_window), 1e-3)
    lower, upper = max(s_min, center - half_window), min(s_max, center + half_window)
    if upper <= lower:
        return center

    s_grid = np.linspace(lower, upper, max(int(line_search_samples), 5))
    distances = np.array(
        [np.linalg.norm(position_xy - evaluate(path_data, s)) for s in s_grid]
    )
    line_s = float(s_grid[int(np.argmin(distances))])
    if (
        np.linalg.norm(position_xy - evaluate(path_data, closest_s)) + 1e-6
        < np.linalg.norm(position_xy - evaluate(path_data, line_s))
    ):
        return closest_s
    return line_s


def progress_bounds(path_data, upper_guard):
    if path_data is None:
        return 0.0, float(upper_guard)
    n_segments = int(path_data["n_segments"])
    lower = float(path_data["s_breaks"][0])
    upper = min(float(path_data["s_breaks"][n_segments]), float(upper_guard))
    return lower, max(lower, upper)
