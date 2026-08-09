import numpy as np
from scipy.interpolate import CubicSpline


def _sanitize_polyline(polyline, min_point_distance):
    points = np.asarray(polyline, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("global_path must be Nx2 or Nx3 array.")

    points = points[:, :2]
    if points.shape[0] == 0:
        raise ValueError("global_path must contain at least one point.")

    min_dist = max(float(min_point_distance), 1e-8)
    filtered = [points[0]]
    for i in range(1, points.shape[0]):
        if np.linalg.norm(points[i] - filtered[-1]) >= min_dist:
            filtered.append(points[i])

    if len(filtered) == 1:
        filtered.append(filtered[0] + np.array([min_dist, 0.0], dtype=float))

    return np.asarray(filtered, dtype=float)


def _fit_cubic_coefficients_with_scipy(points, s_breaks, bc_type):
    spline_x = CubicSpline(s_breaks, points[:, 0], bc_type=bc_type, extrapolate=True)
    spline_y = CubicSpline(s_breaks, points[:, 1], bc_type=bc_type, extrapolate=True)

    # SciPy stores coefficients as:
    # c[0]*(s-s_i)^3 + c[1]*(s-s_i)^2 + c[2]*(s-s_i) + c[3]
    # Convert to [a0, a1, a2, a3] for a0 + a1*tau + a2*tau^2 + a3*tau^3.
    seg_x = np.stack([spline_x.c[3], spline_x.c[2], spline_x.c[1], spline_x.c[0]], axis=1)
    seg_y = np.stack([spline_y.c[3], spline_y.c[2], spline_y.c[1], spline_y.c[0]], axis=1)
    return seg_x, seg_y


def build_cubic_spline_path_data(polyline, min_point_distance=1e-4, bc_type="natural"):
    points = _sanitize_polyline(polyline, min_point_distance=min_point_distance)

    seg_vec = points[1:] - points[:-1]
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len = np.where(seg_len < max(float(min_point_distance), 1e-8), float(min_point_distance), seg_len)
    s_breaks = np.concatenate(([0.0], np.cumsum(seg_len)))

    seg_x, seg_y = _fit_cubic_coefficients_with_scipy(points, s_breaks, bc_type=bc_type)

    n_seg = int(seg_x.shape[0])
    return {
        "segments_x": np.asarray(seg_x, dtype=float),
        "segments_y": np.asarray(seg_y, dtype=float),
        "s_breaks": np.asarray(s_breaks, dtype=float),
        "n_segments": n_seg,
        "points": np.asarray(points, dtype=float),
    }


def evaluate_spline_path(reference_path_data, s_samples):
    scalar_input = np.isscalar(s_samples)
    s_arr = np.asarray([s_samples], dtype=float) if scalar_input else np.asarray(s_samples, dtype=float).reshape(-1)

    n_seg = int(reference_path_data["n_segments"])
    if n_seg <= 0:
        out = np.zeros((s_arr.shape[0], 2), dtype=float)
        return out[0] if scalar_input else out

    s_breaks = np.asarray(reference_path_data["s_breaks"], dtype=float)
    s_min = float(s_breaks[0])
    s_max = float(s_breaks[n_seg])
    s_clamped = np.clip(s_arr, s_min, s_max)

    seg_idx = np.searchsorted(s_breaks[1:n_seg + 1], s_clamped, side="right")
    seg_idx = np.clip(seg_idx, 0, n_seg - 1).astype(int)

    s_left = s_breaks[seg_idx]
    tau = s_clamped - s_left

    cx = np.asarray(reference_path_data["segments_x"], dtype=float)[seg_idx]
    cy = np.asarray(reference_path_data["segments_y"], dtype=float)[seg_idx]

    x = cx[:, 0] + cx[:, 1] * tau + cx[:, 2] * tau * tau + cx[:, 3] * tau * tau * tau
    y = cy[:, 0] + cy[:, 1] * tau + cy[:, 2] * tau * tau + cy[:, 3] * tau * tau * tau

    out = np.stack([x, y], axis=1)
    return out[0] if scalar_input else out


def slice_reference_path_data(reference_path_data, s_start, window_length, max_segments=None):
    n_seg = int(reference_path_data["n_segments"])
    if n_seg <= 0:
        return {
            "segments_x": np.zeros((0, 4), dtype=float),
            "segments_y": np.zeros((0, 4), dtype=float),
            "s_breaks": np.array([0.0], dtype=float),
            "n_segments": 0,
            "slice_start_idx": 0,
            "slice_end_idx": 0,
            "slice_request_s": 0.0,
            "slice_window_end_s": 0.0,
            "slice_start_s": 0.0,
            "slice_end_s": 0.0,
        }

    s_breaks = np.asarray(reference_path_data["s_breaks"], dtype=float)
    s_min = float(s_breaks[0])
    s_max = float(s_breaks[n_seg])
    s0 = float(np.clip(s_start, s_min, s_max))
    s1 = float(np.clip(s0 + max(float(window_length), 1e-6), s_min, s_max))

    start_idx = int(np.searchsorted(s_breaks, s0, side="right") - 1)
    start_idx = int(np.clip(start_idx, 0, n_seg - 1))
    end_idx = int(np.searchsorted(s_breaks, s1, side="right") - 1)
    end_idx = int(np.clip(end_idx, start_idx, n_seg - 1))

    if max_segments is not None:
        max_seg = max(int(max_segments), 1)
        end_idx = min(end_idx, start_idx + max_seg - 1)

    local_slice = slice(start_idx, end_idx + 1)
    local_breaks = s_breaks[start_idx:end_idx + 2]
    local_n_seg = int(end_idx - start_idx + 1)

    return {
        "segments_x": np.asarray(reference_path_data["segments_x"][local_slice], dtype=float),
        "segments_y": np.asarray(reference_path_data["segments_y"][local_slice], dtype=float),
        "s_breaks": np.asarray(local_breaks, dtype=float),
        "n_segments": local_n_seg,
        "slice_start_idx": start_idx,
        "slice_end_idx": end_idx,
        "slice_request_s": s0,
        "slice_window_end_s": s1,
        "slice_start_s": float(local_breaks[0]),
        "slice_end_s": float(local_breaks[-1]),
    }


class SplineReferenceGenerator:
    def __init__(
        self,
        reference_speed=0.6,
        num_horizon=20,
        local_path_timestep=0.1,
        window_length=None,
        max_segments=20,
        proj_dist_buffer=0.03,
        slice_lookback=None,
        sample_count=None,
        min_point_distance=1e-4,
        spline_bc_type="natural",
        projected_s_backtrack_tolerance=0.15,
    ):
        self._reference_speed = float(reference_speed)
        self._num_horizon = int(num_horizon)
        self._local_path_timestep = float(local_path_timestep)
        auto_window = self._reference_speed * self._local_path_timestep * (self._num_horizon + 5)
        self._window_length = float(auto_window if window_length is None else window_length)
        self._window_length = max(self._window_length, 1e-3)
        self._max_segments = int(max_segments)
        self._proj_dist_buffer = float(proj_dist_buffer)
        default_slice_lookback = self._proj_dist_buffer if slice_lookback is None else slice_lookback
        self._slice_lookback = max(float(default_slice_lookback), 0.0)
        self._sample_count = max(int(self._num_horizon if sample_count is None else sample_count), 2)
        self._min_point_distance = float(min_point_distance)
        self._spline_bc_type = spline_bc_type
        self._projected_s_backtrack_tolerance = max(float(projected_s_backtrack_tolerance), 0.0)

        self._global_path = None
        self._global_path_data = None
        self._prev_projected_s = 0.0
        self._local_trajectory = None
        self._local_reference = None
        self._last_debug_info = {}

    def _is_same_global_path(self, global_path):
        if self._global_path is None:
            return False
        points = np.asarray(global_path, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            return False
        points = points[:, :2]
        return self._global_path.shape == points.shape and np.array_equal(self._global_path, points)

    def _project_s_on_polyline(self, position_xy):
        if self._global_path_data is None:
            return 0.0

        points = self._global_path_data["points"]
        s_breaks = self._global_path_data["s_breaks"]
        n_seg = points.shape[0] - 1
        if n_seg <= 0:
            return 0.0

        pos = np.asarray(position_xy, dtype=float).reshape(2)
        best_dist = np.inf
        best_s = float(s_breaks[0])

        for i in range(n_seg):
            p0 = points[i]
            p1 = points[i + 1]
            vec = p1 - p0
            denom = float(np.dot(vec, vec))
            if denom <= 1e-12:
                t = 0.0
            else:
                t = float(np.clip(np.dot(pos - p0, vec) / denom, 0.0, 1.0))
            proj = p0 + t * vec
            dist = float(np.linalg.norm(pos - proj))
            if dist < best_dist:
                best_dist = dist
                best_s = float(s_breaks[i] + t * (s_breaks[i + 1] - s_breaks[i]))

        s_max = float(s_breaks[int(self._global_path_data["n_segments"])])
        return float(np.clip(best_s, s_breaks[0], s_max))

    def generate_trajectory(self, system, global_path):
        if not self._is_same_global_path(global_path):
            self._global_path = np.asarray(global_path, dtype=float)[:, :2].copy()
            self._global_path_data = build_cubic_spline_path_data(
                self._global_path,
                min_point_distance=self._min_point_distance,
                bc_type=self._spline_bc_type,
            )
            self._prev_projected_s = 0.0

        position_xy = np.asarray(system._state._x[:2], dtype=float)
        s_breaks = np.asarray(self._global_path_data["s_breaks"], dtype=float)
        s_min = float(s_breaks[0])
        s_max = float(s_breaks[int(self._global_path_data["n_segments"])])

        projected_s_raw = self._project_s_on_polyline(position_xy)
        projected_s_raw = max(
            projected_s_raw,
            self._prev_projected_s - self._projected_s_backtrack_tolerance,
        )
        projected_s = float(np.clip(projected_s_raw + self._proj_dist_buffer, s_min, s_max))
        slice_start_s = float(np.clip(projected_s - self._slice_lookback, s_min, s_max))
        self._prev_projected_s = projected_s

        local_reference = slice_reference_path_data(
            self._global_path_data,
            s_start=slice_start_s,
            window_length=self._window_length,
            max_segments=self._max_segments,
        )
        local_reference["planner_projected_s_raw"] = float(projected_s_raw)
        local_reference["planner_projected_s"] = projected_s
        local_reference["planner_slice_start_s"] = slice_start_s
        local_reference["planner_slice_lookback"] = float(self._slice_lookback)

        s0 = float(local_reference["s_breaks"][0])
        s1 = float(local_reference["s_breaks"][int(local_reference["n_segments"])])
        s_samples = np.linspace(s0, s1, self._sample_count)
        self._local_trajectory = evaluate_spline_path(local_reference, s_samples)
        self._local_reference = local_reference
        self._last_debug_info = {
            "planner_projected_s_raw": float(projected_s_raw),
            "planner_projected_s": projected_s,
            "planner_slice_start_s": slice_start_s,
            "slice_start_s": float(local_reference["slice_start_s"]),
            "slice_end_s": float(local_reference["slice_end_s"]),
            "slice_start_idx": int(local_reference["slice_start_idx"]),
            "slice_end_idx": int(local_reference["slice_end_idx"]),
            "slice_n_segments": int(local_reference["n_segments"]),
        }
        return local_reference

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)
