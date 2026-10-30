import numpy as np
from utils.spline_path import (
    build_cubic_spline_path_data,
    evaluate_spline_path,
    slice_reference_path_data,
)


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
