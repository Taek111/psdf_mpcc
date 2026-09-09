import copy
import math
import os
import csv
import json
import yaml
import signal
from pathlib import Path

try:
    from IPython.display import HTML
except ImportError:
    HTML = lambda x: x  # Fallback for environments without IPython

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib as mpl

mpl.rcParams['animation.html'] = 'jshtml'  # 'jshtml' or 'html5' for HTML5 video
import statistics as st
import numpy as np
import torch

from control.controller import BaseController

from models.geometry_utils import PolytopeRegion, RectangleRegion
from models.dd import (
    DifferentialDriveDynamics,
    DifferentialDriveRectangleGeometry,
    DifferentialDriveMultipleGeometry,
    DifferentialDrivePolygonGeometry,
    DifferentialDriveStates,
    DifferentialDriveSystem,
    LocalizationErrorModel,
    LocalizedDifferentialDriveSystem,
)
from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator,
    HybridAStarPathGenerator,
)
from planning.trajectory_generator.constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
)
from planning.trajectory_generator.spline_reference_generator import (
    SplineReferenceGenerator,
)
from sim.simulation import Robot, SingleAgentSimulation
from sim.start_perturbation import resolve_start_perturbation, sample_start_pose


class TrialFailureChecker:
    def __init__(self, movement_window_sec, movement_threshold, max_consecutive_solver_failures):
        self._movement_window_sec = float(movement_window_sec)
        self._movement_threshold = float(movement_threshold)
        self._max_consecutive_solver_failures = int(max_consecutive_solver_failures)
        self._recent_positions = []
        self._consecutive_solver_failures = 0

    def __call__(self, simulation):
        robot = simulation._robot
        controller = getattr(robot, "_controller", None)
        status_info = controller.get_last_solver_status_info() if controller is not None else {}
        solver_success = status_info.get("success", None)

        if solver_success is False:
            self._consecutive_solver_failures += 1
        elif solver_success is True:
            self._consecutive_solver_failures = 0

        if (
            self._max_consecutive_solver_failures > 0
            and self._consecutive_solver_failures >= self._max_consecutive_solver_failures
        ):
            raw_status = status_info.get("raw_status", "unknown")
            return {
                "reason": f"consecutive_solver_failures({self._consecutive_solver_failures}): {raw_status}",
            }

        if self._movement_window_sec <= 0.0 or self._movement_threshold < 0.0:
            return None

        current_time = float(robot._system._time)
        current_position = np.asarray(robot._get_navigation_state()[:2], dtype=float)
        self._recent_positions.append((current_time, current_position.copy()))

        while self._recent_positions and (current_time - self._recent_positions[0][0]) > self._movement_window_sec:
            self._recent_positions.pop(0)

        if len(self._recent_positions) < 2:
            return None

        window_duration = current_time - self._recent_positions[0][0]
        if window_duration + 1e-9 < self._movement_window_sec:
            return None

        displacement = float(np.linalg.norm(current_position - self._recent_positions[0][1]))
        if displacement <= self._movement_threshold:
            return {
                "reason": (
                    f"stuck_for_{self._movement_window_sec:.2f}s"
                    f"_disp_{displacement:.4f}m_le_{self._movement_threshold:.4f}m"
                ),
            }

        return None


class simulation_mpc:
    def __init__(self):
        self.sim = None
        self.robot = None
        self.initial_pose = None
        self.start_perturbation_info = None
        self.output_root_dir = ""
        self.last_run_outcome = None
        self.pose_sdf_data = []  # Store pose and SDF data for CSV logging
        self.detected_obstacles_logger = []  # Store detected obstacle caps for animation
        self.profile_heatmap_scales = {}

    def _get_output_dir(self, category, *subdirs):
        output_dir = os.path.join(self.output_root_dir, category, *subdirs) if self.output_root_dir else os.path.join(category, *subdirs)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _get_figure_output_base(self, *subdirs, figure_name):
        output_dir = self._get_output_dir("figures", *subdirs)
        return os.path.join(output_dir, figure_name)

    @staticmethod
    def _collect_optimizer_runtime_stats(optimizer):
        if optimizer is None or not hasattr(optimizer, "get_runtime_stats"):
            return {}

        try:
            runtime_stats = optimizer.get_runtime_stats()
        except Exception:
            return {}

        if not isinstance(runtime_stats, dict):
            return {}

        return dict(runtime_stats)

    @staticmethod
    def _parse_profile_heatmap_scale(metric_name, scale_value):
        if scale_value is None:
            return None

        if isinstance(scale_value, dict):
            vmin = scale_value.get("vmin", None)
            vmax = scale_value.get("vmax", None)
        elif isinstance(scale_value, (list, tuple)) and len(scale_value) == 2:
            vmin, vmax = scale_value
        else:
            raise ValueError(
                f"{metric_name}_range must be [min, max] "
                "or {vmin: ..., vmax: ...}."
            )

        if vmin is None or vmax is None:
            return None

        vmin = float(vmin)
        vmax = float(vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError(f"{metric_name}_range must use finite numeric bounds.")
        if vmin > vmax:
            raise ValueError(f"{metric_name}_range requires min <= max.")
        return (vmin, vmax)

    @classmethod
    def _resolve_profile_heatmap_scales(cls, *sources):
        scales = {}
        for source in sources:
            if not isinstance(source, dict):
                continue

            for metric_name in ("speed", "clearance", "risk"):
                parsed_scale = cls._parse_profile_heatmap_scale(
                    metric_name,
                    source.get(f"{metric_name}_range"),
                )
                if parsed_scale is None:
                    # Allow passing an already-resolved metric scale dict such as
                    # {"speed": (0.0, 0.3), "clearance": (-0.08, 0.08)}.
                    parsed_scale = cls._parse_profile_heatmap_scale(metric_name, source.get(metric_name))
                if parsed_scale is None:
                    scale_block = source.get("profile_heatmap_scales", {})
                    if isinstance(scale_block, dict):
                        parsed_scale = cls._parse_profile_heatmap_scale(metric_name, scale_block.get(metric_name))
                if parsed_scale is not None:
                    scales[metric_name] = parsed_scale
        return scales

    @staticmethod
    def _get_latest_local_trajectory(simulation):
        local_paths = getattr(getattr(simulation._robot, "_local_planner_logger", None), "_trajs", [])
        if not local_paths:
            return None

        latest_local_path = np.asarray(local_paths[-1], dtype=float)
        if latest_local_path.ndim != 2 or latest_local_path.shape[1] < 2:
            return None
        return latest_local_path[:, :2]

    @staticmethod
    def _apply_param_overrides(opt_param, override_config, block_name):
        if not override_config:
            return

        applied_items = []
        unknown_keys = []
        for key, value in override_config.items():
            if hasattr(opt_param, key):
                setattr(opt_param, key, value)
                applied_items.append((key, value))
            else:
                unknown_keys.append(key)

        if hasattr(opt_param, "chance_gamma") and "chance_epsilon" in override_config and "chance_gamma" not in override_config:
            opt_param.chance_gamma = float(st.NormalDist().inv_cdf(1.0 - float(opt_param.chance_epsilon)))
            applied_items.append(("chance_gamma", opt_param.chance_gamma))

        if applied_items:
            formatted_items = ", ".join(f"{key}={value}" for key, value in applied_items)
            print(f"{block_name} configured from config: {formatted_items}")

        if unknown_keys:
            print(f"Warning: ignoring unknown {block_name} config keys: {', '.join(unknown_keys)}")

    @staticmethod
    def _get_config_block(config, block_name):
        if not isinstance(config, dict):
            return {}
        block = config.get(block_name, {})
        return block if isinstance(block, dict) else {}

    def _resolve_output_root_dir(self, config, localization_error_enabled):
        configured_output_root = None
        if isinstance(config, dict):
            configured_output_root = config.get("output_root_dir", None)

        if isinstance(configured_output_root, str) and configured_output_root.strip():
            return configured_output_root
        if localization_error_enabled:
            return "w_error"
        return ""

    def _resolve_output_suffix(self, config, localization_error_enabled):
        suffix_parts = []
        if localization_error_enabled:
            suffix_parts.append("locerr")

        if isinstance(config, dict):
            configured_suffix = config.get("output_suffix", None)
            if isinstance(configured_suffix, str) and configured_suffix.strip():
                suffix_parts.append(configured_suffix.strip())

        return "_".join(suffix_parts) if suffix_parts else None

    def _build_trial_failure_checker(self, config):
        criteria = self._get_config_block(config, "trial_failure_criteria")
        if not criteria.get("enabled", False):
            return None

        movement_window_sec = float(criteria.get("movement_window_sec", 0.0))
        movement_threshold = float(criteria.get("movement_threshold", 0.0))
        max_consecutive_solver_failures = int(criteria.get("max_consecutive_solver_failures", 0))

        print(
            "Trial failure criteria enabled: "
            f"movement_window_sec={movement_window_sec}, "
            f"movement_threshold={movement_threshold}, "
            f"max_consecutive_solver_failures={max_consecutive_solver_failures}"
        )

        return TrialFailureChecker(
            movement_window_sec=movement_window_sec,
            movement_threshold=movement_threshold,
            max_consecutive_solver_failures=max_consecutive_solver_failures,
        )

    def _build_success_criteria(self, config):
        criteria = self._get_config_block(config, "success_criteria")
        position_tolerance = float(criteria.get("position_tolerance", 0.01))
        angle_tolerance = float(criteria.get("angle_tolerance", 0.01))

        if criteria:
            print(
                "Success criteria configured: "
                f"position_tolerance={position_tolerance}, "
                f"angle_tolerance={angle_tolerance}"
            )

        return {
            "position_tolerance": position_tolerance,
            "angle_tolerance": angle_tolerance,
        }

    @staticmethod
    def _coerce_config_vector(value, size, default, field_name):
        if value is None:
            return np.asarray(default, dtype=float)

        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size == 1 and size > 1:
            vector = np.repeat(vector.item(), size)

        if vector.size != size:
            raise ValueError(f"'{field_name}' must contain {size} values.")

        return vector

    @staticmethod
    def _coerce_angle_from_config(config_block, rad_key, deg_key):
        if rad_key in config_block:
            return float(config_block[rad_key])
        if deg_key in config_block:
            return math.radians(float(config_block[deg_key]))
        return 0.0

    def _build_localization_error_model(self, config):
        localization_config = self._get_config_block(config, "localization_error")
        if not localization_config.get("enabled", False):
            return None

        position_noise_frame = str(localization_config.get("position_noise_frame", "local")).lower()
        position_bias = self._coerce_config_vector(
            localization_config.get("position_bias"),
            size=2,
            default=[0.0, 0.0],
            field_name="position_bias",
        )

        if "front_lateral_noise_std" in localization_config:
            position_noise_std = self._coerce_config_vector(
                localization_config.get("front_lateral_noise_std"),
                size=2,
                default=[0.0, 0.0],
                field_name="front_lateral_noise_std",
            )
            position_noise_frame = "local"
        elif "front_noise_std" in localization_config or "lateral_noise_std" in localization_config:
            position_noise_std = np.asarray(
                [
                    float(localization_config.get("front_noise_std", 0.0)),
                    float(localization_config.get("lateral_noise_std", 0.0)),
                ],
                dtype=float,
            )
            position_noise_frame = "local"
        else:
            position_noise_std = self._coerce_config_vector(
                localization_config.get("position_noise_std"),
                size=2,
                default=[0.0, 0.0],
                field_name="position_noise_std",
            )

        heading_bias = self._coerce_angle_from_config(
            localization_config,
            rad_key="heading_bias_rad",
            deg_key="heading_bias_deg",
        )
        heading_noise_std = self._coerce_angle_from_config(
            localization_config,
            rad_key="heading_noise_std_rad",
            deg_key="heading_noise_std_deg",
        )
        seed = localization_config.get("seed")

        if position_noise_frame not in ("local", "world"):
            raise ValueError("localization_error.position_noise_frame must be 'local' or 'world'.")

        translation_label = "front_lateral" if position_noise_frame == "local" else "x_y"
        print(
            "Localization error enabled: "
            f"frame={position_noise_frame}, "
            f"bias_{translation_label}={position_bias.tolist()}, "
            f"bias_theta_deg={math.degrees(heading_bias):.3f}, "
            f"noise_{translation_label}_std={position_noise_std.tolist()}, "
            f"noise_theta_std_deg={math.degrees(heading_noise_std):.3f}, "
            f"seed={seed}"
        )

        return LocalizationErrorModel(
            position_bias=position_bias,
            heading_bias=heading_bias,
            position_noise_std=position_noise_std,
            heading_noise_std=heading_noise_std,
            position_noise_frame=position_noise_frame,
            seed=seed,
        )
        
    @staticmethod
    def _prepare_boole_risk_visual_data(risk_frame):
        """Build screen-space marker data for one nominal MF horizon."""

        empty = {
            "offsets": np.empty((0, 2), dtype=float),
            "sizes": np.empty((0,), dtype=float),
            "normalized_risk": np.empty((0,), dtype=float),
            "budget_exceeded": np.empty((0,), dtype=bool),
        }
        if not isinstance(risk_frame, dict):
            return empty

        try:
            poses = np.asarray(risk_frame["nominal_poses"], dtype=float)
            risk_sum = np.asarray(
                risk_frame["boole_risk_sum"], dtype=float
            ).reshape(-1)
            epsilon = np.asarray(risk_frame["epsilon"], dtype=float).reshape(-1)
            mf_mask = np.asarray(risk_frame["mf_mask"], dtype=bool).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return empty

        if poses.ndim != 2 or poses.shape[1] < 2:
            return empty

        num_stages = min(
            poses.shape[0],
            risk_sum.size,
            epsilon.size,
            mf_mask.size,
        )
        if num_stages <= 1:
            return empty

        stage_indices = np.arange(num_stages)
        risk_roundoff_tolerance = 1e-7
        valid = (
            (stage_indices > 0)
            & mf_mask[:num_stages]
            & np.all(np.isfinite(poses[:num_stages, :2]), axis=1)
            & np.isfinite(risk_sum[:num_stages])
            & (risk_sum[:num_stages] >= -risk_roundoff_tolerance)
            & np.isfinite(epsilon[:num_stages])
            & (epsilon[:num_stages] > 0.0)
        )
        if not np.any(valid):
            return empty

        # Roundoff in the float32 MF graph can produce tiny negative values.
        # Clamp only that numerical noise; materially negative sums are filtered
        # above instead of being rendered as reassuring zero-risk markers.
        displayed_risk = np.maximum(risk_sum[:num_stages][valid], 0.0)
        normalized_risk = displayed_risk / epsilon[:num_stages][valid]
        clipped_risk = np.clip(normalized_risk, 0.0, 1.0)

        # Matplotlib scatter sizes are screen-space areas in points squared.
        # Keep the glyph diameter between 4 and 10 points so a dimensionless
        # risk value is never mistaken for a distance in world coordinates.
        min_radius_pt = 2.0
        max_radius_pt = 5.0
        radius_pt = np.sqrt(
            min_radius_pt**2
            + (max_radius_pt**2 - min_radius_pt**2) * clipped_risk
        )

        return {
            "offsets": poses[:num_stages, :2][valid].copy(),
            "sizes": np.square(2.0 * radius_pt),
            "normalized_risk": clipped_risk,
            "budget_exceeded": normalized_risk >= 1.0,
        }

    @staticmethod
    def _resolve_use_risk_visualization(simulation, override=None):
        """Resolve an explicit override or the RMPCC optimizer parameter."""

        if override is not None:
            return bool(override)

        robot = getattr(simulation, "_robot", None)
        controller = getattr(robot, "_controller", None)
        opt_param = getattr(controller, "_param", None)
        if opt_param is None:
            optimizer = getattr(controller, "_optimizer", None)
            opt_param = getattr(optimizer, "param", None)
        return bool(getattr(opt_param, "use_risk_visualization", False))

    @staticmethod
    def _style_map_patch(patch):
        try:
            patch.set_facecolor("r")
        except AttributeError:
            pass
        try:
            patch.set_edgecolor("k")
        except AttributeError:
            pass
        try:
            patch.set_linewidth(1.0)
        except AttributeError:
            pass
        try:
            patch.set_alpha(1.0)
        except AttributeError:
            pass
        return patch

    @staticmethod
    def _style_robot_patch(patch, alpha):
        try:
            patch.set_facecolor("tab:brown")
        except AttributeError:
            pass
        try:
            patch.set_edgecolor("none")
        except AttributeError:
            pass
        try:
            patch.set_linewidth(0.5)
        except AttributeError:
            pass
        try:
            patch.set_alpha(float(alpha))
        except (AttributeError, TypeError, ValueError):
            pass
        return patch

    def plot_world(
        self,
        simulation,
        snapshot_indexes,
        figure_name="world",
        local_traj_indexes=[],
        maze_type=None,
    ):
        # TODO: make this plotting function general applicable to different systems
        if maze_type == "maze":
            fig, ax = plt.subplots(figsize=(8.3, 5.0))
        elif maze_type == "oblique_maze":
            fig, ax = plt.subplots(figsize=(6.7, 5.0))
        elif maze_type == "straight_corridor":
            fig, ax = plt.subplots(figsize=(9.0, 3.5))
        else:
            # 기본값 추가 (s_path나 다른 maze_type 처리)
            fig, ax = plt.subplots(figsize=(8.0, 6.0))

        global_paths = getattr(getattr(simulation._robot, "_global_planner_logger", None), "_paths", [])
        global_path = global_paths[0] if global_paths else None
        closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
        local_paths = getattr(getattr(simulation._robot, "_local_planner_logger", None), "_trajs", [])
        optimized_trajs = getattr(getattr(simulation._robot, "_controller_logger", None), "_xtrajs", [])

        # Use a denser default snapshot set while still including the final pose.
        if len(snapshot_indexes) == 0:
            traj_length = len(closedloop_traj)
            num_snapshots = 20
            snapshot_indexes = [
                int(i * (traj_length - 1) / (num_snapshots - 1))
                for i in range(num_snapshots - 1)
            ]
            snapshot_indexes.append(traj_length - 1)
            snapshot_indexes = list(dict.fromkeys(snapshot_indexes))

        # plot robot
        for index in snapshot_indexes:
            for i in range(simulation._robot._system._geometry._num_geometry):
                polygon_patch = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i, 0.6)
                polygon_patch = self._style_robot_patch(polygon_patch, alpha=0.6)
                ax.add_patch(polygon_patch)

        # plot global reference
        if global_path is not None:
            ax.plot(global_path[:, 0], global_path[:, 1], "o--", color="grey", linewidth=1.5, markersize=2)

        # plot closed loop trajectory
        ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "-", color="green", linewidth=1, markersize=4)

        # plot obstacles
        for obs in simulation._obstacles:
            obs_patch = self._style_map_patch(obs.get_plot_patch())
            ax.add_patch(obs_patch)

        # plot local reference and local optimized trajectories
        for index in local_traj_indexes:
            if index >= len(local_paths):
                continue
            local_path = np.asarray(local_paths[index], dtype=float)
            ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)

            if index >= len(optimized_trajs):
                continue
            optimized_traj = np.asarray(optimized_trajs[index], dtype=float)
            ax.plot(
                optimized_traj[:, 0],
                optimized_traj[:, 1],
                "-",
                color="gold",
                linewidth=3,
                markersize=4,
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        plt.tight_layout()

        # save figure
        output_base = self._get_figure_output_base("trajectory", figure_name=figure_name)
        plt.savefig(output_base + ".eps", format="eps", dpi=500, pad_inches=0)
        plt.savefig(output_base + ".png", format="png", dpi=500, pad_inches=0)
        plt.close(fig)

    def _create_metric_figure(self, maze_type=None):
        if maze_type == "maze":
            return plt.subplots(figsize=(8.3, 5.0))
        if maze_type == "oblique_maze":
            return plt.subplots(figsize=(6.7, 5.0))
        if maze_type == "straight_corridor":
            return plt.subplots(figsize=(9.0, 3.5))
        return plt.subplots(figsize=(8.0, 6.0))

    def _set_world_axes(self, fig, ax, simulation, path_xy, maze_type=None, extra_width_inches=0.0):
        local_path = self._get_latest_local_trajectory(simulation)

        x_sources = []
        y_sources = []
        if path_xy.size > 0:
            x_sources.append(path_xy[:, 0])
            y_sources.append(path_xy[:, 1])
        if local_path is not None:
            x_sources.append(local_path[:, 0])
            y_sources.append(local_path[:, 1])

        if maze_type is not None:
            _, _, grid, _ = self.create_env(maze_type)
            (x_min, y_min), (x_max, y_max) = grid[0]
            x_limits = (float(x_min), float(x_max))
            y_limits = (float(y_min), float(y_max))
        elif x_sources and y_sources:
            all_x = np.concatenate(x_sources)
            all_y = np.concatenate(y_sources)
            x_limits = (float(all_x.min()), float(all_x.max()))
            y_limits = (float(all_y.min()), float(all_y.max()))
        else:
            x_limits = (0.0, 1.0)
            y_limits = (0.0, 1.0)

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        if x_range <= 0.0:
            x_center = 0.5 * (x_limits[0] + x_limits[1])
            x_limits = (x_center - 0.5, x_center + 0.5)
            x_range = 1.0
        if y_range <= 0.0:
            y_center = 0.5 * (y_limits[0] + y_limits[1])
            y_limits = (y_center - 0.5, y_center + 0.5)
            y_range = 1.0

        _, fig_height = fig.get_size_inches()
        map_width_inches = fig_height * (x_range / y_range)
        extra_width_inches = max(float(extra_width_inches), 0.0)
        total_width_inches = map_width_inches + extra_width_inches
        fig.set_size_inches(total_width_inches, fig_height, forward=True)

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if total_width_inches > 0.0:
            ax.set_position([0.0, 0.0, map_width_inches / total_width_inches, 1.0])

        return {
            "map_width_inches": map_width_inches,
            "fig_height_inches": fig_height,
            "total_width_inches": total_width_inches,
        }

    @staticmethod
    def _build_metric_norm(metric_values, center=None, value_range=None):
        if value_range is None:
            vmin = float(np.min(metric_values))
            vmax = float(np.max(metric_values))
        else:
            vmin = float(value_range[0])
            vmax = float(value_range[1])
        if center is not None and vmin < center < vmax:
            return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        if np.isclose(vmin, vmax):
            eps = max(abs(vmin) * 1e-6, 1e-6)
            return mpl.colors.Normalize(vmin=vmin - eps, vmax=vmax + eps)
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    def _plot_metric_heatmap(
        self,
        simulation,
        path_xy,
        metric_values,
        figure_name,
        output_subdirs,
        output_suffix,
        title,
        colorbar_label,
        cmap,
        maze_type=None,
        center=None,
        value_range=None,
    ):
        path_xy = np.asarray(path_xy, dtype=float)
        metric_values = np.asarray(metric_values, dtype=float).reshape(-1)
        fig, ax = self._create_metric_figure(maze_type)

        colorbar_pad_inches = 0.20
        colorbar_width_inches = 0.34
        colorbar_margin_inches = 0.52
        layout_info = None
        if metric_values.size > 0:
            layout_info = self._set_world_axes(
                fig,
                ax,
                simulation,
                path_xy,
                maze_type=maze_type,
                extra_width_inches=colorbar_pad_inches + colorbar_width_inches + colorbar_margin_inches,
            )
        else:
            layout_info = self._set_world_axes(fig, ax, simulation, path_xy, maze_type=maze_type)

        for obs in simulation._obstacles:
            obs_patch = self._style_map_patch(obs.get_plot_patch())
            ax.add_patch(obs_patch)

        if path_xy.shape[0] > 0:
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.82", linewidth=1.5, zorder=2)
            ax.scatter(path_xy[0, 0], path_xy[0, 1], color="tab:blue", s=30, zorder=4)
            ax.scatter(path_xy[-1, 0], path_xy[-1, 1], color="k", s=30, marker="x", zorder=4)

        if path_xy.shape[0] >= 2 and metric_values.size > 0:
            segments = np.stack([path_xy[:-1], path_xy[1:]], axis=1)
            if metric_values.size == path_xy.shape[0]:
                segment_values = 0.5 * (metric_values[:-1] + metric_values[1:])
            elif metric_values.size == path_xy.shape[0] - 1:
                segment_values = metric_values
            else:
                usable_len = min(metric_values.size, path_xy.shape[0] - 1)
                segments = segments[:usable_len]
                segment_values = metric_values[:usable_len]

            finite_mask = np.isfinite(segment_values)
            segments = segments[finite_mask]
            segment_values = segment_values[finite_mask]

            if segment_values.size > 0:
                norm = self._build_metric_norm(segment_values, center=center, value_range=value_range)
                line_collection = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidth=4.0,
                    zorder=3,
                )
                line_collection.set_array(segment_values)
                line_collection.set_alpha(0.85)
                ax.add_collection(line_collection)

                segment_midpoints = 0.5 * (segments[:, 0, :] + segments[:, 1, :])
                ax.scatter(
                    segment_midpoints[:, 0],
                    segment_midpoints[:, 1],
                    c=segment_values,
                    cmap=cmap,
                    norm=norm,
                    s=72.0,
                    alpha=0.92,
                    edgecolors="none",
                    zorder=3.2,
                )

                cbar_left = (layout_info["map_width_inches"] + colorbar_pad_inches) / layout_info["total_width_inches"]
                cbar_width = colorbar_width_inches / layout_info["total_width_inches"]
                cax = fig.add_axes([cbar_left, 0.08, cbar_width, 0.84])
                colorbar = fig.colorbar(line_collection, cax=cax)
                colorbar.set_label(colorbar_label)
                colorbar.locator = mpl.ticker.MaxNLocator(nbins=5)
                colorbar.formatter = mpl.ticker.FormatStrFormatter("%.3f")
                colorbar.update_ticks()
                colorbar.ax.tick_params(
                    which="both",
                    labelsize=9,
                    colors="black",
                    length=3.5,
                    width=0.8,
                    direction="out",
                    pad=2,
                )
                colorbar.ax.yaxis.set_ticks_position("right")
                colorbar.ax.yaxis.set_label_position("right")
                for spine in colorbar.ax.spines.values():
                    spine.set_visible(True)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No finite metric data available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Not enough trajectory data for heatmap",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_axis_off()
        output_base = self._get_figure_output_base(*output_subdirs, figure_name=figure_name + output_suffix)
        fig.savefig(output_base + ".eps", format="eps", dpi=500)
        fig.savefig(output_base + ".png", format="png", dpi=500)
        plt.close(fig)

    def plot_profiles(
        self,
        simulation,
        figure_name="world",
        maze_type=None,
        include_risk_profile=False,
        metric_scales=None,
    ):
        robot = getattr(simulation, "_robot", None)
        if robot is None:
            raise ValueError("No robot data is available for profile plotting.")

        if metric_scales is None:
            metric_scales = dict(getattr(self, "profile_heatmap_scales", {}) or {})
        speed_heatmap_cmap = "inferno"
        clearance_heatmap_cmap = "RdYlGn"
        risk_heatmap_cmap = "RdYlGn_r"

        logged_inputs = getattr(getattr(robot, "_system_logger", None), "_us", [])
        logged_states = getattr(getattr(robot, "_system_logger", None), "_xs", [])

        linear_speed = np.array([], dtype=float)
        if logged_inputs:
            control_history = np.asarray(logged_inputs, dtype=float)
            if control_history.ndim == 1:
                control_history = control_history.reshape(-1, 1)
            linear_speed = control_history[:, 0]

        speed_path_xy = np.empty((0, 2), dtype=float)
        if logged_states:
            state_path_xy = np.asarray(logged_states, dtype=float)[:, :2]
            if self.initial_pose is not None:
                initial_xy = np.asarray(self.initial_pose[:2], dtype=float).reshape(1, 2)
                speed_path_xy = np.vstack([initial_xy, state_path_xy])
            else:
                speed_path_xy = state_path_xy

        self._plot_metric_heatmap(
            simulation=simulation,
            path_xy=speed_path_xy,
            metric_values=linear_speed,
            figure_name=figure_name,
            output_subdirs=("profile", "speed"),
            output_suffix="_speed_profile",
            title="Speed heatmap along trajectory",
            colorbar_label="speed [m/s]",
            cmap=speed_heatmap_cmap,
            maze_type=maze_type,
            value_range=metric_scales.get("speed"),
        )

        sdf_path_xy = np.empty((0, 2), dtype=float)
        sdf_values = np.array([], dtype=float)
        if self.pose_sdf_data:
            sdf_path_xy = np.asarray(
                [[entry.get("x", float("nan")), entry.get("y", float("nan"))] for entry in self.pose_sdf_data],
                dtype=float,
            )
            sdf_values = np.asarray(
                [entry.get("sdf_value", float("nan")) for entry in self.pose_sdf_data],
                dtype=float,
            )
            finite_pose_mask = np.all(np.isfinite(sdf_path_xy), axis=1)
            sdf_path_xy = sdf_path_xy[finite_pose_mask]
            sdf_values = sdf_values[finite_pose_mask]

        self._plot_metric_heatmap(
            simulation=simulation,
            path_xy=sdf_path_xy,
            metric_values=sdf_values,
            figure_name=figure_name,
            output_subdirs=("profile", "clearance"),
            output_suffix="_clearance_profile",
            title="Clearance (SDF) heatmap along trajectory",
            colorbar_label="clearance [m]",
            cmap=clearance_heatmap_cmap,
            maze_type=maze_type,
            center=0.0,
            value_range=metric_scales.get("clearance"),
        )

        if include_risk_profile and self.pose_sdf_data:
            risk_path_xy = np.asarray(
                [[entry.get("x", float("nan")), entry.get("y", float("nan"))] for entry in self.pose_sdf_data],
                dtype=float,
            )
            risk_metric_key = "risk_margin_max"
            if not any(
                entry.get("risk_margin_max", None) not in (None, "")
                for entry in self.pose_sdf_data
            ):
                risk_metric_key = "risk_margin_5"

            risk_values = np.asarray(
                [entry.get(risk_metric_key, float("nan")) for entry in self.pose_sdf_data],
                dtype=float,
            )
            finite_pose_mask = np.all(np.isfinite(risk_path_xy), axis=1)
            risk_path_xy = risk_path_xy[finite_pose_mask]
            risk_values = risk_values[finite_pose_mask]

            if np.any(np.isfinite(risk_values)):
                colorbar_label = "risk margin max [m]"
                if risk_metric_key == "risk_margin_5":
                    colorbar_label = "risk margin step 5 [m]"
                self._plot_metric_heatmap(
                    simulation=simulation,
                    path_xy=risk_path_xy,
                    metric_values=risk_values,
                    figure_name=figure_name,
                    output_subdirs=("profile", "risk"),
                    output_suffix="_risk_profile",
                    title="Risk heatmap along trajectory",
                    colorbar_label=colorbar_label,
                    cmap=risk_heatmap_cmap,
                    maze_type=maze_type,
                    center=0.0,
                    value_range=metric_scales.get("risk"),
                )

    def _normalize_animation_method(self, animation_name, method_name=None):
        if method_name:
            return method_name.lower()

        normalized_name = animation_name.lower()
        known_methods = (
            "dcbf_casadi",
            "obca",
            "minkowski_cbf",
            "psdf",
            "mpcc",
            "rmpcc_pv",
            "rmpcc",
            "dcbf",
            "sqp",
        )
        for known_method in known_methods:
            if f"mpc_{known_method}_" in normalized_name:
                return known_method

        return None

    def _get_animation_output_path(self, animation_name, method_name=None):
        normalized_method = self._normalize_animation_method(animation_name, method_name)
        output_dir = self._get_output_dir("animations")
        if normalized_method:
            output_dir = os.path.join(output_dir, normalized_method)
            os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{animation_name}.mp4")

    @staticmethod
    def build_output_name(robot_shape, maze_type, optimizer_type, interrupted=False, extra_suffix=None):
        prefix_map = {
            "acados": "MPC_SQP_",
            "obca": "MPC_OBCA_",
            "minkowski_cbf": "MPC_minkowski_cbf_",
            "psdf": "MPC_psdf_",
            "mpcc": "MPC_mpcc_",
            "rmpcc_pv": "MPC_rmpcc_pv_",
            "rmpcc": "MPC_rmpcc_",
            "dcbf": "MPC_DCBF_",
            "dcbf_casadi": "MPC_DCBF_CASADI_",
            "casadi": "MPC_",
        }
        name = prefix_map.get(optimizer_type, "MPC_") + f"{robot_shape}_{maze_type}"
        if extra_suffix:
            name += f"_{extra_suffix}"
        if interrupted:
            name += "_interrupted"
        return name

    @staticmethod
    def has_animation_data_for(simulation):
        if simulation is None:
            return False

        robot = getattr(simulation, "_robot", None)
        if robot is None:
            return False

        if len(getattr(getattr(robot, "_system_logger", None), "_xs", [])) > 0:
            return True

        return getattr(robot, "_system", None) is not None and hasattr(robot._system, "get_state")

    @staticmethod
    def is_interrupt_runtime_error(exc, interrupt_requested):
        if not interrupt_requested:
            return False

        message = str(exc)
        return (
            "KeyboardInterrupt" in message
            or "NonIpopt_Exception_Thrown" in message
            or "KeyboardInterruptException" in message
        )

    @staticmethod
    def _resolve_optimizer_variant(optimizer_type):
        return optimizer_type

    def animate_world(
        self,
        simulation,
        animation_name="world",
        maze_type=None,
        frame_skip=1,
        method_name=None,
        use_risk_visualization=None,
        return_html=True,
        include_initial_pose=False,
    ):
        use_risk_visualization = self._resolve_use_risk_visualization(
            simulation,
            use_risk_visualization,
        )
        robot = simulation._robot
        logged_states = getattr(getattr(robot, "_system_logger", None), "_xs", [])
        if logged_states:
            closedloop_traj = np.vstack(logged_states)
            if include_initial_pose and self.initial_pose is not None:
                closedloop_traj = np.vstack([self.initial_pose, closedloop_traj])
        elif getattr(robot, "_system", None) is not None:
            closedloop_traj = np.asarray(robot._system.get_state(), dtype=float).reshape(1, -1)
        else:
            raise ValueError("No robot trajectory data is available for animation.")

        local_paths = getattr(getattr(robot, "_local_planner_logger", None), "_trajs", [])
        optimized_trajs = getattr(getattr(robot, "_controller_logger", None), "_xtrajs", [])
        risk_margin_trajs = getattr(
            getattr(robot, "_controller_logger", None),
            "_risk_margin_trajs",
            [],
        )
        boole_risk_frames = getattr(
            getattr(robot, "_controller_logger", None),
            "_boole_risk_visualization_data",
            [],
        )
        logged_detected_caps = self.detected_obstacles_logger

        # Set figure size based on maze type
        if maze_type == "maze":
            fig, ax = plt.subplots(figsize=(8.3, 5.0))
        elif maze_type == "oblique_maze":
            fig, ax = plt.subplots(figsize=(6.7, 5.0))
        elif maze_type == "straight_corridor":
            fig, ax = plt.subplots(figsize=(9.0, 3.5))
        else:
            fig, ax = plt.subplots(figsize=(8.0, 6.0))

        # Plot static elements once
        global_paths = getattr(getattr(robot, "_global_planner_logger", None), "_paths", [])
        global_path = global_paths[0] if global_paths else None
        if global_path is not None:
            ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1.5, markersize=4, zorder=1)

        for obs in simulation._obstacles:
            obs_patch = self._style_map_patch(obs.get_plot_patch())
            ax.add_patch(obs_patch)

        # Initialize dynamic plot elements
        reference_traj_line, = ax.plot(
            [],
            [],
            "-",
            color="blue",
            linewidth=3,
            markersize=4,
            zorder=2,
        )
        if local_paths:
            local_path = local_paths[0]
            reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])

        optimized_traj_line, = ax.plot(
            [],
            [],
            "-",
            color="gold",
            linewidth=3,
            markersize=4,
            zorder=2,
        )
        if optimized_trajs:
            optimized_traj = optimized_trajs[0]
            optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])

        def build_risk_margin_patches(frame_idx):
            if (
                frame_idx >= len(optimized_trajs)
                or frame_idx >= len(risk_margin_trajs)
            ):
                return []

            optimized_traj = np.asarray(optimized_trajs[frame_idx], dtype=float)
            risk_margin_traj = risk_margin_trajs[frame_idx]
            if risk_margin_traj is None:
                return []

            risk_margin_traj = np.asarray(
                risk_margin_traj,
                dtype=float,
            ).reshape(-1)
            num_stages = min(len(optimized_traj), risk_margin_traj.size)
            patches_out = []
            for stage_idx in range(num_stages):
                radius = float(risk_margin_traj[stage_idx])
                if not np.isfinite(radius) or radius <= 0.0:
                    continue
                patch = patches.Circle(
                    (
                        float(optimized_traj[stage_idx, 0]),
                        float(optimized_traj[stage_idx, 1]),
                    ),
                    radius=radius,
                    facecolor="limegreen",
                    edgecolor="forestgreen",
                    alpha=0.22,
                    linewidth=1.0,
                    zorder=1.5,
                )
                ax.add_patch(patch)
                patches_out.append(patch)
            return patches_out

        risk_scatter = None
        has_boole_risk = use_risk_visualization and any(
            self._prepare_boole_risk_visual_data(risk_frame)[
                "normalized_risk"
            ].size > 0
            for risk_frame in boole_risk_frames
        )
        if has_boole_risk:
            risk_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
            risk_cmap = plt.get_cmap("YlOrRd")
            risk_scatter = ax.scatter(
                [],
                [],
                s=[],
                c=[],
                cmap=risk_cmap,
                norm=risk_norm,
                zorder=3.1,
            )
            risk_mappable = mpl.cm.ScalarMappable(norm=risk_norm, cmap=risk_cmap)
            risk_mappable.set_array([])
            risk_colorbar = fig.colorbar(
                risk_mappable,
                ax=ax,
                fraction=0.045,
                pad=0.025,
                extend="max",
            )
            risk_colorbar.set_ticks([0.0, 0.5, 1.0])
            risk_colorbar.set_ticklabels(["0", "0.5", "≥1"])
            risk_colorbar.set_label(
                r"Nominal Boole risk utilization $B_i / \varepsilon_i$ "
                r"(black ring: $\geq 1$)"
            )

        def update_risk_scatter(frame_idx):
            if risk_scatter is None:
                return

            risk_frame = (
                boole_risk_frames[frame_idx]
                if frame_idx < len(boole_risk_frames)
                else None
            )
            visual_data = self._prepare_boole_risk_visual_data(risk_frame)
            risk_scatter.set_offsets(visual_data["offsets"])
            risk_scatter.set_sizes(visual_data["sizes"])
            risk_scatter.set_array(visual_data["normalized_risk"])

            num_points = visual_data["normalized_risk"].size
            edge_colors = np.tile(
                mpl.colors.to_rgba("dimgray", alpha=0.55),
                (num_points, 1),
            )
            edge_widths = np.full((num_points,), 0.45, dtype=float)
            exceeded = visual_data["budget_exceeded"]
            edge_colors[exceeded] = mpl.colors.to_rgba("black", alpha=1.0)
            edge_widths[exceeded] = 1.2
            risk_scatter.set_edgecolors(edge_colors)
            risk_scatter.set_linewidths(edge_widths)

        risk_margin_patches = build_risk_margin_patches(0)
        update_risk_scatter(0)

        # Initialize robot patches
        robot_patches = []
        for i in range(robot._system._geometry._num_geometry):
            initial_patch = robot._system._geometry.get_plot_patch(closedloop_traj[0, :], i)
            initial_patch = self._style_robot_patch(initial_patch, alpha=0.5)
            robot_patches.append(initial_patch)
            ax.add_patch(robot_patches[i])

        # Initialize detected obstacle patches
        detected_obstacle_patches = []
        if logged_detected_caps and logged_detected_caps[0]:
            for cap_verts in logged_detected_caps[0]:
                if len(cap_verts) > 0:
                    patch = patches.Polygon(cap_verts, closed=True, facecolor='cyan', alpha=0.4, zorder=0)
                    ax.add_patch(patch)
                    detected_obstacle_patches.append(patch)

        # Match the legacy view window used in the old animation.
        x_sources = [closedloop_traj[:, 0]]
        y_sources = [closedloop_traj[:, 1]]
        if global_path is not None:
            x_sources.append(global_path[:, 0])
            y_sources.append(global_path[:, 1])

        all_x = np.concatenate(x_sources)
        all_y = np.concatenate(y_sources)
        margin = 0.1
        ax.set_xlim(float(all_x.min()) - margin, float(all_x.max()) + margin)
        ax.set_ylim(float(all_y.min()) - margin, float(all_y.max()) + margin)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        plt.tight_layout()

        final_traj_line = None

        def update(index):
            nonlocal final_traj_line

            # Update trajectory lines
            if index < len(local_paths):
                local_path = local_paths[index]
                reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
            if index < len(optimized_trajs):
                optimized_traj = optimized_trajs[index]
                optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])

            for patch in risk_margin_patches:
                patch.remove()
            risk_margin_patches.clear()
            risk_margin_patches.extend(build_risk_margin_patches(index))

            update_risk_scatter(index)

            # Efficiently update robot patches
            for i in range(robot._system._geometry._num_geometry):
                new_patch = robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i)
                new_patch = self._style_robot_patch(new_patch, alpha=0.5)
                old_patch = robot_patches[i]
                # Try to update patch in-place if possible
                if isinstance(old_patch, patches.Polygon) and isinstance(new_patch, patches.Polygon):
                    old_patch.set_xy(new_patch.get_xy())
                elif isinstance(old_patch, patches.Circle) and isinstance(new_patch, patches.Circle):
                    old_patch.center = new_patch.center
                    old_patch.radius = new_patch.radius
                else:
                    # fallback: replace patch
                    old_patch.remove()
                    robot_patches[i] = new_patch
                    ax.add_patch(robot_patches[i])

            # Update detected obstacle patches
            for patch in detected_obstacle_patches:
                patch.remove()
            detected_obstacle_patches.clear()

            if index < len(logged_detected_caps):
                current_caps = logged_detected_caps[index]
                if current_caps:
                    for cap_verts in current_caps:
                        if len(cap_verts) > 0:
                            patch = patches.Polygon(cap_verts, closed=True, facecolor='cyan', alpha=0.4, zorder=0)
                            ax.add_patch(patch)
                            detected_obstacle_patches.append(patch)

            # Add final trajectory only once at the end
            if index == len(closedloop_traj) - 1 and final_traj_line is None:
                final_traj_line, = ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1],
                                         "k-", linewidth=3, markersize=4, zorder=3)

            return (
                [reference_traj_line, optimized_traj_line]
                + risk_margin_patches
                + ([risk_scatter] if risk_scatter is not None else [])
                + robot_patches
                + detected_obstacle_patches
                + ([final_traj_line] if final_traj_line else [])
            )

        frames = list(range(0, len(closedloop_traj), max(1, frame_skip)))
        if frames[-1] != len(closedloop_traj) - 1:
            frames.append(len(closedloop_traj) - 1)
        anim = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True, repeat=False)

        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        output_path = self._get_animation_output_path(animation_name, method_name)
        anim.save(output_path, dpi=300, writer=writer)

        plt.close(fig)
        return HTML(anim.to_jshtml()) if return_html else output_path

    def log_pose_sdf_data(
        self,
        pose,
        sdf_value,
        timestep,
        risk_margin_stage0=None,
        risk_margin_5=None,
        risk_margin_max=None,
        path_s=None,
        predicted_s=None,
        solver_status=None,
        planner_projected_s_raw=None,
        planner_projected_s=None,
        planner_slice_start_s=None,
        planner_slice_lookback=None,
        slice_start_s=None,
        slice_end_s=None,
        slice_start_idx=None,
        slice_end_idx=None,
        slice_n_segments=None,
    ):
        """Log pose, SDF, MPCC progress, and local-slice diagnostics."""
        self.pose_sdf_data.append(
            {
                "timestep": timestep,
                "x": pose[0],
                "y": pose[1],
                "theta": pose[2],
                "sdf_value": sdf_value,
                "risk_margin_stage0": risk_margin_stage0,
                "risk_margin_5": risk_margin_5,
                "risk_margin_max": risk_margin_max,
                "path_s": path_s,
                "predicted_s": predicted_s,
                "solver_status": solver_status,
                "planner_projected_s_raw": planner_projected_s_raw,
                "planner_projected_s": planner_projected_s,
                "planner_slice_start_s": planner_slice_start_s,
                "planner_slice_lookback": planner_slice_lookback,
                "slice_start_s": slice_start_s,
                "slice_end_s": slice_end_s,
                "slice_start_idx": slice_start_idx,
                "slice_end_idx": slice_end_idx,
                "slice_n_segments": slice_n_segments,
            }
        )

    def save_pose_sdf_to_csv(self, filename):
        """Save collected pose and SDF data to CSV file"""
        if not self.pose_sdf_data:
            print("No pose-SDF data to save")
            return
            
        output_dir = self._get_output_dir("data")
        filepath = os.path.join(output_dir, f"{filename}.csv")
        
        # Save to CSV
        fieldnames = [
            "timestep",
            "x",
            "y",
            "theta",
            "sdf_value",
            "risk_margin_stage0",
            "risk_margin_5",
            "risk_margin_max",
            "path_s",
            "predicted_s",
            "solver_status",
            "planner_projected_s_raw",
            "planner_projected_s",
            "planner_slice_start_s",
            "planner_slice_lookback",
            "slice_start_s",
            "slice_end_s",
            "slice_start_idx",
            "slice_end_idx",
            "slice_n_segments",
        ]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.pose_sdf_data)
        print(f"Pose-SDF data saved to {filepath}")
        print(f"Total data points: {len(self.pose_sdf_data)}")
        
        # Print some statistics
        if len(self.pose_sdf_data) > 0:
            sdf_values = np.array([entry["sdf_value"] for entry in self.pose_sdf_data], dtype=float)
            print(f"SDF statistics - Min: {sdf_values.min():.4f}, Max: {sdf_values.max():.4f}, Mean: {sdf_values.mean():.4f}")

    def save_trial_history_to_csv(self, filename):
        if self.sim is None or getattr(self.sim, "_robot", None) is None:
            print("No simulation data to save")
            return

        robot = self.sim._robot
        logged_states = list(getattr(getattr(robot, "_system_logger", None), "_xs", []))
        logged_inputs = list(getattr(getattr(robot, "_system_logger", None), "_us", []))
        controller_logger = getattr(robot, "_controller_logger", None)
        solver_status_infos = list(getattr(controller_logger, "_solver_status_infos", []))
        controller_times = list(getattr(controller_logger, "_computation_times", []))
        optimizer = getattr(getattr(robot, "_controller", None), "_optimizer", None)
        solver_times = list(getattr(optimizer, "solver_times", []))

        if not logged_states:
            print("No trial state history to save")
            return

        dt = float(getattr(robot._system, "_dt", 0.1))
        output_dir = self._get_output_dir("data")
        filepath = os.path.join(output_dir, f"{filename}.csv")

        fieldnames = [
            "timestep",
            "time",
            "x",
            "y",
            "theta",
            "v",
            "omega",
            "controller_computation_time",
            "solver_computation_time",
            "solver_raw_status",
            "solver_status_code",
            "solver_success",
        ]

        rows = []
        for idx, state in enumerate(logged_states):
            state_arr = np.asarray(state, dtype=float).reshape(-1)
            input_arr = np.asarray(logged_inputs[idx], dtype=float).reshape(-1) if idx < len(logged_inputs) else np.array([])
            status_info = solver_status_infos[idx] if idx < len(solver_status_infos) else {}
            rows.append(
                {
                    "timestep": idx,
                    "time": float((idx + 1) * dt),
                    "x": float(state_arr[0]) if state_arr.size > 0 else float("nan"),
                    "y": float(state_arr[1]) if state_arr.size > 1 else float("nan"),
                    "theta": float(state_arr[2]) if state_arr.size > 2 else float("nan"),
                    "v": float(input_arr[0]) if input_arr.size > 0 else float("nan"),
                    "omega": float(input_arr[1]) if input_arr.size > 1 else float("nan"),
                    "controller_computation_time": (
                        float(controller_times[idx])
                        if idx < len(controller_times)
                        else ""
                    ),
                    "solver_computation_time": (
                        float(solver_times[idx]) if idx < len(solver_times) else ""
                    ),
                    "solver_raw_status": status_info.get("raw_status", None),
                    "solver_status_code": status_info.get("status_code", None),
                    "solver_success": status_info.get("success", None),
                }
            )

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Trial history saved to {filepath}")

    def save_rmpcc_constraint_history_to_csv(self, filename):
        """Save only the values needed to assess Row G/Row M feasibility."""
        controller = getattr(self.robot, "_controller", None)
        optimizer = getattr(controller, "_optimizer", None)
        if optimizer is None or not hasattr(optimizer, "get_constraint_log_rows"):
            return

        rows = optimizer.get_constraint_log_rows()
        if not rows:
            return

        fieldnames = optimizer.get_constraint_log_fields()
        output_dir = self._get_output_dir("data")
        filepath = os.path.join(output_dir, f"{filename}.csv")
        with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"RMPCC constraint history saved to {filepath}")

    def save_rmpcc_debug_history_to_csv(self):
        """Save the compact tuning records in the current run directory."""
        controller = getattr(self.robot, "_controller", None)
        optimizer = getattr(controller, "_optimizer", None)
        if optimizer is None:
            return

        output_dir = self._get_output_dir("data")
        exports = (
            (
                "cycle_raw.csv",
                "get_cycle_log_rows",
                "get_cycle_log_fields",
            ),
            (
                "covariance_trajectory.csv",
                "get_covariance_log_rows",
                "get_covariance_log_fields",
            ),
        )
        for filename, rows_method, fields_method in exports:
            if not hasattr(optimizer, rows_method) or not hasattr(
                optimizer, fields_method
            ):
                continue
            rows = getattr(optimizer, rows_method)()
            if not rows:
                continue
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=getattr(optimizer, fields_method)(),
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"RMPCC debug history saved to {filepath}")

    def mpc_test(self, maze_type, robot_shape, optimizer_type="casadi", dynamics_type="differential_drive", path_planner="astar", simulation_time=20.0, config=None):
        """
        Run MPC test with specified optimizer and dynamics.
        
        Args:
            maze_type: Type of maze environment ("s_path", "straight_corridor", "maze", "oblique_maze", "parking")
            robot_shape: Shape of robot ("rectangle", "pentagon", "triangle", "lshape")
            optimizer_type: Type of optimizer ("casadi", "acados", "obca",
                "minkowski_cbf", "psdf", "mpcc", "rmpcc", "rmpcc_pv",
                "dcbf", "dcbf_casadi")
            dynamics_type: Type of dynamics ("differential_drive")
            path_planner: Type of path planner.
            simulation_time: Duration of the simulation.
        """
        requested_optimizer_type = optimizer_type
        optimizer_type = self._resolve_optimizer_variant(optimizer_type)

        self.profile_heatmap_scales = self._resolve_profile_heatmap_scales(
            config.get("defaults", {}) if isinstance(config, dict) else {},
            config,
            self.profile_heatmap_scales,
        )

        # Clear previous data
        self.initial_pose = None
        self.start_perturbation_info = None
        self.output_root_dir = ""
        self.last_run_outcome = None
        self.pose_sdf_data = []
        self.detected_obstacles_logger = []
        
        start_pos, goal_pos, grid, obstacles = self.create_env(maze_type)
        geometry_regions = DifferentialDriveMultipleGeometry()
        localization_error_model = self._build_localization_error_model(config)
        self.output_root_dir = self._resolve_output_root_dir(config, localization_error_model is not None)
        output_suffix = self._resolve_output_suffix(config, localization_error_model is not None)
        self.current_name = self.build_output_name(
            robot_shape,
            maze_type,
            requested_optimizer_type,
            extra_suffix=output_suffix,
        )

        robot_indexes = [0, 17, 24, 33, 39, 48, 58, 66, 76, 86, 92, 102, 112, 129]
        traj_indexes = [0, 24, 33, 48, 66, 92, 112]
        
        if robot_shape == "rectangle":
            # vehicle_length = 0.15
            # vehicle_width = 0.06
            vehicle_length = 0.15
            vehicle_width = 0.09 #0.09
            geometry_regions.add_geometry(
                DifferentialDriveRectangleGeometry(length=vehicle_length, width=vehicle_width, rear_dist=0.0)
            )

        elif robot_shape == "pentagon":
            s = 0.05
            geometry_regions.add_geometry(DifferentialDrivePolygonGeometry(np.array([
            [s * np.cos(-np.pi/2),          s * np.sin(-np.pi/2)],            # k=0
            [s * np.cos( 2*np.pi/5 - np.pi/2), s * np.sin( 2*np.pi/5 - np.pi/2)],  # k=1
            [s * np.cos( 4*np.pi/5 - np.pi/2), s * np.sin( 4*np.pi/5 - np.pi/2)],  # k=2
            [s * np.cos( 6*np.pi/5 - np.pi/2), s * np.sin( 6*np.pi/5 - np.pi/2)],  # k=3
            [s * np.cos( 8*np.pi/5 - np.pi/2), s * np.sin( 8*np.pi/5 - np.pi/2)],  # k=4
            ])))
        
        elif robot_shape == "triangle":
            geometry_regions.add_geometry(DifferentialDrivePolygonGeometry(np.array([[0.10,0.00], [-0.05,0.05], [-0.05,-0.05]])))
           
        elif robot_shape == "lshape":
            geometry_regions.add_geometry(
                DifferentialDriveRectangleGeometry(0.4 * np.array([[0, 0.1], [0.02, 0.08], [-0.2, -0.1], [-0.22, -0.08]]))
            )
   

        # Perturb the true pose before constructing the plant or planning a path.
        start_pos, self.start_perturbation_info = sample_start_pose(
            start_pos, geometry_regions, obstacles, grid[0],
            (config or {}).get("start_perturbation"),
        )
        self.initial_pose = np.asarray(start_pos, dtype=float).copy()
        if self.start_perturbation_info is not None:
            info = self.start_perturbation_info
            self.current_name += f"_startseed{info['seed']}"
            metadata_path = os.path.join(
                self._get_output_dir("data"), f"start_pose_{self.current_name}.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump({
                    "maze_type": maze_type,
                    "robot_shape": robot_shape,
                    "optimizer_type": requested_optimizer_type,
                    **info,
                }, metadata_file, indent=2, allow_nan=False)
            print(
                f"Start perturbation: seed={info['seed']}, "
                f"pose={info['initial_pose']}, delta={info['delta_pose']}, "
                f"clearance={info['initial_clearance']:.6f} m, "
                f"attempts={info['attempts']}\nStart pose saved to {metadata_path}"
            )

        # Create dynamics based on dynamics_type
        if dynamics_type == "differential_drive":
            dynamics = DifferentialDriveDynamics()
        else:
            # Add other dynamics types here as needed
            dynamics = DifferentialDriveDynamics()  # Default fallback

        system_state = DifferentialDriveStates(x=np.block(start_pos))
        if localization_error_model is not None:
            robot_system = LocalizedDifferentialDriveSystem(
                state=system_state,
                geometry=geometry_regions,
                dynamics=dynamics,
                localization_error_model=localization_error_model,
            )
        else:
            robot_system = DifferentialDriveSystem(
                state=system_state,
                geometry=geometry_regions,
                dynamics=dynamics,
            )

        self.robot = Robot(
            robot_system
        )
        
        # 환경별 적절한 마진 설정
        if maze_type == "parking":
            global_path_margin = 0.02  # parking 환경은 더 작은 마진 사용
        elif maze_type in ("maze", "oblique_maze", "straight_corridor"):
            global_path_margin = 0.03  # maze/corridor 환경은 중간 마진 사용
        else:
            global_path_margin = 0.05  # s_path 등 기본 환경
            
        # Set global planner based on path_planner type
        if path_planner == "hybrid_astar":
            # Hybrid A* uses the vehicle dimensions defined above
            print(f"Using Hybrid A* with vehicle dimensions: L={vehicle_length}, W={vehicle_width}")
            self.robot.set_global_planner(HybridAStarPathGenerator(
                grid=grid, 
                margin=global_path_margin,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
                xy_resolution=0.05,
                theta_resolution=0.15
            ))
        else:  # Default to astar
            self.robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
        self.robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
        
        # Set up optimizer and controller based on optimizer_type
        if optimizer_type == "acados":
            from control.nmpc_optimizer_sqp import NmpcOptimizerSqp, NmpcOptimizerSqpParam

            opt_param = NmpcOptimizerSqpParam()
            optimizer = NmpcOptimizerSqp()
            optimizer_name = "acados"
        elif optimizer_type == "obca":
            from control.obca_optimizer import OBCAOptimizer, OBCAOptimizerParam

            opt_param = OBCAOptimizerParam()
            self._apply_param_overrides(
                opt_param, self._get_config_block(config, "obca"), "OBCA",
            )
            optimizer = OBCAOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
            optimizer_name = "obca"
        elif optimizer_type == "psdf":
            from control.psdf_optimizer import PSDFOptimizer, PSDFOptimizerParam

            opt_param = PSDFOptimizerParam()
            
            # Configure PSDF detection parameters from config if available
            if config and 'psdf_detection' in config:
                psdf_config = config['psdf_detection']
                opt_param.detection_window_width = psdf_config.get('window_width', opt_param.detection_window_width)
                opt_param.detection_window_height = psdf_config.get('window_height', opt_param.detection_window_height)
                opt_param.detection_safety_margin = psdf_config.get('safety_margin', opt_param.detection_safety_margin)
                opt_param.detection_frequency = psdf_config.get('detection_frequency', opt_param.detection_frequency)
                print(f"PSDF detection configured from config: "
                      f"window={opt_param.detection_window_width}x{opt_param.detection_window_height}m, "
                      f"margin={opt_param.detection_safety_margin}m, freq={opt_param.detection_frequency}Hz")
            
            optimizer = PSDFOptimizer()
            optimizer_name = "psdf"
        elif optimizer_type == "mpcc":
            from control.mpcc_optimizer import MPCCOptimizer, MPCCOptimizerParam

            opt_param = MPCCOptimizerParam()

            # Reuse psdf_detection config block for MPCC's PSDF update settings.
            if config and 'psdf_detection' in config:
                psdf_config = config['psdf_detection']
                opt_param.detection_window_width = psdf_config.get('window_width', opt_param.detection_window_width)
                opt_param.detection_window_height = psdf_config.get('window_height', opt_param.detection_window_height)
                opt_param.detection_safety_margin = psdf_config.get('safety_margin', opt_param.detection_safety_margin)
                opt_param.detection_frequency = psdf_config.get('detection_frequency', opt_param.detection_frequency)
                print(f"MPCC detection configured from config: "
                      f"window={opt_param.detection_window_width}x{opt_param.detection_window_height}m, "
                      f"margin={opt_param.detection_safety_margin}m, freq={opt_param.detection_frequency}Hz")

            dt = float(opt_param.tf) / float(opt_param.horizon)
            local_window_length = max(float(opt_param.v_s_max) * float(opt_param.tf) * 1.25, 0.8)
            self.robot.set_local_planner(
                SplineReferenceGenerator(
                    reference_speed=max(0.25, 0.7 * float(opt_param.v_s_max)),
                    num_horizon=opt_param.horizon,
                    local_path_timestep=dt,
                    window_length=local_window_length,
                    max_segments=opt_param.max_segments,
                    proj_dist_buffer=0.03,
                    sample_count=opt_param.horizon,
                    min_point_distance=1e-4,
                    spline_bc_type="natural",
                    projected_s_backtrack_tolerance=getattr(
                        opt_param,
                        "planner_projected_s_backtrack_tolerance",
                        0.15,
                    ),
                )
            )

            optimizer = MPCCOptimizer()
            optimizer_name = "mpcc"
        elif optimizer_type == "rmpcc":
            from control.rmpcc_optimizer import RMPCCOptimizer, RMPCCOptimizerParam

            opt_param = RMPCCOptimizerParam()

            if config and 'rmpcc' in config:
                self._apply_param_overrides(opt_param, config['rmpcc'], "RMPCC")

            # Keep legacy configs loadable even though RMPCC does not use
            # local-window detection parameters.
            if config and 'psdf_detection' in config:
                print(
                    "RMPCC ignores legacy 'psdf_detection' settings; "
                    "detection parameters are unused."
                )

            dt = float(opt_param.tf) / float(opt_param.horizon)
            local_window_length = max(float(opt_param.v_s_max) * float(opt_param.tf) * 1.25, 0.8)
            self.robot.set_local_planner(
                SplineReferenceGenerator(
                    reference_speed=max(0.25, 0.7 * float(opt_param.v_s_max)),
                    num_horizon=opt_param.horizon,
                    local_path_timestep=dt,
                    window_length=local_window_length,
                    max_segments=opt_param.max_segments,
                    proj_dist_buffer=0.03,
                    sample_count=opt_param.horizon,
                    min_point_distance=1e-4,
                    spline_bc_type="natural",
                    projected_s_backtrack_tolerance=getattr(
                        opt_param,
                        "planner_projected_s_backtrack_tolerance",
                        0.15,
                    ),
                )
            )

            optimizer = RMPCCOptimizer()
            optimizer_name = "rmpcc"
        elif optimizer_type == "rmpcc_pv":
            from control.rmpcc_pv_optimizer import (
                RMPCCPVOptimizer,
                RMPCCPVOptimizerParam,
            )

            opt_param = RMPCCPVOptimizerParam()
            pv_config = {}
            if config:
                shared_rmpcc_config = dict(config.get("rmpcc", {}))
                shared_rmpcc_config.pop("use_risk_visualization", None)
                pv_config.update(shared_rmpcc_config)
                pv_config.update(config.get("rmpcc_pv", {}))
            if pv_config:
                self._apply_param_overrides(opt_param, pv_config, "RMPCC-PV")

            if config and "psdf_detection" in config:
                print(
                    "RMPCC-PV ignores legacy 'psdf_detection' settings; "
                    "detection parameters are unused."
                )

            dt = float(opt_param.tf) / float(opt_param.horizon)
            local_window_length = max(
                float(opt_param.v_s_max) * float(opt_param.tf) * 1.25,
                0.8,
            )
            self.robot.set_local_planner(
                SplineReferenceGenerator(
                    reference_speed=max(0.25, 0.7 * float(opt_param.v_s_max)),
                    num_horizon=opt_param.horizon,
                    local_path_timestep=dt,
                    window_length=local_window_length,
                    max_segments=opt_param.max_segments,
                    proj_dist_buffer=0.03,
                    sample_count=opt_param.horizon,
                    min_point_distance=1e-4,
                    spline_bc_type="natural",
                    projected_s_backtrack_tolerance=getattr(
                        opt_param,
                        "planner_projected_s_backtrack_tolerance",
                        0.15,
                    ),
                )
            )

            optimizer = RMPCCPVOptimizer()
            optimizer_name = "rmpcc_pv"
        elif optimizer_type == "minkowski_cbf":
            from control.minkowski_cbf_optimizer import (
                MinkowskiCBFOptimizer,
                MinkowskiCBFOptimizerParam,
            )

            opt_param = MinkowskiCBFOptimizerParam()
            self._apply_param_overrides(
                opt_param,
                self._get_config_block(config, "minkowski_cbf"),
                "Minkowski CBF",
            )
            optimizer = MinkowskiCBFOptimizer()
            optimizer_name = "minkowski_cbf"
        elif optimizer_type in ("dcbf", "dcbf_casadi"):
            from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam

            opt_param = NmpcDcbfOptimizerParam()
            optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
            optimizer_name = optimizer_type
        else:  # Default to casadi
            from control.nmpc_optimizer import NmpcOptimizer, NmpcOptimizerParam

            opt_param = NmpcOptimizerParam()
            optimizer = NmpcOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
            optimizer_name = "CasADi"
        
        # Ensure optimizer starts with clean state
        if hasattr(optimizer, 'reset'):
            print(f"Resetting {optimizer_name} optimizer for clean state...")
            optimizer.reset()
        
        controller = BaseController(optimizer, opt_param)
        
        self.robot.set_controller(controller)
        
        ## Run simulation
        failure_checker = self._build_trial_failure_checker(config)
        success_criteria = self._build_success_criteria(config)
        self.sim = SingleAgentSimulation(
            self.robot,
            obstacles,
            goal_pos,
            goal_position_tolerance=success_criteria["position_tolerance"],
            goal_angle_tolerance=success_criteria["angle_tolerance"],
            failure_checker=failure_checker,
        )
        
        # If using PSDF-family optimizer, set up data logging and obstacle update hooks
        if optimizer_type in ("psdf", "mpcc", "rmpcc", "rmpcc_pv", "dcbf", "dcbf_casadi"):
            # Monkey patch the controller to log pose-SDF data and perform obstacle detection
            original_generate_control = controller.generate_control_input
            
            def enhanced_generate_control(system, global_path, local_trajectory, obstacles):
                optimizer_obj = controller._optimizer

                # Perform obstacle detection if available
                detected_caps = []
                current_pose = system._state._x
                if hasattr(optimizer_obj, "update_obstacles_with_detection"):
                    if (hasattr(optimizer_obj, "obstacle_detector")
                        and optimizer_obj.obstacle_detector is not None
                        and hasattr(optimizer_obj, "should_update_detection")
                        and optimizer_obj.should_update_detection()):
                        optimizer_obj.update_obstacles_with_detection(obstacles, current_pose, force_update=False)
                    else:
                        optimizer_obj.update_obstacles_with_detection(obstacles, current_pose, force_update=True)

                elif hasattr(optimizer_obj, "update_obstacles"):
                    optimizer_obj.update_obstacles(obstacles)

                if (hasattr(optimizer_obj, 'obstacle_detector') and 
                    optimizer_obj.obstacle_detector is not None):
                    current_pose = system._state._x

                    # Log detected caps for visualization
                    if hasattr(optimizer_obj.obstacle_detector, 'last_detected_caps_world'):
                        detected_caps = optimizer_obj.obstacle_detector.last_detected_caps_world
                
                # Always log to keep lists synchronized
                self.detected_obstacles_logger.append(detected_caps)

                # Call original method
                u_opt = original_generate_control(system, global_path, local_trajectory, obstacles)
                
                # Collect values for CSV logging.
                sdf_value = float("nan")
                if hasattr(optimizer_obj, "psdf_wrapper") and optimizer_obj.psdf_wrapper is not None:
                    pose_tensor = torch.tensor(current_pose, dtype=torch.float32, device=optimizer_obj.device).unsqueeze(0)
                    with torch.no_grad():
                        psdf_output = optimizer_obj.psdf_wrapper(pose_tensor)
                        phi_tensor = psdf_output[0] if isinstance(psdf_output, tuple) else psdf_output
                        sdf_value = phi_tensor.item()

                path_s = None
                predicted_s = None
                if hasattr(optimizer_obj, "get_progress_info"):
                    progress_info = optimizer_obj.get_progress_info()
                    path_s = progress_info.get("path_s", None)
                    predicted_s = progress_info.get("predicted_s", None)

                solver_status = None
                if hasattr(optimizer_obj, "solver") and optimizer_obj.solver is not None:
                    solver_status = int(optimizer_obj.solver.status)

                planner_projected_s_raw = None
                planner_projected_s = None
                planner_slice_start_s = None
                planner_slice_lookback = None
                slice_start_s = None
                slice_end_s = None
                slice_start_idx = None
                slice_end_idx = None
                slice_n_segments = None
                if isinstance(local_trajectory, dict):
                    planner_projected_s_raw = local_trajectory.get("planner_projected_s_raw")
                    planner_projected_s = local_trajectory.get("planner_projected_s")
                    planner_slice_start_s = local_trajectory.get("planner_slice_start_s")
                    planner_slice_lookback = local_trajectory.get("planner_slice_lookback")
                    slice_start_s = local_trajectory.get("slice_start_s")
                    slice_end_s = local_trajectory.get("slice_end_s")
                    slice_start_idx = local_trajectory.get("slice_start_idx")
                    slice_end_idx = local_trajectory.get("slice_end_idx")
                    slice_n_segments = local_trajectory.get("n_segments")

                # Always log for PSDF-family optimizers so MPCC can record path parameter s.
                timestep = len(self.pose_sdf_data)
                risk_margin_stage0 = None
                risk_margin_5 = None
                risk_margin_max = None
                if hasattr(controller._optimizer, "get_last_risk_margin_trajectory"):
                    try:
                        risk_margin_traj = controller._optimizer.get_last_risk_margin_trajectory()
                    except Exception:
                        risk_margin_traj = None
                    if risk_margin_traj is not None:
                        risk_margin_traj = np.asarray(risk_margin_traj, dtype=float).reshape(-1)
                        if risk_margin_traj.size > 0:
                            risk_margin_stage0 = float(risk_margin_traj[0])
                            risk_margin_max = float(np.max(risk_margin_traj))
                            if risk_margin_traj.size > 5:
                                risk_margin_5 = float(risk_margin_traj[5])
                if risk_margin_5 is None and hasattr(controller._optimizer, "get_last_risk_margin_at_stage"):
                    try:
                        risk_margin_5 = controller._optimizer.get_last_risk_margin_at_stage(5)
                    except Exception:
                        risk_margin_5 = None

                self.log_pose_sdf_data(
                    current_pose,
                    sdf_value,
                    timestep,
                    risk_margin_stage0=risk_margin_stage0,
                    risk_margin_5=risk_margin_5,
                    risk_margin_max=risk_margin_max,
                    path_s=path_s,
                    predicted_s=predicted_s,
                    solver_status=solver_status,
                    planner_projected_s_raw=planner_projected_s_raw,
                    planner_projected_s=planner_projected_s,
                    planner_slice_start_s=planner_slice_start_s,
                    planner_slice_lookback=planner_slice_lookback,
                    slice_start_s=slice_start_s,
                    slice_end_s=slice_end_s,
                    slice_start_idx=slice_start_idx,
                    slice_end_idx=slice_end_idx,
                    slice_n_segments=slice_n_segments,
                )
                
                return u_opt
            
            controller.generate_control_input = enhanced_generate_control
        
        self.last_run_outcome = self.sim.run_navigation(simulation_time)
        optimizer_runtime_stats = self._collect_optimizer_runtime_stats(self.robot._controller._optimizer)
        if self.last_run_outcome is None:
            self.last_run_outcome = {}
        if self.start_perturbation_info is not None:
            self.last_run_outcome["start_perturbation"] = copy.deepcopy(self.start_perturbation_info)
        if optimizer_runtime_stats:
            self.last_run_outcome["optimizer_runtime_stats"] = optimizer_runtime_stats
            if "safe_stop_count" in optimizer_runtime_stats:
                self.last_run_outcome["safe_stop_count"] = int(optimizer_runtime_stats["safe_stop_count"])
            if "backup_feasible_qp_success_count" in optimizer_runtime_stats:
                self.last_run_outcome["backup_feasible_qp_success_count"] = int(
                    optimizer_runtime_stats["backup_feasible_qp_success_count"]
                )
            if "plant_input_apply_count" in optimizer_runtime_stats:
                self.last_run_outcome["plant_input_apply_count"] = int(
                    optimizer_runtime_stats["plant_input_apply_count"]
                )
            if "last_solve_mode" in optimizer_runtime_stats:
                self.last_run_outcome["last_solve_mode"] = optimizer_runtime_stats["last_solve_mode"]

        # Print performance statistics
        print(f"Using {optimizer_name} optimizer with {dynamics_type} dynamics:")
        solver_times = list(self.robot._controller._optimizer.solver_times)
        print("median: ", st.median(solver_times))
        print("std: ", st.stdev(solver_times) if len(solver_times) > 1 else 0.0)
        print("min: ", min(solver_times))
        print("max: ", max(solver_times))
        print("Simulation finished.")
        if self.last_run_outcome is not None:
            print(
                "Run outcome: "
                f"status={self.last_run_outcome.get('status')}, "
                f"reason={self.last_run_outcome.get('failure_reason')}, "
                f"final_time={self.last_run_outcome.get('final_time')}, "
                f"distance_to_goal={self.last_run_outcome.get('distance_to_goal')}"
            )
            if optimizer_runtime_stats:
                print(
                    "Optimizer runtime stats: "
                    f"safe_stop_count={optimizer_runtime_stats.get('safe_stop_count', 0)}, "
                    f"backup_feasible_qp_success_count="
                    f"{optimizer_runtime_stats.get('backup_feasible_qp_success_count', 0)}, "
                    f"plant_input_apply_count={optimizer_runtime_stats.get('plant_input_apply_count', 0)}, "
                    f"last_solve_mode={optimizer_runtime_stats.get('last_solve_mode')}"
                )

        self.save_trial_history_to_csv(f"trial_history_{self.current_name}")

        if optimizer_type == "rmpcc":
            self.save_rmpcc_constraint_history_to_csv(
                f"rmpcc_constraints_{self.current_name}"
            )
            self.save_rmpcc_debug_history_to_csv()

        # Save pose-SDF data to CSV for PSDF-family optimizers
        if optimizer_type in ("psdf", "mpcc", "rmpcc", "rmpcc_pv"):
            csv_filename = f"pose_sdf_{self.current_name}"
            self.save_pose_sdf_to_csv(csv_filename)
        
        return robot_indexes, traj_indexes

    def create_env(self, env_type):
        if env_type == "s_path":
            s = 1.0  # scale of environment
            start = np.array([0.3 * s, 0.2 * s, 0.0])
            goal = np.array([0.8 * s, 0.8 * s, 0.0])  # Add theta=0.0 for goal orientation
            bounds = ((0.0 * s, 0.0 * s), (1.0 * s, 1.0 * s))
            cell_size = 0.05 * s
            grid = (bounds, cell_size)
            x_min, y_min = bounds[0]
            x_max, y_max = bounds[1]
            wall_thickness = 0.03 * s
            obstacles = []
            
            corridor_width = 0.15

            obstacles.append(RectangleRegion(x_min, x_min + wall_thickness, y_min, y_max))
            obstacles.append(RectangleRegion(x_max - wall_thickness, x_max, y_min, y_max))
            obstacles.append(RectangleRegion(x_min, x_max, y_min, y_min + wall_thickness))
            obstacles.append(RectangleRegion(x_min, x_max, y_max - wall_thickness, y_max))
            obstacles.append(RectangleRegion(x_min+wall_thickness, (0.5-corridor_width/2)*s, 0.35 * s, y_max-wall_thickness))
            obstacles.append(RectangleRegion((0.5+corridor_width/2)*s, x_max-wall_thickness, 0.0 * s, 0.7 * s))
            return start, goal, grid, obstacles

        elif env_type == "straight_corridor":
            s = 1.0
            corridor_center_y = 0.50 * s
            corridor_width = 0.108 * s
            start = np.array([0.15 * s, corridor_center_y, 0.0])
            goal = np.array([1.45 * s, corridor_center_y, 0.0])

            # Shift y-bounds by half a cell so y=0.50 lands on a grid-cell center.
            bounds = ((0.0 * s, -0.01 * s), (1.6 * s, 1.01 * s))
            cell_size = 0.02 * s
            grid = (bounds, cell_size)
            x_min, y_min = bounds[0]
            x_max, y_max = bounds[1]
            wall_thickness = 0.03 * s
            half_gap = 0.5 * corridor_width
            obstacles = []

            obstacles.append(RectangleRegion(x_min, x_min + wall_thickness, y_min, y_max))
            obstacles.append(RectangleRegion(x_max - wall_thickness, x_max, y_min, y_max))
            obstacles.append(RectangleRegion(x_min, x_max, y_min, y_min + wall_thickness))
            obstacles.append(RectangleRegion(x_min, x_max, y_max - wall_thickness, y_max))

            # Two small middle obstacles create a narrow straight corridor around y=0.5.
            obstacles.append(
                RectangleRegion(
                    0.68 * s,
                    0.92 * s,
                    y_min + wall_thickness,
                    corridor_center_y - half_gap,
                )
            )
            obstacles.append(
                RectangleRegion(
                    0.68 * s,
                    0.92 * s,
                    corridor_center_y + half_gap,
                    y_max - wall_thickness,
                )
            )
            return start, goal, grid, obstacles
            
        elif env_type == "parking":
            s = 1.0  # scale of environment
            start = np.array([0.3 * s, 0.3 * s, math.pi / 2.0])
            goal = np.array([0.7 * s, 0.48 * s, 0.0])  # Add theta=0.0 for final parking orientation
            bounds = ((-0.2 * s, 0.0 * s), (1.2 * s, 1.2 * s))
            cell_size = 0.02 * s  # parking 환경은 더 작은 셀 크기 사용 (기존 0.05 → 0.02)
            grid = (bounds, cell_size)
            obstacles = []
            obstacles.append(RectangleRegion(0.0 * s, 0.1 * s, 0.0 * s, 1.0 * s))
            obstacles.append(RectangleRegion(0.9 * s, 1.0 * s, 0.0 * s, 1.0 * s))
            obstacles.append(RectangleRegion(0.0 * s, 1.0 * s, 0.0 * s, 0.1 * s))
            obstacles.append(RectangleRegion(0.0 * s, 1.0 * s, 0.9 * s, 1.0 * s))
            obstacles.append(RectangleRegion(0.5 * s, 1.0 * s, 0.0 * s, 0.40 * s))
            obstacles.append(RectangleRegion(0.5 * s, 1.0 * s, 0.6* s, 1.0 * s))
            return start, goal, grid, obstacles
        

        elif env_type == "maze":
            s = 0.15  # scale of environment
            start = np.array([0.5 * s, 5.3 * s, -math.pi / 2.0])
            goal = np.array([12.0 * s, 0.6 * s, 0.0])  # Add theta=0.0 for goal orientation
            bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
            cell_size = 0.25 * s
            grid = (bounds, cell_size)
            obstacles = []
            obstacles.append(RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s))
            obstacles.append(RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s))
            obstacles.append(RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s))
            obstacles.append(RectangleRegion(6.0 * s, 7.0 * s, 4.0 * s, 6.0 * s))
            obstacles.append(RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s))
            obstacles.append(RectangleRegion(5.0 * s, 7.0 * s, 2.0 * s, 3.0 * s))
            obstacles.append(RectangleRegion(6.0 * s, 9.0 * s, 1.0 * s, 2.0 * s))
            obstacles.append(RectangleRegion(8.0 * s, 9.0 * s, 2.0 * s, 4.0 * s))
            obstacles.append(RectangleRegion(9.0 * s, 12.0 * s, 3.0 * s, 4.0 * s))
            obstacles.append(RectangleRegion(11.0 * s, 12.0 * s, 4.0 * s, 5.0 * s))
            obstacles.append(RectangleRegion(8.0 * s, 10.0 * s, 5.0 * s, 6.0 * s))
            obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 2.0 * s))
            obstacles.append(RectangleRegion(12.0 * s, 13.0 * s, 1.0 * s, 2.0 * s))
            obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, 6.0 * s, 7.0 * s))
            obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, -1.0 * s, 7.0 * s))
            obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, -1.0 * s, 0.0 * s))
            obstacles.append(RectangleRegion(13.0 * s, 14.0 * s, -1.0 * s, 7.0 * s))
            return start, goal, grid, obstacles
            
        elif env_type == "oblique_maze":
            s = 0.15  # scale of environment
            start = np.array([1.0 * s, 1.5 * s, 0.0])
            goal = np.array([8.5 * s, 6.5 * s, math.pi / 4.0])  # Add theta=45° for interesting final orientation
            bounds = ((0.0 * s, 1.0 * s), (10.0 * s, 7.0 * s))
            cell_size = 0.2 * s
            grid = (bounds, cell_size)
            obstacles = []
            # TODO: Overload the constructor of RectangleRegion() 
            obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, 0.0 * s, 8.0 * s))
            obstacles.append(RectangleRegion(0.0 * s, 10.0 * s, 0.0 * s, 1.0 * s))
            obstacles.append(RectangleRegion(0.0 * s, 8.0 * s, 7.0 * s, 8.0 * s))
            obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 8.0 * s))
            obstacles.append(
                PolytopeRegion.convex_hull(s * np.array([[0.0, 2.0], [1.25, 3.875], [2.875, 3.125], [2.5, 2.25]]))
            )
            obstacles.append(PolytopeRegion.convex_hull(s * np.array([[1, 4.75], [0.0, 5.0], [0.875, 7], [1.875, 6.375]])))
            obstacles.append(
                PolytopeRegion.convex_hull(s * np.array([[2.75, 1], [4.2, 3.25], [5.125, 3.75], [6.625, 2.5], [6.5, 1.0]]))
            )
            obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.0, 7.0], [6, 6], [6.5, 7.0]])))
            obstacles.append(
                PolytopeRegion.convex_hull(
                    s * np.array([[2.375, 4.875], [2.875, 5.875], [4.5, 5.875], [4.75, 4], [3.375, 4]])
                )
            )
            obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.75, 1.0], [7.25, 2.375], [8.5, 2.0], [8.5, 1.0]])))
            obstacles.append(PolytopeRegion.convex_hull(s * np.array([[8.625, 1.0], [10.0, 2.5], [10.0, 1.0]])))
            obstacles.append(PolytopeRegion.convex_hull(s * np.array([[10.0, 2.875], [9.5, 5.75], [10.0, 5.875]])))
            obstacles.append(
                PolytopeRegion.convex_hull(
                    s * np.array([[8.875, 3.125], [8.0, 5.5], [6.875, 6.375], [5.875, 5.875], [6.25, 4.375], [7.125, 3.5]])
                )
            )
            return start, goal, grid, obstacles

    @staticmethod
    def _resolve_config_path(config_file):
        config_path = Path(config_file)
        if config_path.is_absolute() and config_path.exists():
            return config_path

        project_root = Path(__file__).resolve().parent.parent
        candidates = [project_root / config_path]
        if len(config_path.parts) == 1:
            candidates.append(project_root / "config" / config_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return config_path


    def load_config(self, config_file):
        """Load configuration from YAML file"""
        resolved_config_file = self._resolve_config_path(config_file)
        try:
            with open(resolved_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            return config
        except FileNotFoundError:
            print(f"Config file {resolved_config_file} not found. Using default configuration.")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration when config file is not available"""
        return {
            'test_configs': [
                {
                    'maze_type': 's_path',
                    'robot_shape': 'pentagon',
                    'optimizer_type': 'psdf',
                    'dynamics_type': 'differential_drive'
                }
            ],
            'defaults': {
                'frame_skip': 5,
                'simulation_time': 15.0,
                'generate_animation': True,
                'generate_plots': True,
                'generate_risk_profile': False,
            },
        }

    def run_tests_from_config(self, config_file='config/config.yaml'):
        """Run tests based on a configuration file path or loaded config dict."""
        if isinstance(config_file, dict):
            config = config_file
        else:
            config = self.load_config(config_file)
        test_configs = config.get('test_configs', [])
        # YAML에서 test_configs가 단일 딕셔너리로 작성된 경우 리스트로 변환
        if isinstance(test_configs, dict):
            test_configs = [test_configs]
        defaults = config.get('defaults', {})
        # Resolve an automatic seed once so methods in this batch share a trial.
        start_perturbation = resolve_start_perturbation(config.get("start_perturbation"))
        
        if not test_configs:
            print("No test configurations found in config file.")
            return
        
        print(f"Running {len(test_configs)} test configurations...")
        
        for i, test_config in enumerate(test_configs, 1):
            maze_type = test_config.get('maze_type', 's_path')
            robot_shape = test_config.get('robot_shape', 'pentagon')
            optimizer_type = test_config.get('optimizer_type', 'psdf')
            dynamics_type = test_config.get('dynamics_type', 'differential_drive')
            path_planner = test_config.get('path_planner', 'astar')
            
            # 'name' 옵션을 별도로 받지 않고, 설정 조합으로 자동 생성
            test_name = f"{maze_type}_{robot_shape}_{optimizer_type}_{dynamics_type}_{path_planner}"
            
            print(f"\n{'='*60}")
            print(f"Running Test {i}/{len(test_configs)}: {test_name}")
            print(f"Configuration: {maze_type} | {robot_shape} | {optimizer_type} | {dynamics_type} | {path_planner}")
            print(f"{'='*60}")
            
            frame_skip = defaults.get('frame_skip', 5)
            generate_animation = defaults.get('generate_animation', True)
            generate_plots = defaults.get('generate_plots', True)
            use_risk_visualization = test_config.get(
                'use_risk_visualization',
                defaults.get('use_risk_visualization', None),
            )
            generate_risk_profile = test_config.get(
                'generate_risk_profile',
                defaults.get('generate_risk_profile', False),
            )
            interrupt_requested = False
            interrupted = False
            previous_sigint_handler = signal.getsignal(signal.SIGINT)

            def handle_sigint(signum, frame):
                nonlocal interrupt_requested
                interrupt_requested = True
                raise KeyboardInterrupt

            signal.signal(signal.SIGINT, handle_sigint)

            try:
                # **중요**: 각 테스트마다 완전히 새로운 simulation_mpc 인스턴스 생성
                print("Creating new simulation instance for independent test...")
                test_sim = simulation_mpc()
                test_sim.profile_heatmap_scales = test_sim._resolve_profile_heatmap_scales(
                    defaults,
                    config,
                    test_config,
                )

                # Get simulation time from defaults and run test
                simulation_time = defaults.get('simulation_time', 20.0)
                run_config = copy.deepcopy(config)
                per_test_perturbation = test_config.get("start_perturbation", {})
                if not isinstance(per_test_perturbation, dict):
                    raise ValueError("Per-test start_perturbation must be a mapping.")
                run_config["start_perturbation"] = {
                    **start_perturbation, **per_test_perturbation,
                }
                robot_indexes, traj_indexes = test_sim.mpc_test(
                    maze_type, robot_shape, optimizer_type, dynamics_type, path_planner, simulation_time, run_config
                )

                if generate_animation:
                    print("Generating animation...")
                    test_sim.animate_world(
                        test_sim.sim,
                        animation_name=test_sim.current_name,
                        maze_type=maze_type,
                        frame_skip=frame_skip,
                        method_name=optimizer_type,
                        use_risk_visualization=use_risk_visualization,
                    )

                if generate_plots:
                    print("Generating plots...")
                    test_sim.plot_world(
                        test_sim.sim,
                        snapshot_indexes=[],
                        figure_name=test_sim.current_name.lower(),
                        local_traj_indexes=[],
                        maze_type=maze_type,
                    )
                    test_sim.plot_profiles(
                        test_sim.sim,
                        figure_name=test_sim.current_name.lower(),
                        maze_type=maze_type,
                        include_risk_profile=generate_risk_profile,
                    )

                print(f"✓ Test {test_name} completed successfully")

                # **중요**: 테스트 완료 후 정리
                print("Cleaning up test resources...")
                if hasattr(test_sim.robot, '_controller') and hasattr(test_sim.robot._controller, '_optimizer'):
                    optimizer = test_sim.robot._controller._optimizer
                    if hasattr(optimizer, 'cleanup'):
                        optimizer.cleanup()
                        print("Optimizer cleaned up successfully")

                # 명시적으로 변수들을 None으로 설정하여 메모리 해제 촉진
                test_sim.sim = None
                test_sim.robot = None
                del test_sim

            except KeyboardInterrupt:
                interrupt_requested = True
                interrupted = True
                print("\nKeyboard interrupt received. Finalizing outputs from the partial run...")
                if generate_animation:
                    if self.has_animation_data_for(getattr(test_sim, 'sim', None)):
                        animation_name = getattr(
                            test_sim,
                            'current_name',
                            self.build_output_name(robot_shape, maze_type, optimizer_type, interrupted=True),
                        )
                        if animation_name == getattr(test_sim, 'current_name', None):
                            animation_name += "_interrupted"

                        print("Generating interrupted-run animation...")
                        test_sim.animate_world(
                            test_sim.sim,
                            animation_name=animation_name,
                            maze_type=maze_type,
                            frame_skip=frame_skip,
                            method_name=optimizer_type,
                            use_risk_visualization=use_risk_visualization,
                        )
                    else:
                        print("Interrupted before any trajectory state was available; skipping animation generation.")

            except RuntimeError as e:
                if not self.is_interrupt_runtime_error(e, interrupt_requested):
                    print(f"✗ Test {test_name} failed with error: {e}")
                    import traceback
                    traceback.print_exc()

                    # 오류 발생시에도 정리 시도
                    try:
                        if 'test_sim' in locals():
                            if hasattr(test_sim.robot, '_controller') and hasattr(test_sim.robot._controller, '_optimizer'):
                                optimizer = test_sim.robot._controller._optimizer
                                if hasattr(optimizer, 'cleanup'):
                                    optimizer.cleanup()
                            del test_sim
                    except Exception as cleanup_error:
                        print(f"Warning: Error during cleanup: {cleanup_error}")

                    continue

                interrupted = True
                print("\nSIGINT interrupted the solver. Finalizing outputs from the partial run...")
                if generate_animation:
                    if self.has_animation_data_for(getattr(test_sim, 'sim', None)):
                        animation_name = getattr(
                            test_sim,
                            'current_name',
                            self.build_output_name(robot_shape, maze_type, optimizer_type, interrupted=True),
                        )
                        if animation_name == getattr(test_sim, 'current_name', None):
                            animation_name += "_interrupted"

                        print("Generating interrupted-run animation...")
                        test_sim.animate_world(
                            test_sim.sim,
                            animation_name=animation_name,
                            maze_type=maze_type,
                            frame_skip=frame_skip,
                            method_name=optimizer_type,
                            use_risk_visualization=use_risk_visualization,
                        )
                    else:
                        print("Interrupted before any trajectory state was available; skipping animation generation.")

            except Exception as e:
                print(f"✗ Test {test_name} failed with error: {e}")
                import traceback
                traceback.print_exc()

                # 오류 발생시에도 정리 시도
                try:
                    if 'test_sim' in locals():
                        if hasattr(test_sim.robot, '_controller') and hasattr(test_sim.robot._controller, '_optimizer'):
                            optimizer = test_sim.robot._controller._optimizer
                            if hasattr(optimizer, 'cleanup'):
                                optimizer.cleanup()
                        del test_sim
                except Exception as cleanup_error:
                    print(f"Warning: Error during cleanup: {cleanup_error}")

                continue
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
                if interrupted:
                    print("Cleaning up test resources...")
                    try:
                        if hasattr(test_sim.robot, '_controller') and hasattr(test_sim.robot._controller, '_optimizer'):
                            optimizer = test_sim.robot._controller._optimizer
                            if hasattr(optimizer, 'cleanup'):
                                optimizer.cleanup()
                                print("Optimizer cleaned up successfully")
                    except Exception as cleanup_error:
                        print(f"Warning: Error during cleanup: {cleanup_error}")

                    if 'test_sim' in locals():
                        test_sim.sim = None
                        test_sim.robot = None
                        del test_sim

            if interrupted:
                break
        
        print(f"\n{'='*60}")
        if interrupted:
            print("Test run interrupted by user.")
        else:
            print("All tests completed!")
        print(f"{'='*60}")
