"""Reproducible, clearance-checked Gaussian perturbations of the true start pose."""

import secrets

import numpy as np


DEFAULT_START_PERTURBATION = {
    "enabled": False,
    "seed": None,
    "position_std": 0.005,  # Independent world-frame x/y noise [m].
    "heading_std_deg": 0.0,  # Preserve the nominal heading by default.
    "max_sigma": 3.0,
    "min_clearance": 0.01,  # Matches the current OBCA safety margin [m].
    "max_attempts": 1000,
}


def resolve_start_perturbation(settings=None):
    """Validate settings and record an entropy seed when none was supplied."""
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("start_perturbation must be a mapping.")
    options = {**DEFAULT_START_PERTURBATION, **(settings or {})}
    if not isinstance(options["enabled"], bool):
        raise ValueError("start_perturbation.enabled must be true or false.")
    for key in ("position_std", "heading_std_deg", "max_sigma", "min_clearance"):
        value = float(options[key])
        if not np.isfinite(value) or value < 0 or (key == "max_sigma" and value == 0):
            raise ValueError(f"start_perturbation.{key} has an invalid value: {value}")
        options[key] = value
    for key in ("seed", "max_attempts"):
        value = options[key]
        if key == "seed" and value is None:
            continue
        minimum = 1 if key == "max_attempts" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"start_perturbation.{key} must be an integer >= {minimum}.")
    if options["enabled"] and options["seed"] is None:
        options["seed"] = secrets.randbits(32)
    return options


def convex_polygon_clearance(first, second):
    """Euclidean distance between convex polygons; zero for overlap or contact."""
    edges_first = np.roll(first, -1, axis=0) - first
    edges_second = np.roll(second, -1, axis=0) - second
    edges = np.vstack((edges_first, edges_second))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    axes = axes[np.linalg.norm(axes, axis=1) > 1e-12]
    projection_first = first @ axes.T
    projection_second = second @ axes.T
    separated = (
        (projection_first.max(axis=0) < projection_second.min(axis=0))
        | (projection_second.max(axis=0) < projection_first.min(axis=0))
    )
    if not np.any(separated):
        return 0.0

    def point_edge_distance(points, vertices, segments):
        delta = points[:, None, :] - vertices[None, :, :]
        squared_lengths = np.sum(segments * segments, axis=1)
        fraction = np.sum(delta * segments[None, :, :], axis=2) / np.maximum(
            squared_lengths, 1e-24
        )
        closest_delta = delta - np.clip(fraction, 0.0, 1.0)[:, :, None] * segments
        return float(np.min(np.linalg.norm(closest_delta, axis=2)))

    return min(
        point_edge_distance(first, second, edges_second),
        point_edge_distance(second, first, edges_first),
    )


def pose_clearance(pose, footprint_polygons, obstacle_polygons, bounds):
    """Minimum footprint clearance to obstacles and the map boundary [m]."""
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    lower, upper = np.asarray(bounds, dtype=float)
    clearance = float("inf")
    for polygon in footprint_polygons:
        world_vertices = polygon @ rotation.T + pose[:2]
        clearance = min(
            clearance,
            float(np.min(world_vertices - lower)),
            float(np.min(upper - world_vertices)),
        )
        for obstacle in obstacle_polygons:
            clearance = min(clearance, convex_polygon_clearance(world_vertices, obstacle))
    return clearance


def sample_start_pose(nominal_pose, geometry, obstacles, bounds, settings=None):
    """Rejection-sample a bounded Gaussian, independently of optimizer/RNG state.

    The accepted distribution is conditioned on component-wise 3-sigma bounds
    (configurable) and footprint clearance. This checks geometric start safety,
    not feasibility or convergence of the full nonlinear MPC horizon.
    """
    options = resolve_start_perturbation(settings)
    nominal = np.asarray(nominal_pose, dtype=float).copy()
    if not options["enabled"]:
        return nominal, None
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("Start pose must contain finite [x, y, theta].")
    footprints = [
        np.asarray(region.get_ccw_vertices(), dtype=float)
        for region in geometry.equiv_rep()
    ]
    obstacle_polygons = [
        np.asarray(obstacle.get_ccw_vertices(), dtype=float)
        for obstacle in obstacles
    ]
    if not footprints or any(len(polygon) < 3 for polygon in footprints + obstacle_polygons):
        raise ValueError("Start perturbation requires polygonal robot/obstacle geometry.")

    standard_deviation = np.array([
        options["position_std"], options["position_std"],
        np.deg2rad(options["heading_std_deg"]),
    ])
    rng = np.random.default_rng(options["seed"])
    nominal_clearance = pose_clearance(nominal, footprints, obstacle_polygons, bounds)
    for attempt in range(1, options["max_attempts"] + 1):
        noise = rng.normal(size=3)
        if np.any(np.abs(noise[standard_deviation > 0]) > options["max_sigma"]):
            continue
        delta = noise * standard_deviation
        candidate = nominal + delta
        if standard_deviation[2] > 0:
            candidate[2] = (candidate[2] + np.pi) % (2 * np.pi) - np.pi
        clearance = pose_clearance(candidate, footprints, obstacle_polygons, bounds)
        if clearance > 0 and clearance >= options["min_clearance"]:
            return candidate, {
                **options,
                "nominal_pose": nominal.tolist(),
                "initial_pose": candidate.tolist(),
                "delta_pose": delta.tolist(),
                "nominal_clearance": nominal_clearance,
                "initial_clearance": clearance,
                "attempts": attempt,
            }
    raise ValueError(
        f"Could not sample a safe start in {options['max_attempts']} attempts "
        f"(seed={options['seed']}, required clearance={options['min_clearance']:.4f} m, "
        f"nominal clearance={nominal_clearance:.4f} m). "
        "Check the nominal pose, footprint and start_perturbation settings."
    )
