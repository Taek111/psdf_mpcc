"""Read the archived PSDF's actual branch selectors without modifying its code.

The profiler observes local variables only during an offline forward call. It
does not wrap/reimplement the geometry or run a controller. Raw IDs identify
computational branches, not necessarily different geometric contacts. Canonical
separation IDs collapse duplicate A/B endpoint aliases and point-to-segment
families that select the same vertex pair. SAT IDs remain axis/support selectors.
Ties, duplicate parallel SAT axes, and continuously moving closest points mean
an ID change alone is not evidence of a gradient or risk-margin discontinuity.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch


_LOCALS = (
    "active_cluster", "separated", "idx_sep", "idx_v", "idx_e", "axis_star",
    "q_active", "c_R", "c_O", "feature_sq", "d_v_sq", "d_e_sq",
    "signed_cluster", "delta_all", "ov_poly_side_1", "ov_poly_side_2",
    "idx_ends_min", "idx_ends_max", "ov_seg_side_1", "ov_seg_side_2",
    "idx_poly_min", "idx_poly_max", "proj_all",
)


def _numpy(value):
    return value.detach().cpu().numpy()


def _capture_forward(wrapper, poses, verify):
    target = inspect.unwrap(wrapper.psdf._forward_impl).__code__
    point_segment = inspect.unwrap(wrapper.psdf._p2seg).__code__
    captured, projections = [], []
    previous_profiler = sys.getprofile()

    def observe(frame, event, result):
        if event == "return" and frame.f_code is target:
            captured.append({key: frame.f_locals[key] for key in _LOCALS})
        elif event == "return" and frame.f_code is point_segment:
            projections.append(frame.f_locals["t"])

    try:
        sys.setprofile(observe)
        with torch.no_grad():
            phi, gradient = wrapper(poses)
    finally:
        sys.setprofile(previous_profiler)
    if len(captured) != 1 or len(projections) != 2:
        raise RuntimeError("Archived PSDF selector/projection call structure changed")
    if verify:
        with torch.no_grad():
            direct_phi, direct_gradient = wrapper(poses)
        if not torch.equal(phi, direct_phi) or not torch.equal(gradient, direct_gradient):
            raise ValueError("Observed PSDF forward differs from direct forward")
    values = {key: _numpy(value) for key, value in captured[0].items()}
    values["projection_v"], values["projection_e"] = map(_numpy, projections)
    return _numpy(phi), _numpy(gradient), values


def _endpoint_aliases(wrapper):
    """Map each stored A/B endpoint to one coordinate-stable vertex per cluster."""
    a, b = _numpy(wrapper.A), _numpy(wrapper.B)
    valid = _numpy(wrapper.mask)
    aliases = np.full((wrapper.active_clusters, 2 * a.shape[1]), -1, dtype=int)
    for cluster in range(wrapper.active_clusters):
        identifiers = {}
        for offset, points in ((0, a[cluster]), (a.shape[1], b[cluster])):
            for edge in np.flatnonzero(valid[cluster]):
                coordinates = tuple(float(value) for value in points[edge])
                if coordinates not in identifiers:
                    identifiers[coordinates] = len(identifiers)
                aliases[cluster, offset + edge] = identifiers[coordinates]
    return aliases


def _gap(values):
    ordered = np.sort(np.asarray(values))
    return float(ordered[1] - ordered[0]) if len(ordered) > 1 else np.nan


def extract_feature_selectors(wrapper, poses, *, batch_size=128,
                              verify=True, tie_tolerance_m=1e-7):
    """Return cycle-aligned arrays for the wrapper's active PSDF branch.

    ``wrapper`` must be an initialized archived AugmentedPSDFWrapper with a
    static obstacle snapshot. ``poses`` has shape (N, 3). Geometry uses the
    wrapper's own dtype/device; profiling is restored even if evaluation fails.

    Main fields: ``phi``, ``world_gradient``, ``local_q``, ``raw_id``,
    ``canonical_id``, ``active_cluster``, ``separated``, selected body-frame
    robot/obstacle points (separation only), and explicit selector indices.
    ``feature_gap_m`` is the nearest/second-nearest distance gap among the
    pooled 2*m features of the active cluster; ``subfeature_gap_m`` is the gap
    within its selected unpooled edge/endpoint candidates. Duplicate witnesses
    can yield zero gaps without distinct normals. ``near_tie`` is a numerical
    diagnostic with the stated tolerance, not proof of physical ambiguity.

    SAT canonical IDs identify computational axis/side/support choices; opposite
    parallel axes are intentionally not merged. No switching event is inferred
    here. The caller must preserve chronology and handle unavailable samples.
    """
    if wrapper.active_clusters <= 0:
        raise ValueError("Selector extraction requires at least one obstacle cluster")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(tie_tolerance_m) or tie_tolerance_m < 0:
        raise ValueError("tie_tolerance_m must be finite and nonnegative")
    pose_tensor = torch.as_tensor(poses, dtype=wrapper.A.dtype, device=wrapper.A.device)
    if pose_tensor.ndim != 2 or pose_tensor.shape[1] != 3 or len(pose_tensor) == 0:
        raise ValueError("poses must be a nonempty (N, 3) batch")
    if not torch.isfinite(pose_tensor).all():
        raise ValueError("poses must be finite")
    aliases = _endpoint_aliases(wrapper)
    edges, vertices = wrapper.A.shape[1], wrapper.psdf.V.shape[0]
    obstacle_mask = _numpy(wrapper.mask)
    rows, phis, gradients, local_q = [], [], [], []
    for start in range(0, len(pose_tensor), batch_size):
        phi, gradient, data = _capture_forward(
            wrapper, pose_tensor[start:start + batch_size], verify)
        phis.append(phi)
        gradients.append(gradient)
        local_q.append(data["q_active"])
        for i in range(len(phi)):
            cluster = int(data["active_cluster"][i])
            separated = bool(data["separated"][i, cluster])
            row = {
                "active_cluster": cluster, "separated": separated,
                "feature_index": -1, "obstacle_subfeature_index": -1,
                "projection_region": -1, "sat_axis": -1, "sat_side": -1,
                "sat_robot_support": -1, "sat_obstacle_support": -1,
                "selected_robot_point_body": np.full(2, np.nan),
                "selected_obstacle_point_body": np.full(2, np.nan),
                "feature_gap_m": np.nan, "subfeature_gap_m": np.nan,
                "feature_tie_count": 0, "sat_axis_gap_m": np.nan,
                "cluster_gap_m": _gap(data["signed_cluster"][i]),
            }
            if separated:
                feature = int(data["idx_sep"][i, cluster])
                row["feature_index"] = feature
                distances = np.sqrt(np.maximum(data["feature_sq"][i, cluster], 0))
                row["feature_gap_m"] = _gap(distances)
                row["feature_tie_count"] = int(np.sum(distances <= distances.min() + tie_tolerance_m))
                row["selected_robot_point_body"] = data["c_R"][i, cluster, feature]
                row["selected_obstacle_point_body"] = data["c_O"][i, cluster, feature]
                if feature < vertices:
                    edge = int(data["idx_v"][i, cluster, feature])
                    t = data["projection_v"][i, cluster, edge, feature, 0]
                    region = 0 if t == 0 else 2 if t == 1 else 1
                    robot = f"V{feature}"
                    obstacle = (f"V{aliases[cluster, edge if region == 0 else edges + edge]}"
                                if region != 1 else f"E{edge}")
                    row["raw_id"] = f"C{cluster}|sep|RV{feature}|OE{edge}|p{region}"
                    candidates = data["d_v_sq"][i, cluster, obstacle_mask[cluster], feature]
                    row["obstacle_subfeature_index"] = edge
                else:
                    edge = feature - vertices
                    endpoint = int(data["idx_e"][i, cluster, edge])
                    t = data["projection_e"][i, cluster, endpoint, edge, 0]
                    region = 0 if t == 0 else 2 if t == 1 else 1
                    robot = (f"V{edge if region == 0 else (edge + 1) % vertices}"
                             if region != 1 else f"E{edge}")
                    obstacle = f"V{aliases[cluster, endpoint]}"
                    row["raw_id"] = f"C{cluster}|sep|RE{edge}|OP{endpoint}|p{region}"
                    valid_endpoints = np.r_[obstacle_mask[cluster], obstacle_mask[cluster]]
                    candidates = data["d_e_sq"][i, cluster, valid_endpoints, edge]
                    row["obstacle_subfeature_index"] = endpoint
                row["projection_region"] = region
                row["subfeature_gap_m"] = _gap(np.sqrt(np.maximum(candidates, 0)))
                row["canonical_id"] = f"C{cluster}|sep|R{robot}|O{obstacle}"
                row["canonical_kind"] = "geometric_pair"
            else:
                axis = int(data["axis_star"][i, cluster])
                row["sat_axis"] = axis
                valid_axes = np.r_[np.ones(vertices, dtype=bool), obstacle_mask[cluster]]
                row["sat_axis_gap_m"] = _gap(data["delta_all"][i, cluster, valid_axes])
                if axis < vertices:
                    side = 1 if data["ov_poly_side_1"][i, cluster, axis] <= data["ov_poly_side_2"][i, cluster, axis] else 2
                    key = "idx_ends_max" if side == 1 else "idx_ends_min"
                    endpoint = int(data[key][i, cluster, axis])
                    raw_support = f"OP{endpoint}"
                    canonical_support = f"OV{aliases[cluster, endpoint]}"
                    axis_name = f"RA{axis}"
                    row["sat_obstacle_support"] = endpoint
                else:
                    edge = axis - vertices
                    side = 1 if data["ov_seg_side_1"][i, cluster, edge] <= data["ov_seg_side_2"][i, cluster, edge] else 2
                    key = "idx_poly_min" if side == 1 else "idx_poly_max"
                    vertex = int(data[key][i, cluster, edge])
                    projection = data["proj_all"][i, cluster, edge].copy()
                    projection[~obstacle_mask[cluster]] = -np.inf if side == 1 else np.inf
                    support = int(np.argmax(projection) if side == 1 else np.argmin(projection))
                    raw_support = f"RV{vertex}|OA{support}"
                    canonical_support = f"RV{vertex}|OV{aliases[cluster, support]}"
                    axis_name = f"OA{edge}"
                    row["sat_robot_support"], row["sat_obstacle_support"] = vertex, support
                row["sat_side"] = side
                row["raw_id"] = f"C{cluster}|sat|{axis_name}|side{side}|{raw_support}"
                row["canonical_id"] = f"C{cluster}|sat|{axis_name}|side{side}|{canonical_support}"
                row["canonical_kind"] = "sat_selector"
            gaps = [row[key] for key in ("feature_gap_m", "subfeature_gap_m", "sat_axis_gap_m", "cluster_gap_m")]
            row["near_tie"] = any(np.isfinite(gap) and gap <= tie_tolerance_m for gap in gaps)
            rows.append(row)
    result = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    result.update(phi=np.concatenate(phis), world_gradient=np.concatenate(gradients),
                  local_q=np.concatenate(local_q), poses_geometry=_numpy(pose_tensor))
    source = Path(inspect.getsourcefile(inspect.unwrap(wrapper.psdf._forward_impl)))
    result["metadata"] = {
        "archived_geometry_source": str(source),
        "archived_geometry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "geometry_dtype": str(wrapper.A.dtype), "geometry_device": str(wrapper.A.device),
        "direct_forward_bitwise_verified": bool(verify), "tie_tolerance_m": tie_tolerance_m,
        "projection_region": {"0": "segment start", "1": "segment interior", "2": "segment end"},
        "feature_gap_definition": "top-two pooled separation distances within the selected cluster",
        "canonical_id_caveat": "Endpoint aliases are merged; SAT axes and distinct tied witnesses are not. ID changes do not establish gradient/risk jumps.",
    }
    return result
