"""Benchmark-only, gradient-free counterpart of the augmented scalar PSDF.

No controller imports or production files are changed. The existing scalar
``models.psdf.PSDF.forward`` supplies the distance/SAT arithmetic. Its separate
ray-inside override is removed because augmented_psdf uses the SAT penetration
depth for every nonseparated cluster, including containment. The three source
assignments below are matched exactly; changes to the source fail explicitly.

The scalar pipeline and augmented pipeline reduce the same squared-distance
candidates in different orders. Scalar values should agree, but this prototype
must pass numerical parity checks against AugmentedPSDFWrapper before timing
results can be interpreted as a potential replacement. It does not compute or
validate gradients and is not an implemented safety mechanism.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.augmented_psdf_wrapper import AugmentedPSDFWrapper
from models.psdf import PSDF as ScalarPSDF


def _make_sat_only_forward():
    """Adapt precisely three scalar-forward assignments at import time."""
    source = textwrap.dedent(inspect.getsource(ScalarPSDF.forward))
    tree = ast.parse(source)
    forward = tree.body[0]
    expected = {
        "inside": "inside = self._ray_inside(A_loc, B_loc, mask)",
        "inside_override": "separated = separated & (~inside)",
        "signed_cluster": (
            "signed_cluster = torch.where(separated, sep_dist, "
            "torch.where(inside, -sep_dist, -penetration))"
        ),
    }
    signatures = {
        name: ast.dump(ast.parse(statement).body[0], include_attributes=False)
        for name, statement in expected.items()
    }
    matches = {name: 0 for name in expected}
    updated = []
    for statement in forward.body:
        signature = ast.dump(statement, include_attributes=False)
        matching = next(
            (name for name, value in signatures.items() if signature == value),
            None,
        )
        if matching is None:
            updated.append(statement)
            continue
        matches[matching] += 1
        if matching == "signed_cluster":
            replacement = ast.parse(
                "signed_cluster = torch.where(separated, sep_dist, -penetration)"
            ).body[0]
            updated.append(ast.copy_location(replacement, statement))
    if any(count != 1 for count in matches.values()):
        raise RuntimeError(
            "Scalar PSDF source changed; audit prototype transformation: "
            f"{matches}"
        )
    forward.body = updated
    ast.fix_missing_locations(tree)
    namespace = {"torch": torch, "F": F, "Tensor": Tensor}
    exec(compile(tree, inspect.getsourcefile(ScalarPSDF), "exec"), namespace)
    return torch.no_grad()(namespace["forward"])


_SAT_ONLY_FORWARD = _make_sat_only_forward()


class PhiOnlyPSDF(ScalarPSDF):
    """Exact scalar candidate with augmented hard-SAT sign/depth semantics."""

    def forward(self, A, B, mask, poses):
        # Match the augmented entry point's metadata and CPU value checks, so
        # benchmarking does not conflate removing gradients with skipping
        # production validation. CUDA keeps its existing asynchronous contract.
        if A.ndim != 3 or A.shape[-1] != 2 or B.shape != A.shape:
            raise ValueError("A and B must have the same shape (K, E, 2)")
        if A.shape[0] == 0 or A.shape[1] == 0:
            raise ValueError("A and B must contain at least one cluster and edge slot")
        if mask.shape != A.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool with shape (K, E)")
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError("poses must have shape (H, 3)")
        if not A.is_floating_point() or not B.is_floating_point():
            raise TypeError("A and B must be floating-point tensors")
        if not poses.is_floating_point():
            raise TypeError("poses must be a floating-point tensor")
        expected_tensors = (A, B, poses)
        if any(t.dtype != self.V.dtype for t in expected_tensors):
            raise TypeError("A, B, poses, and verts must use the same dtype")
        if any(t.device != self.V.device for t in (*expected_tensors, mask)):
            raise ValueError("all inputs and the PSDF module must use the same device")
        if A.device.type == "cpu":
            if not mask.any(-1).all():
                raise ValueError("every obstacle cluster must contain at least one valid edge")
            if not all(torch.isfinite(t).all() for t in expected_tensors):
                raise ValueError("all floating-point inputs must be finite")
        return _SAT_ONLY_FORWARD(self, A, B, mask, poses)


class PhiOnlyPSDFWrapper(AugmentedPSDFWrapper):
    """Reuse production obstacle storage and pose normalization, return phi."""

    def __init__(self, verts: Tensor, K_max: int, E_max: int, device="cpu"):
        super().__init__(verts, K_max, E_max, device)
        self.psdf = PhiOnlyPSDF(self.psdf.V).to(self.device)

    def forward(self, pose: Tensor) -> Tensor:
        pose_t = self._normalize_pose(pose)
        if not torch.isfinite(pose_t).all():
            raise ValueError("pose values must be finite")
        if self.active_clusters == 0:
            return torch.full(
                (pose_t.shape[0],), 1000.0,
                dtype=pose_t.dtype, device=pose_t.device,
            )
        return self.psdf(
            self.A[:self.active_clusters],
            self.B[:self.active_clusters],
            self.mask[:self.active_clusters],
            pose_t,
        )


def compute_exact_phi(wrapper: PhiOnlyPSDFWrapper, poses):
    """NumPy-to-NumPy adapter including transfer and finite-value validation."""
    poses_np = np.asarray(poses, dtype=float)
    squeeze = poses_np.ndim == 1
    poses_np = np.atleast_2d(poses_np)
    if poses_np.ndim != 2 or poses_np.shape[1] != 3:
        raise ValueError("poses must have shape (H, 3) or (3,)")
    if not np.isfinite(poses_np).all():
        raise ValueError("pose values must be finite")
    poses_t = torch.as_tensor(
        poses_np, dtype=wrapper.A.dtype, device=wrapper.device,
    )
    phi = wrapper(poses_t).detach().cpu().numpy()
    if not np.isfinite(phi).all():
        raise ValueError("PSDF returned non-finite signed-distance data")
    return float(phi[0]) if squeeze else phi
