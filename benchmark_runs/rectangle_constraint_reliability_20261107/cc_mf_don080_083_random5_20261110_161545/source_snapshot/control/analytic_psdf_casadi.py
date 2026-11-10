"""CasADi affine bridge backed by the augmented PSDF analytic gradient."""

from __future__ import annotations

from typing import Union

import casadi as ca
import numpy as np
import torch
from torch import Tensor


class AnalyticPSDFCasADi:
    """Expose ``phi_bar + g_bar.T @ (x - x_bar)`` to CasADi.

    The numeric parameter layout intentionally matches first-order
    ``RealTimeL4CasADi``: ``[x_bar(3), phi_bar(1), gradient(3)]``.
    """

    parameter_dim = 7
    requires_external_shared_lib = False

    def __init__(
        self,
        model,
        device: Union[str, torch.device] = "cpu",
        name: str = "analytic_psdf",
    ):
        self.model = model
        self.device = torch.device(device)
        self.name = str(name)

        self._x_bar = ca.MX.sym(f"{self.name}_x_bar", 3, 1)
        self._phi_bar = ca.MX.sym(f"{self.name}_phi_bar", 1, 1)
        self._gradient = ca.MX.sym(f"{self.name}_gradient", 3, 1)
        self._sym_params = ca.vertcat(self._x_bar, self._phi_bar, self._gradient)

    def __call__(self, pose):
        if not isinstance(pose, (ca.MX, ca.SX, ca.DM)):
            raise TypeError("AnalyticPSDFCasADi expects a CasADi MX, SX, or DM input")
        if int(np.prod(pose.shape)) != 3:
            raise ValueError(f"expected a three-element pose vector, got shape {pose.shape}")

        pose_vector = ca.reshape(pose, 3, 1)
        return self._phi_bar + ca.dot(self._gradient, pose_vector - self._x_bar)

    def get_sym_params(self):
        return self._sym_params

    def get_params(self, pose: Union[np.ndarray, Tensor]) -> np.ndarray:
        pose_array = np.asarray(
            pose.detach().cpu().numpy() if isinstance(pose, Tensor) else pose
        )
        single_pose = pose_array.ndim == 1
        if single_pose:
            if pose_array.shape != (3,):
                raise ValueError(f"expected pose shape (3,), got {pose_array.shape}")
            pose_array = pose_array[None, :]
        elif pose_array.ndim != 2 or pose_array.shape[1] != 3:
            raise ValueError(f"expected pose shape (B, 3), got {pose_array.shape}")
        if not np.issubdtype(pose_array.dtype, np.number):
            raise TypeError("pose values must be numeric")
        if not np.isfinite(pose_array).all():
            raise ValueError("pose values must be finite")

        model_dtype = self.model.A.dtype
        pose_tensor = torch.as_tensor(pose_array, dtype=model_dtype, device=self.device)
        phi, gradient = self.model(pose_tensor)
        if phi.shape != (pose_tensor.shape[0],):
            raise ValueError(f"model phi must have shape (B,), got {tuple(phi.shape)}")
        if gradient.shape != (pose_tensor.shape[0], 3):
            raise ValueError(f"model gradient must have shape (B, 3), got {tuple(gradient.shape)}")
        if not torch.isfinite(phi).all() or not torch.isfinite(gradient).all():
            raise ValueError("model phi and gradient outputs must be finite")

        params = torch.cat((pose_tensor, phi.unsqueeze(1), gradient), dim=1)
        params_array = params.detach().cpu().numpy()
        return params_array[0] if single_pose else params_array
