"""Buffered obstacle wrapper for the augmented analytic PSDF."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from models.augmented_psdf import PSDF as AugmentedPSDF


class AugmentedPSDFWrapper(nn.Module):
    """Store padded obstacle clusters and return ``(phi, gradient)``."""

    def __init__(
        self,
        verts: Tensor,
        K_max: int,
        E_max: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if K_max <= 0 or E_max <= 0:
            raise ValueError("K_max and E_max must be positive")

        self.device = torch.device(device)
        verts_t = torch.as_tensor(verts, device=self.device)
        if not verts_t.is_floating_point():
            raise TypeError("verts must be a floating-point tensor")

        self.psdf = AugmentedPSDF(verts_t).to(self.device)
        self.K_max = int(K_max)
        self.E_max = int(E_max)
        self.register_buffer(
            "A",
            torch.zeros(
                self.K_max,
                self.E_max,
                2,
                dtype=verts_t.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "B",
            torch.zeros(self.K_max, self.E_max, 2, dtype=verts_t.dtype, device=self.device),
        )
        self.register_buffer(
            "mask",
            torch.zeros(self.K_max, self.E_max, dtype=torch.bool, device=self.device),
        )
        self.active_clusters = 0

    def _reset_buffers(self) -> None:
        self.A.zero_()
        self.B.zero_()
        self.mask.zero_()
        self.active_clusters = 0

    @torch.no_grad()
    def update_edge_clusters(
        self,
        clusters_A,
        clusters_B,
        clusters_mask: Optional[Tensor] = None,
    ) -> None:
        """Replace all active obstacle clusters with a new ROI snapshot."""

        self._reset_buffers()
        if isinstance(clusters_A, (list, tuple)):
            if not isinstance(clusters_B, (list, tuple)):
                raise TypeError(
                    "clusters_B must be a sequence when clusters_A is a sequence"
                )
            if len(clusters_A) != len(clusters_B):
                raise ValueError("clusters_A and clusters_B must have the same length")

            K_roi = len(clusters_A)
            if K_roi > self.K_max:
                raise ValueError(f"ROI clusters ({K_roi}) exceeds K_max ({self.K_max})")

            for cluster_index, (edges_A, edges_B) in enumerate(
                zip(clusters_A, clusters_B)
            ):
                edges_A_t = torch.as_tensor(
                    edges_A,
                    dtype=self.A.dtype,
                    device=self.A.device,
                )
                edges_B_t = torch.as_tensor(
                    edges_B,
                    dtype=self.B.dtype,
                    device=self.B.device,
                )
                if edges_A_t.ndim != 2 or edges_A_t.shape[-1] != 2:
                    raise ValueError("every edge cluster must have shape (E, 2)")
                if edges_B_t.shape != edges_A_t.shape:
                    raise ValueError("matching A and B clusters must have the same shape")

                E_roi = edges_A_t.shape[0]
                if E_roi == 0:
                    raise ValueError(
                        "every obstacle cluster must contain at least one edge"
                    )
                if E_roi > self.E_max:
                    raise ValueError(
                        f"Cluster {cluster_index} edge count ({E_roi}) exceeds E_max ({self.E_max})"
                    )

                self.A[cluster_index, :E_roi].copy_(edges_A_t)
                self.B[cluster_index, :E_roi].copy_(edges_B_t)
                self.mask[cluster_index, :E_roi] = True

            self.active_clusters = K_roi
            return

        edges_A_t = torch.as_tensor(
            clusters_A,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        edges_B_t = torch.as_tensor(
            clusters_B,
            dtype=self.B.dtype,
            device=self.B.device,
        )
        if edges_A_t.ndim != 3 or edges_A_t.shape[-1] != 2:
            raise ValueError("tensor edge clusters must have shape (K, E, 2)")
        if edges_B_t.shape != edges_A_t.shape:
            raise ValueError("clusters_A and clusters_B must have the same shape")

        K_roi, E_roi = edges_A_t.shape[:2]
        if K_roi > self.K_max:
            raise ValueError(f"ROI clusters ({K_roi}) exceeds K_max ({self.K_max})")
        if E_roi > self.E_max:
            raise ValueError(f"ROI edges ({E_roi}) exceeds E_max ({self.E_max})")
        if K_roi == 0:
            return

        self.A[:K_roi, :E_roi].copy_(edges_A_t)
        self.B[:K_roi, :E_roi].copy_(edges_B_t)
        if clusters_mask is None:
            inferred_mask = (edges_A_t.abs().sum(-1) > 1e-6) | (
                edges_B_t.abs().sum(-1) > 1e-6
            )
            self.mask[:K_roi, :E_roi].copy_(inferred_mask)
        else:
            mask_t = torch.as_tensor(
                clusters_mask,
                dtype=torch.bool,
                device=self.mask.device,
            )
            if mask_t.shape != (K_roi, E_roi):
                raise ValueError("clusters_mask must have shape (K, E)")
            self.mask[:K_roi, :E_roi].copy_(mask_t)

        if not self.mask[:K_roi, :E_roi].any(-1).all():
            self._reset_buffers()
            raise ValueError("every obstacle cluster must contain at least one valid edge")
        self.active_clusters = K_roi

    def _normalize_pose(self, pose: Tensor) -> Tensor:
        pose_t = torch.as_tensor(pose, dtype=self.A.dtype, device=self.A.device)
        if pose_t.ndim == 1:
            if pose_t.shape[0] != 3:
                raise ValueError(f"expected pose shape (3,), got {tuple(pose_t.shape)}")
            return pose_t.unsqueeze(0)
        if pose_t.ndim == 2 and pose_t.shape == (3, 1):
            return pose_t.squeeze(1).unsqueeze(0)
        if pose_t.ndim != 2 or pose_t.shape[1] != 3:
            raise ValueError(
                "expected pose shape (B, 3), (3,), or (3, 1), "
                f"got {tuple(pose_t.shape)}"
            )
        return pose_t

    def _normalize_z_bar(self, z_bar: Tensor) -> Tensor:
        z_bar_t = torch.as_tensor(
            z_bar,
            dtype=self.A.dtype,
            device=self.A.device,
        )
        if z_bar_t.ndim != 2 or z_bar_t.shape[1] != 8:
            raise ValueError(
                "expected z_bar shape (H, 8) ordered as "
                "[x, y, theta, s, P_f, P_l, P_psi, P_lpsi]"
            )
        if not torch.isfinite(z_bar_t).all():
            raise ValueError("z_bar values must be finite")
        return z_bar_t

    @staticmethod
    def _normalize_epsilon(epsilon, z_bar: Tensor) -> Tensor:
        epsilon_t = torch.as_tensor(
            epsilon,
            dtype=z_bar.dtype,
            device=z_bar.device,
        )
        if epsilon_t.ndim == 0:
            epsilon_t = epsilon_t.expand(z_bar.shape[0])
        elif epsilon_t.ndim != 1 or epsilon_t.shape[0] != z_bar.shape[0]:
            raise ValueError("epsilon must be a scalar or have shape (H,)")
        if not torch.isfinite(epsilon_t).all():
            raise ValueError("epsilon values must be finite")
        if ((epsilon_t <= 0.0) | (epsilon_t >= 1.0)).any():
            raise ValueError("epsilon values must lie strictly between 0 and 1")
        return epsilon_t

    def forward(self, pose: Tensor) -> tuple[Tensor, Tensor]:
        pose_t = self._normalize_pose(pose)
        if not torch.isfinite(pose_t).all():
            raise ValueError("pose values must be finite")

        if self.active_clusters == 0:
            batch_size = pose_t.shape[0]
            phi = torch.full(
                (batch_size,),
                1000.0,
                dtype=pose_t.dtype,
                device=pose_t.device,
            )
            gradient = torch.zeros(
                (batch_size, 3),
                dtype=pose_t.dtype,
                device=pose_t.device,
            )
            return phi, gradient

        return self.psdf(
            self.A[: self.active_clusters],
            self.B[: self.active_clusters],
            self.mask[: self.active_clusters],
            pose_t,
        )

    @torch.no_grad()
    def forward_mf(
        self,
        z_bar: Tensor,
        epsilon,
        d_min: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(phi, gradient, A_mf, c_mf)`` for a nominal horizon."""

        z_bar_t = self._normalize_z_bar(z_bar)
        epsilon_t = self._normalize_epsilon(epsilon, z_bar_t)
        d_min_value = float(d_min)
        if not torch.isfinite(
            torch.tensor(d_min_value, dtype=z_bar_t.dtype, device=z_bar_t.device)
        ):
            raise ValueError("d_min must be finite")

        if self.active_clusters == 0:
            batch_size = z_bar_t.shape[0]
            phi = torch.full(
                (batch_size,),
                1000.0,
                dtype=z_bar_t.dtype,
                device=z_bar_t.device,
            )
            gradient = torch.zeros(
                (batch_size, 3),
                dtype=z_bar_t.dtype,
                device=z_bar_t.device,
            )
            A_mf = torch.zeros_like(z_bar_t)
            return phi, gradient, A_mf, epsilon_t.clone()

        return self.psdf.forward_mf(
            self.A[: self.active_clusters],
            self.B[: self.active_clusters],
            self.mask[: self.active_clusters],
            z_bar_t,
            epsilon_t,
            d_min_value,
        )

    def clear_clusters(self) -> None:
        self._reset_buffers()

    def get_cluster_info(self) -> dict:
        info = {
            "active_clusters": self.active_clusters,
            "K_max": self.K_max,
            "E_max": self.E_max,
            "total_active_edges": int(self.mask[: self.active_clusters].sum().item()),
        }
        for cluster_index in range(self.active_clusters):
            info[f"cluster_{cluster_index}_edges"] = int(
                self.mask[cluster_index].sum().item()
            )
        return info
