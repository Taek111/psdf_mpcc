import torch
import torch.nn as nn

from models.psdf_risk import PSDF as RiskPSDF


class psdfRiskWrapper(nn.Module):
    """
    Risk-aware PSDF wrapper.

    `forward()` keeps the same scalar signed-distance interface used by
    `RealTimeL4CasADi`, while `forward_with_risk()` exposes `(phi, R)` for the
    external risk-margin update in RMPCC.
    """

    def __init__(
        self,
        verts,
        K_max: int,
        E_max: int,
        device: str = "cpu",
        risk_beta: float = 20.0,
        risk_topk=None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.psdf = RiskPSDF(
            verts,
            risk_beta=risk_beta,
            risk_topk=risk_topk,
        ).to(self.device)

        self.K_max = K_max
        self.E_max = E_max

        self.register_buffer("A", torch.zeros(K_max, E_max, 2, device=self.device))
        self.register_buffer("B", torch.zeros(K_max, E_max, 2, device=self.device))
        self.register_buffer("mask", torch.zeros(K_max, E_max, dtype=torch.bool, device=self.device))

        self.active_clusters: int = 0

    @torch.no_grad()
    def update_edge_clusters(self, clusters_A, clusters_B, clusters_mask=None):
        if isinstance(clusters_A, list):
            K_roi = len(clusters_A)
            assert K_roi <= self.K_max, f"ROI clusters ({K_roi}) exceeds K_max ({self.K_max})"

            self.A.zero_()
            self.B.zero_()
            self.mask.zero_()

            for k in range(K_roi):
                E_k = clusters_A[k].shape[0]
                assert E_k <= self.E_max, f"Cluster {k} edge count ({E_k}) exceeds E_max ({self.E_max})"

                self.A[k, :E_k].copy_(clusters_A[k].to(self.device))
                self.B[k, :E_k].copy_(clusters_B[k].to(self.device))
                self.mask[k, :E_k] = True

            self.active_clusters = K_roi
            return

        K_roi, E_roi = clusters_A.shape[:2]
        assert K_roi <= self.K_max, f"ROI clusters ({K_roi}) exceeds K_max ({self.K_max})"
        assert E_roi <= self.E_max, f"ROI edges per cluster ({E_roi}) exceeds E_max ({self.E_max})"

        self.A.zero_()
        self.B.zero_()
        self.mask.zero_()

        self.active_clusters = K_roi
        self.A[:K_roi, :E_roi].copy_(clusters_A.to(self.device))
        self.B[:K_roi, :E_roi].copy_(clusters_B.to(self.device))

        if clusters_mask is not None:
            self.mask[:K_roi, :E_roi].copy_(clusters_mask.to(self.device))
            return

        edge_valid = (clusters_A.abs().sum(dim=-1) > 1e-6) | (clusters_B.abs().sum(dim=-1) > 1e-6)
        self.mask[:K_roi, :E_roi].copy_(edge_valid.to(self.device))

    def _normalize_pose_input(self, pose):
        pose_t = pose.to(self.device)
        if pose_t.ndim == 1:
            return pose_t.unsqueeze(0)
        if pose_t.ndim == 2 and pose_t.size(0) == 3 and pose_t.size(1) == 1:
            return pose_t.squeeze(1).unsqueeze(0)
        if pose_t.ndim != 2 or pose_t.size(-1) != 3:
            raise ValueError(f"Expected pose shape [B, 3] or [3], got {pose_t.shape}")
        return pose_t

    def forward_with_risk(self, pose):
        pose_t = self._normalize_pose_input(pose)

        if self.active_clusters == 0:
            batch_size = pose_t.size(0)
            phi = torch.full((batch_size,), 1000.0, device=self.device, dtype=pose_t.dtype)
            risk = torch.zeros((batch_size, 3, 3), device=self.device, dtype=pose_t.dtype)
            return phi, risk

        A = self.A[:self.active_clusters]
        B = self.B[:self.active_clusters]
        mask = self.mask[:self.active_clusters]
        return self.psdf(A, B, mask, pose_t)

    def forward_with_local_gradient(self, pose):
        pose_t = self._normalize_pose_input(pose)

        if self.active_clusters == 0:
            batch_size = pose_t.size(0)
            phi = torch.full((batch_size,), 1000.0, device=self.device, dtype=pose_t.dtype)
            grad = torch.zeros((batch_size, 3), device=self.device, dtype=pose_t.dtype)
            return phi, grad

        A = self.A[:self.active_clusters]
        B = self.B[:self.active_clusters]
        mask = self.mask[:self.active_clusters]

        with torch.enable_grad():
            pose_var = pose_t.detach().clone().requires_grad_(True)
            phi, _ = self.psdf(A, B, mask, pose_var)
            grad_world = torch.autograd.grad(
                phi.sum(),
                pose_var,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]
            T_w2l = self.psdf.local_pose_transform(pose_var)
            grad_local = torch.bmm(T_w2l, grad_world.unsqueeze(-1)).squeeze(-1)

        return phi.detach(), grad_local.detach()

    def forward(self, pose):
        phi, _ = self.forward_with_risk(pose)
        if phi.ndim == 1:
            phi = phi.unsqueeze(1)
        return phi

    def clear_clusters(self):
        self.A.zero_()
        self.B.zero_()
        self.mask.zero_()
        self.active_clusters = 0

    def get_cluster_info(self):
        info = {
            "active_clusters": self.active_clusters,
            "K_max": self.K_max,
            "E_max": self.E_max,
            "total_active_edges": self.mask[:self.active_clusters].sum().item(),
        }
        for k in range(self.active_clusters):
            info[f"cluster_{k}_edges"] = self.mask[k].sum().item()
        return info
