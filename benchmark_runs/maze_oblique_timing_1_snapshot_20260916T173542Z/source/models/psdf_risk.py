
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional


class PSDF(nn.Module):
    r"""Poly-Edge Distance Network (cluster-aware) with a local-frame risk tensor head.

    Outputs
    -------
    phi_k : (B,)
        Global signed distance (same as the original implementation).
    R_k   : (B,3,3)
        Risk tensor in the **robot-local pose perturbation frame**
        [dx_loc, dy_loc, dtheta].

    Notes
    -----
    * R_k is NOT in world-frame [dx_world, dy_world, dtheta].
      If your covariance is in world frame, convert it with
        T_w2l = diag(R(theta)^T, 1)
        Sigma_loc = T_w2l @ Sigma_world @ T_w2l^T
      and then use   tr(R_k @ Sigma_loc).
    * The signed-distance branch is unchanged from the original implementation:
      separation uses both candidate pools
        - robot vertex -> obstacle edge
        - obstacle endpoint -> robot edge
      and penetration uses the same SAT-style overlap path.
    * The risk head is revised to use **d_v only + hard local pooling**.
      For each obstacle cluster and each robot vertex, only the nearest obstacle
      edge is retained before aggregation. This suppresses raw edge-count bias
      while preserving additive multi-feature effects across clusters and robot
      vertices.
    """

    _GEOM_L2_EPS = 1e-8
    _SEP_DIST_SQ_FLOOR = 1e-12

    def __init__(
        self,
        verts: Tensor,
        eps: float = 1e-8,
        risk_beta: float = 20.0,
        risk_topk: Optional[int] = None,
    ):
        """
        verts        : (m,2) CCW convex robot footprint vertices in robot frame.
        eps          : numerical stability constant for risk-head near-zero cases.
        risk_beta    : proximity decay for locally pooled multi-feature risk.
        risk_topk    : optional top-k pooled (cluster, vertex) features used in R_k
                       after hard local pooling. None means use all pooled features.
        """
        super().__init__()

        V = verts.clone().detach()          # (m,2)
        S = torch.roll(V, -1, 0) - V        # (m,2)
        LS = (S ** 2).sum(1, keepdim=True) + eps

        n = F.normalize(torch.stack([-S[:, 1], S[:, 0]], 1), dim=1)
        c = -(n * V).sum(-1)

        self.register_buffer("V", V)
        self.register_buffer("S", S)
        self.register_buffer("LS", LS)
        self.register_buffer("n", n)
        self.register_buffer("c", c)

        proj = V @ n.T
        self.register_buffer("poly_min", proj.amin(0))
        self.register_buffer("poly_max", proj.amax(0))

        self.eps = float(eps)
        self.risk_beta = float(risk_beta)
        self.risk_topk = risk_topk
        self.inf = 1e12

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _p2seg_sq(P: Tensor, A: Tensor, v: Tensor, vL2: Tensor) -> Tensor:
        u = ((P - A) * v).sum(-1) / vL2.squeeze(-1).clamp_min(1e-12)
        t = u.clamp(0, 1)[..., None]
        Q = A + t * v
        return (P - Q).pow(2).sum(-1)

    @staticmethod
    def _p2seg_features(P: Tensor, A: Tensor, v: Tensor, vL2: Tensor):
        """Branch-free point-to-segment primitive returning distance and closest point."""
        u = ((P - A) * v).sum(-1) / vL2.squeeze(-1).clamp_min(1e-12)
        t = u.clamp(0, 1)[..., None]
        Q = A + t * v
        d_sq = (P - Q).pow(2).sum(-1)
        return d_sq, Q, t

    @staticmethod
    def _ray_inside(A_loc: Tensor, B_loc: Tensor, mask: Tensor) -> Tensor:
        Ay, By = A_loc[..., 1], B_loc[..., 1]
        Ax, Bx = A_loc[..., 0], B_loc[..., 0]
        crosses = ((Ay > 0) ^ (By > 0))
        x_int = Ax + (-Ay) * (Bx - Ax) / (By - Ay + 1e-12)
        crosses = crosses & (x_int > 0)
        crosses = crosses & mask.unsqueeze(0)
        return (crosses.sum(-1) & 1).bool()

    def _q_from_support_pair(self, s_robot: Tensor, o_obs: Tensor):
        """Build q = [n_x, n_y, n^T J s]^T for local pose perturbation [dx_loc, dy_loc, dtheta]."""
        r = s_robot - o_obs
        d = r.norm(dim=-1)
        safe_d = torch.where(d > self.eps, d, torch.ones_like(d))
        n = torch.where((d > self.eps).unsqueeze(-1), r / safe_d.unsqueeze(-1), torch.zeros_like(r))
        Js = torch.stack([-s_robot[..., 1], s_robot[..., 0]], dim=-1)
        q_theta = (n * Js).sum(-1)
        q = torch.cat([n, q_theta.unsqueeze(-1)], dim=-1)
        return d, q

    def _pool_dv_hard_local(
        self,
        d_v_sq_full: Tensor,
        Q_obs_v: Tensor,
        mask: Tensor,
    ):
        """Hard local pooling over the obstacle-edge axis.

        For each batch b, obstacle cluster k, and robot vertex i, select the
        nearest valid obstacle edge e* = argmin_e d_v_sq_full[b,k,e,i], then
        keep only the corresponding support pair (V_i, Q_obs_v[b,k,e*,i]).
        """
        Bsz, K, _, m = d_v_sq_full.shape

        valid_edge = mask.unsqueeze(0)  # (1,K,E)
        d_v_sq_masked = d_v_sq_full.masked_fill(~valid_edge.unsqueeze(-1), self.inf)

        # argmin over obstacle-edge axis: no sorting, only a reduction.
        idx_edge = d_v_sq_masked.argmin(dim=2)  # (B,K,m)

        gather_q_idx = idx_edge.unsqueeze(2).unsqueeze(-1).expand(-1, -1, 1, -1, 2)
        o_sel = torch.gather(Q_obs_v, 2, gather_q_idx).squeeze(2)  # (B,K,m,2)

        s_sel = self.V.view(1, 1, m, 2).expand(Bsz, K, m, 2)
        d_sel, q_sel = self._q_from_support_pair(s_sel, o_sel)

        valid_cluster = mask.any(dim=1).view(1, K).expand(Bsz, K)  # (B,K)
        valid_sel = valid_cluster.unsqueeze(-1).expand(Bsz, K, m)  # (B,K,m)

        d_sel = torch.where(valid_sel, d_sel, torch.zeros_like(d_sel))
        q_sel = torch.where(valid_sel.unsqueeze(-1), q_sel, torch.zeros_like(q_sel))
        return d_sel, q_sel, valid_sel

    def _aggregate_local_pooled_risk(self, d_local: Tensor, q_local: Tensor, valid_local: Tensor) -> Tensor:
        """Aggregate locally pooled (cluster, vertex) features into (B,3,3)."""
        alpha = torch.exp(-self.risk_beta * d_local) * valid_local.to(d_local.dtype)
        qq_flat = (q_local.unsqueeze(-1) * q_local.unsqueeze(-2)).reshape(q_local.shape[0], -1, 3, 3)
        alpha_flat = alpha.reshape(alpha.shape[0], -1)

        if self.risk_topk is not None and self.risk_topk < alpha_flat.shape[1]:
            vals, idx = torch.topk(alpha_flat, k=self.risk_topk, dim=1)
            gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
            R_sel = torch.gather(qq_flat, 1, gather_idx)
            return (vals.unsqueeze(-1).unsqueeze(-1) * R_sel).sum(dim=1)

        return (alpha_flat.unsqueeze(-1).unsqueeze(-1) * qq_flat).sum(dim=1)

    @staticmethod
    def local_pose_transform(poses: Tensor) -> Tensor:
        """T_w2l for pose perturbations [dx, dy, dtheta].

        If delta_x_world = [dx_world, dy_world, dtheta], then
            delta_x_loc = T_w2l @ delta_x_world.
        """
        cos, sin = poses[:, 2].cos(), poses[:, 2].sin()
        T = poses.new_zeros((poses.shape[0], 3, 3))
        T[:, 0, 0] = cos
        T[:, 0, 1] = sin
        T[:, 1, 0] = -sin
        T[:, 1, 1] = cos
        T[:, 2, 2] = 1.0
        return T

    @classmethod
    def cov_world_to_local(cls, poses: Tensor, Sigma_world: Tensor) -> Tensor:
        T = cls.local_pose_transform(poses)
        return T @ Sigma_world @ T.transpose(1, 2)

    @classmethod
    def risk_local_to_world(cls, poses: Tensor, R_local: Tensor) -> Tensor:
        T = cls.local_pose_transform(poses)
        return T.transpose(1, 2) @ R_local @ T

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        A: Tensor,      # (K,E,2)
        B: Tensor,      # (K,E,2)
        mask: Tensor,   # (K,E) bool
        poses: Tensor,  # (B,3)
    ):
        """Return (phi_k, R_k) where R_k is in local pose coordinates [dx_loc, dy_loc, dtheta]."""
        Bsz, K, E = poses.size(0), *A.shape[:2]
        m = self.V.size(0)

        # 1) world -> robot-local transform ---------------------------------
        cos, sin = poses[:, 2].cos(), poses[:, 2].sin()
        R_w2l_row = torch.stack(
            [torch.stack([cos, sin], 1), torch.stack([-sin, cos], 1)], 1
        )  # row-vector world->local compatible matrix, shape (B,2,2)

        A_rel = A.unsqueeze(0) - poses[:, :2].view(Bsz, 1, 1, 2)
        B_rel = B.unsqueeze(0) - poses[:, :2].view(Bsz, 1, 1, 2)
        A_loc = torch.bmm(A_rel.view(Bsz, -1, 2), R_w2l_row.transpose(1, 2)).view(Bsz, K, E, 2)
        B_loc = torch.bmm(B_rel.view(Bsz, -1, 2), R_w2l_row.transpose(1, 2)).view(Bsz, K, E, 2)

        valid_edge = mask.unsqueeze(0)  # (1,K,E)
        A_loc_masked = A_loc.masked_fill(~valid_edge.unsqueeze(-1), self.inf)
        B_loc_masked = B_loc.masked_fill(~valid_edge.unsqueeze(-1), self.inf)
        v_obs = B_loc - A_loc
        # Keep the SDF path numerically aligned with models/psdf.py.
        L2_obs = (v_obs ** 2).sum(-1, keepdim=True) + self._GEOM_L2_EPS

        # ------------------- distance part (separation) ------------------- #
        P_v = self.V.view(1, 1, 1, m, 2)

        # pool A: robot vertices -> obstacle edges --------------------------
        d_v_sq_full, Q_obs_v, _ = self._p2seg_features(
            P_v,
            A_loc.unsqueeze(3),
            v_obs.unsqueeze(3),
            L2_obs.unsqueeze(3),
        )

        d_v_sq = d_v_sq_full.masked_fill(~valid_edge.unsqueeze(-1), self.inf)
        d_v_sq_min = d_v_sq.amin(2).amin(2)

        # pool B: obstacle endpoints -> robot edges -------------------------
        Pts = torch.cat([A_loc, B_loc], 2)
        mask2 = torch.cat([mask, mask], dim=1)
        valid_pt = mask2.unsqueeze(0)

        d_e_sq_full = self._p2seg_sq(
            Pts.unsqueeze(3),
            self.V.view(1, 1, 1, m, 2),
            self.S.view(1, 1, 1, m, 2),
            self.LS.view(1, 1, 1, m, 1),
        )

        d_e_sq = d_e_sq_full.masked_fill(~valid_pt.unsqueeze(-1), self.inf)
        d_e_sq_min = d_e_sq.amin(3).amin(2)

        sep_dist = (
            torch.minimum(d_v_sq_min, d_e_sq_min)
            .clamp_min(self._SEP_DIST_SQ_FLOOR)
            .sqrt()
        )

        # ------------------- overlap part (penetration) ------------------- #
        ends_proj = (Pts @ self.n.T)
        mask_proj = valid_pt.unsqueeze(-1)
        ends_min = ends_proj.masked_fill(~mask_proj, self.inf).amin(2)
        ends_max = ends_proj.masked_fill(~mask_proj, -self.inf).amax(2)
        ov_poly = torch.minimum(ends_max - self.poly_min, self.poly_max - ends_min)

        v_obs_masked = v_obs.masked_fill(~valid_edge.unsqueeze(-1), 0.0)
        n_seg = F.normalize(
            torch.stack([-v_obs_masked[..., 1], v_obs_masked[..., 0]], -1),
            dim=-1,
            eps=1e-12,
        )

        proj_poly_seg = (n_seg @ self.V.T)
        proj_poly_seg_masked = proj_poly_seg.masked_fill(~valid_edge.unsqueeze(-1), self.inf)
        poly_min_seg = proj_poly_seg_masked.amin(-1)
        proj_poly_seg_masked = proj_poly_seg.masked_fill(~valid_edge.unsqueeze(-1), -self.inf)
        poly_max_seg = proj_poly_seg_masked.amax(-1)

        mask_all = mask.unsqueeze(0).unsqueeze(2)
        proj_all = (n_seg.unsqueeze(3) * A_loc_masked.unsqueeze(2)).sum(-1)
        seg_min = proj_all.masked_fill(~mask_all, self.inf).amin(-1)
        seg_max = proj_all.masked_fill(~mask_all, -self.inf).amax(-1)

        ov_seg = torch.minimum(seg_max - poly_min_seg, poly_max_seg - seg_min)
        ov_seg = ov_seg.masked_fill(~valid_edge, self.inf)

        all_ov = torch.cat([ov_poly, ov_seg], dim=2)
        eps_sat = 1e-6
        separated = (all_ov < -eps_sat).any(2)

        beta_pen = 100.0
        penetration = -torch.logsumexp(-beta_pen * all_ov.clamp_min(0), 2) / beta_pen

        inside = self._ray_inside(A_loc, B_loc, mask)
        separated = separated & (~inside)

        signed_cluster = torch.where(
            separated,
            sep_dist,
            torch.where(inside, -sep_dist, -penetration),
        )
        phi_k = signed_cluster.amin(1)

        # --------------------------- risk head ---------------------------- #
        # d_v only + hard local pooling:
        #   for each (cluster k, robot vertex i), keep only the nearest obstacle edge.
        d_v_pool, q_v_pool, valid_v_pool = self._pool_dv_hard_local(
            d_v_sq_full=d_v_sq_full,
            Q_obs_v=Q_obs_v,
            mask=mask,
        )
        R_k = self._aggregate_local_pooled_risk(d_v_pool, q_v_pool, valid_v_pool)

        return phi_k, R_k
