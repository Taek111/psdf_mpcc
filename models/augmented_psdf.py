"""Augmented polygon-set distance field with an analytic pose gradient.

The public forward path returns the hard-SAT signed distance and the analytic
world-frame gradient of its active branch.  ``forward_mf`` additionally
returns the batched affine coefficients for the Boole multi-feature chance
constraint used by RMPCC.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PSDF(nn.Module):
    """Cluster-aware hard-SAT PSDF with an analytic world-frame gradient."""

    def __init__(self, verts: Tensor, eps: float = 1e-8):
        super().__init__()

        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError("verts must have shape (m, 2) with m >= 3")
        if not verts.is_floating_point():
            raise TypeError("verts must be a floating-point tensor")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a positive finite scalar")

        V = verts.clone().detach()
        S = torch.roll(V, -1, 0) - V
        LS = (S**2).sum(1, keepdim=True) + eps

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
        self.inf = 1e12

    @staticmethod
    def _p2seg(
        P: Tensor,
        A: Tensor,
        v: Tensor,
        vL2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return squared point-to-segment distance and the closest point."""

        u = ((P - A) * v).sum(-1) / vL2.squeeze(-1).clamp_min(1e-12)
        t = u.clamp(0, 1)[..., None]
        Q = A + t * v
        return (P - Q).pow(2).sum(-1), Q

    @staticmethod
    def _ray_inside(A_loc: Tensor, B_loc: Tensor, mask: Tensor) -> Tensor:
        """Apply the odd-even ray rule to the body-frame origin."""

        Ay, By = A_loc[..., 1], B_loc[..., 1]
        Ax, Bx = A_loc[..., 0], B_loc[..., 0]

        crosses = (Ay > 0) ^ (By > 0)
        x_int = Ax + (-Ay) * (Bx - Ax) / (By - Ay + 1e-12)
        crosses = crosses & (x_int > 0)
        crosses = crosses & mask.unsqueeze(0)
        return (crosses.sum(-1) & 1).bool()

    @staticmethod
    def _compute_A(
        d: Tensor,
        q: Tensor,
        feature_mask: Tensor,
        z_bar: Tensor,
        epsilon: Tensor,
        d_min: float,
    ) -> tuple[Tensor, Tensor]:
        """Compute the MF Jacobian and its nominal nonlinear residual."""

        q_f, q_l, q_theta = q.unbind(-1)
        sigma_ff = z_bar[:, 4, None]
        sigma_ll = z_bar[:, 5, None]
        sigma_tt = z_bar[:, 6, None]
        sigma_lt = z_bar[:, 7, None]

        sigma_sq = (
            q_f.square() * sigma_ff
            + q_l.square() * sigma_ll
            + q_theta.square() * sigma_tt
            + 2.0 * q_l * q_theta * sigma_lt
        )

        # Keep detailed value validation on CPU, where scalar inspection does
        # not synchronize a device stream.  The CUDA production path relies on
        # the documented positive-variance input contract and stays fully
        # asynchronous.  Invalid CUDA variances still propagate non-finite
        # values through the unmodified tensor formulas below.
        if sigma_sq.device.type == "cpu":
            bad = (~torch.isfinite(sigma_sq) | (sigma_sq <= 0.0)) & feature_mask
            if bad.any():
                bad_index = bad.nonzero(as_tuple=False)[0].tolist()
                raise ValueError(
                    "every valid feature must have positive finite projected "
                    f"variance; first invalid (stage, feature) is {tuple(bad_index)}"
                )

        safe_sigma_sq = torch.where(feature_mask, sigma_sq, torch.ones_like(sigma_sq))
        sigma = safe_sigma_sq.sqrt()
        zeta = (d_min - d) / sigma

        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(zeta * inv_sqrt_2))
        pdf = torch.exp(-0.5 * zeta.square()) * inv_sqrt_2pi

        cdf = torch.where(feature_mask, cdf, torch.zeros_like(cdf))
        pdf = torch.where(feature_mask, pdf, torch.zeros_like(pdf))
        h_bar = epsilon - cdf.sum(-1)

        pose_body = ((pdf / sigma)[..., None] * q).sum(1)
        theta = z_bar[:, 2]
        cos_theta, sin_theta = theta.cos(), theta.sin()
        pose_world = torch.stack(
            [
                cos_theta * pose_body[:, 0] - sin_theta * pose_body[:, 1],
                sin_theta * pose_body[:, 0] + cos_theta * pose_body[:, 1],
                pose_body[:, 2],
            ],
            dim=-1,
        )

        covariance_basis = torch.stack(
            [
                q_f.square(),
                q_l.square(),
                q_theta.square(),
                2.0 * q_l * q_theta,
            ],
            dim=-1,
        )
        covariance_scale = pdf * zeta / (2.0 * safe_sigma_sq)
        covariance_block = (covariance_scale[..., None] * covariance_basis).sum(1)

        A_mf = torch.zeros_like(z_bar)
        A_mf[:, :3] = pose_world
        A_mf[:, 4:] = covariance_block
        return A_mf, h_bar

    @staticmethod
    def _compute_c(A_mf: Tensor, h_bar: Tensor, z_bar: Tensor) -> Tensor:
        """Compute the affine constant for ``A_mf @ z + c_mf``."""

        return h_bar - (A_mf * z_bar).sum(-1)

    @torch.no_grad()
    def _forward_impl(
        self,
        A: Tensor,
        B: Tensor,
        mask: Tensor,
        poses: Tensor,
        return_features: bool = False,
    ):
        """Evaluate the PSDF and its analytic world-frame pose gradient.

        Args:
            A, B: Obstacle segment start/end points with shape ``(K, E, 2)``.
            mask: Valid-segment mask with shape ``(K, E)``.
            poses: Robot poses with shape ``(H, 3)`` ordered as
                ``[x, y, theta]``.

        Returns:
            ``phi (H,)`` and world-frame ``g (H,3)``.
        """

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
        # Tensor-value checks are CPU-only.  Evaluating these predicates in a
        # Python ``if`` on CUDA would force a host/device synchronization on
        # every real-time call.  Shape, dtype, and device metadata checks above
        # remain active for both CPU and CUDA without inspecting device data.
        if A.device.type == "cpu":
            if not mask.any(-1).all():
                raise ValueError(
                    "every obstacle cluster must contain at least one valid edge"
                )
            if not all(torch.isfinite(t).all() for t in expected_tensors):
                raise ValueError("all floating-point inputs must be finite")

        H, K, E = poses.shape[0], A.shape[0], A.shape[1]
        m = self.V.shape[0]

        # World-to-body transform, kept identical to the inherited PSDF.
        cos_theta, sin_theta = poses[:, 2].cos(), poses[:, 2].sin()
        R = torch.stack(
            [
                torch.stack([cos_theta, sin_theta], 1),
                torch.stack([-sin_theta, cos_theta], 1),
            ],
            1,
        )
        translation = poses[:, :2].reshape(H, 1, 1, 2)
        A_rel = A.unsqueeze(0) - translation
        B_rel = B.unsqueeze(0) - translation
        A_loc = torch.bmm(A_rel.reshape(H, -1, 2), R.transpose(1, 2)).reshape(
            H, K, E, 2
        )
        B_loc = torch.bmm(B_rel.reshape(H, -1, 2), R.transpose(1, 2)).reshape(
            H, K, E, 2
        )

        edge_mask = mask.unsqueeze(0)
        A_loc_masked = A_loc.masked_fill(~edge_mask.unsqueeze(-1), self.inf)
        v_obs = B_loc - A_loc
        L2_obs = (v_obs**2).sum(-1, keepdim=True) + 1e-8

        # Shared bidirectional point-to-segment candidates.
        P_v = self.V.reshape(1, 1, 1, m, 2)
        d_v_sq, Q_v = self._p2seg(
            P_v,
            A_loc.unsqueeze(3),
            v_obs.unsqueeze(3),
            L2_obs.unsqueeze(3),
        )
        d_v_sq = d_v_sq.masked_fill(~edge_mask.unsqueeze(-1), self.inf)

        Pts = torch.cat([A_loc, B_loc], 2)
        mask2 = torch.cat([mask, mask], 1)
        d_e_sq, Q_e = self._p2seg(
            Pts.unsqueeze(3),
            self.V.reshape(1, 1, 1, m, 2),
            self.S.reshape(1, 1, 1, m, 2),
            self.LS.reshape(1, 1, 1, m, 1),
        )
        d_e_sq = d_e_sq.masked_fill(~mask2.unsqueeze(0).unsqueeze(-1), self.inf)

        # Pool R->O once per robot vertex and O->R once per robot edge.
        idx_v = d_v_sq.argmin(2)
        gather_v = idx_v.unsqueeze(2)
        d_v = torch.gather(d_v_sq, 2, gather_v).squeeze(2)
        Q_v_star = torch.gather(
            Q_v,
            2,
            gather_v.unsqueeze(-1).expand(H, K, 1, m, 2),
        ).squeeze(2)
        c_R_v = self.V.reshape(1, 1, m, 2).expand(H, K, m, 2)

        idx_e = d_e_sq.argmin(2)
        gather_e = idx_e.unsqueeze(2)
        d_e = torch.gather(d_e_sq, 2, gather_e).squeeze(2)
        Q_e_star = torch.gather(
            Q_e,
            2,
            gather_e.unsqueeze(-1).expand(H, K, 1, m, 2),
        ).squeeze(2)
        Pts_star = torch.gather(
            Pts,
            2,
            idx_e.unsqueeze(-1).expand(H, K, m, 2),
        )

        feature_sq = torch.cat([d_v, d_e], 2)
        feature_d = feature_sq.clamp_min(0.0).sqrt()
        c_R = torch.cat([c_R_v, Q_e_star], 2)
        c_O = torch.cat([Q_v_star, Pts_star], 2)
        feature_mask = (
            mask.any(-1).reshape(1, K, 1).expand(H, K, 2 * m)
            & torch.isfinite(feature_sq)
            & (feature_sq > 0.0)
        )

        delta = c_R - c_O
        safe_d = torch.where(feature_mask, feature_d, torch.ones_like(feature_d))
        direction = delta / safe_d.unsqueeze(-1)
        direction = torch.where(
            feature_mask.unsqueeze(-1), direction, torch.zeros_like(direction)
        )
        J_c_R = torch.stack([-c_R[..., 1], c_R[..., 0]], -1)
        q_feature = torch.cat(
            [direction, (direction * J_c_R).sum(-1, keepdim=True)], -1
        )

        # The active separation feature is the first minimum in the same
        # bidirectional candidate set used by the inherited scalar PSDF.
        idx_sep = feature_sq.argmin(2)
        sep_sq = torch.gather(feature_sq, 2, idx_sep.unsqueeze(-1)).squeeze(-1)
        q_sep = torch.gather(
            q_feature,
            2,
            idx_sep.unsqueeze(-1).unsqueeze(-1).expand(H, K, 1, 3),
        ).squeeze(2)
        sep_dist = sep_sq.clamp_min(1e-12).sqrt()

        # SAT axes fixed to the robot polygon.
        ends_proj = Pts @ self.n.T
        endpoint_mask = mask2.unsqueeze(0).unsqueeze(-1)
        ends_min_values = ends_proj.masked_fill(~endpoint_mask, self.inf)
        ends_max_values = ends_proj.masked_fill(~endpoint_mask, -self.inf)
        ends_min, idx_ends_min = ends_min_values.min(2)
        ends_max, idx_ends_max = ends_max_values.max(2)

        ov_poly_side_1 = ends_max - self.poly_min
        ov_poly_side_2 = self.poly_max - ends_min
        ov_poly = torch.minimum(ov_poly_side_1, ov_poly_side_2)

        y_min = torch.gather(
            Pts,
            2,
            idx_ends_min.unsqueeze(-1).expand(H, K, m, 2),
        )
        y_max = torch.gather(
            Pts,
            2,
            idx_ends_max.unsqueeze(-1).expand(H, K, m, 2),
        )
        axes_poly = self.n.reshape(1, 1, m, 2).expand(H, K, m, 2)
        J_y_min = torch.stack([-y_min[..., 1], y_min[..., 0]], -1)
        J_y_max = torch.stack([-y_max[..., 1], y_max[..., 0]], -1)
        q_end_min = torch.cat(
            [-axes_poly, -(axes_poly * J_y_min).sum(-1, keepdim=True)], -1
        )
        q_end_max = torch.cat(
            [-axes_poly, -(axes_poly * J_y_max).sum(-1, keepdim=True)], -1
        )
        q_ov_poly = torch.where(
            (ov_poly_side_1 <= ov_poly_side_2).unsqueeze(-1),
            q_end_max,
            -q_end_min,
        )
        valid_poly_axis = mask.any(-1).reshape(1, K, 1)
        q_ov_poly = torch.where(
            valid_poly_axis.unsqueeze(-1), q_ov_poly, torch.zeros_like(q_ov_poly)
        )
        delta_poly = ov_poly.masked_fill(~valid_poly_axis, self.inf)

        # SAT axes attached to obstacle edges.  ``proj_all`` intentionally
        # follows the inherited implementation and projects every valid A end.
        v_obs_masked = v_obs.masked_fill(~edge_mask.unsqueeze(-1), 0.0)
        n_seg = F.normalize(
            torch.stack([-v_obs_masked[..., 1], v_obs_masked[..., 0]], -1),
            dim=-1,
            eps=1e-12,
        )
        proj_poly_seg = n_seg @ self.V.T
        poly_min_values = proj_poly_seg.masked_fill(
            ~edge_mask.unsqueeze(-1), self.inf
        )
        poly_max_values = proj_poly_seg.masked_fill(
            ~edge_mask.unsqueeze(-1), -self.inf
        )
        poly_min_seg, idx_poly_min = poly_min_values.min(-1)
        poly_max_seg, idx_poly_max = poly_max_values.max(-1)

        mask_all = mask.unsqueeze(0).unsqueeze(2)
        proj_all = (n_seg.unsqueeze(3) * A_loc_masked.unsqueeze(2)).sum(-1)
        seg_min = proj_all.masked_fill(~mask_all, self.inf).amin(-1)
        seg_max = proj_all.masked_fill(~mask_all, -self.inf).amax(-1)

        ov_seg_side_1 = seg_max - poly_min_seg
        ov_seg_side_2 = poly_max_seg - seg_min
        ov_seg = torch.minimum(ov_seg_side_1, ov_seg_side_2)
        ov_seg = ov_seg.masked_fill(~edge_mask, self.inf)

        V_min = self.V[idx_poly_min]
        V_max = self.V[idx_poly_max]
        J_V_min = torch.stack([-V_min[..., 1], V_min[..., 0]], -1)
        J_V_max = torch.stack([-V_max[..., 1], V_max[..., 0]], -1)
        theta_poly_min = (n_seg * J_V_min).sum(-1, keepdim=True)
        theta_poly_max = (n_seg * J_V_max).sum(-1, keepdim=True)
        zeros = torch.zeros_like(theta_poly_min)
        q_seg_projection = torch.cat([-n_seg, zeros], -1)
        q_poly_min = torch.cat([torch.zeros_like(n_seg), theta_poly_min], -1)
        q_poly_max = torch.cat([torch.zeros_like(n_seg), theta_poly_max], -1)
        q_ov_seg = torch.where(
            (ov_seg_side_1 <= ov_seg_side_2).unsqueeze(-1),
            q_seg_projection - q_poly_min,
            q_poly_max - q_seg_projection,
        )
        q_ov_seg = torch.where(
            edge_mask.unsqueeze(-1), q_ov_seg, torch.zeros_like(q_ov_seg)
        )
        delta_seg = ov_seg.masked_fill(~edge_mask, self.inf)

        delta_all = torch.cat([delta_poly, delta_seg], 2)
        q_all = torch.cat([q_ov_poly, q_ov_seg], 2)
        eps_sat = 1e-6
        separated = (delta_all < -eps_sat).any(2)

        axis_star = delta_all.argmin(2)
        penetration = torch.gather(
            delta_all.clamp_min(0.0), 2, axis_star.unsqueeze(-1)
        ).squeeze(-1)
        q_pen = torch.gather(
            q_all,
            2,
            axis_star.unsqueeze(-1).unsqueeze(-1).expand(H, K, 1, 3),
        ).squeeze(2)

        signed_cluster = torch.where(
            separated,
            sep_dist,
            -penetration,
        )
        q_cluster = torch.where(
            separated.unsqueeze(-1),
            q_sep,
            -q_pen,
        )

        active_cluster = signed_cluster.argmin(1)
        phi = torch.gather(signed_cluster, 1, active_cluster.unsqueeze(-1)).squeeze(-1)
        q_active = torch.gather(
            q_cluster,
            1,
            active_cluster.unsqueeze(-1).unsqueeze(-1).expand(H, 1, 3),
        ).squeeze(1)
        g = torch.stack(
            [
                cos_theta * q_active[:, 0] - sin_theta * q_active[:, 1],
                sin_theta * q_active[:, 0] + cos_theta * q_active[:, 1],
                q_active[:, 2],
            ],
            -1,
        )

        if return_features:
            return (
                phi,
                g,
                feature_d.reshape(H, -1),
                q_feature.reshape(H, -1, 3),
                feature_mask.reshape(H, -1),
            )
        return phi, g

    @torch.no_grad()
    def forward(
        self,
        A: Tensor,
        B: Tensor,
        mask: Tensor,
        poses: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the scalar PSDF and its analytic world-frame gradient."""

        return self._forward_impl(A, B, mask, poses)

    @torch.no_grad()
    def forward_mf(
        self,
        A: Tensor,
        B: Tensor,
        mask: Tensor,
        z_bar: Tensor,
        epsilon,
        d_min: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate PSDF outputs and one MF affine row per stage.

        ``z_bar`` uses the augmented-state order
        ``[x, y, theta, s, P_f, P_l, P_psi, P_lpsi]``.  Feature identities,
        distances, gradients, and masks stay inside this tensor graph; only
        ``A_mf`` and ``c_mf`` need to be transferred to the OCP.
        """

        if z_bar.ndim != 2 or z_bar.shape[1] != 8:
            raise ValueError("z_bar must have shape (H, 8)")
        if not z_bar.is_floating_point():
            raise TypeError("z_bar must be a floating-point tensor")
        if z_bar.dtype != self.V.dtype:
            raise TypeError("z_bar and verts must use the same dtype")
        if z_bar.device != self.V.device:
            raise ValueError("z_bar and the PSDF module must use the same device")
        if z_bar.device.type == "cpu" and not torch.isfinite(z_bar).all():
            raise ValueError("z_bar values must be finite")

        epsilon_t = torch.as_tensor(
            epsilon,
            dtype=z_bar.dtype,
            device=z_bar.device,
        )
        if epsilon_t.ndim == 0:
            epsilon_t = epsilon_t.expand(z_bar.shape[0])
        elif epsilon_t.ndim != 1 or epsilon_t.shape[0] != z_bar.shape[0]:
            raise ValueError("epsilon must be a scalar or have shape (H,)")
        if epsilon_t.device.type == "cpu":
            if not torch.isfinite(epsilon_t).all():
                raise ValueError("epsilon values must be finite")
            if ((epsilon_t <= 0.0) | (epsilon_t >= 1.0)).any():
                raise ValueError("epsilon values must lie strictly between 0 and 1")

        d_min_value = float(d_min)
        if not math.isfinite(d_min_value):
            raise ValueError("d_min must be finite")

        phi, gradient, feature_d, q_feature, feature_mask = self._forward_impl(
            A,
            B,
            mask,
            z_bar[:, :3],
            return_features=True,
        )
        A_mf, h_bar = self._compute_A(
            feature_d,
            q_feature,
            feature_mask,
            z_bar,
            epsilon_t,
            d_min_value,
        )
        c_mf = self._compute_c(A_mf, h_bar, z_bar)
        return phi, gradient, A_mf, c_mf
