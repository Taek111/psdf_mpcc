"""Fresh signed-distance Gaussian surrogates for fixed obstacle identities.

Offline geometry only. A cluster fixes an obstacle, NOT a vertex/edge primitive:
the closest pair or penetration branch within that obstacle may change. These
values are not native optimized-stage chance residuals or MF feature sums.
"""

from __future__ import annotations

import numpy as np
import torch


def evaluate_cluster_risks(wrapper, poses, xi, d_min, epsilon=0.1):
    """Evaluate every obstacle at each actual pose and common local covariance.

    ``xi`` is [log L_ff, log L_ll, log L_tt, L_tl]. Per-row broadcasting uses
    exactly the same covariance for every obstacle. Signed PSDF remains signed
    through contact/penetration; its Gaussian transform is an algebraic local
    surrogate there, not a certified collision probability. ``mf_domain_active``
    explicitly marks the archived MF gate; no MF risk is computed by this helper.
    """
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    xi = np.broadcast_to(np.asarray(xi, dtype=float), (len(poses), 4))
    if poses.shape[1] != 3 or not np.isfinite(poses).all() or not np.isfinite(xi).all():
        raise ValueError("Expected finite poses (N,3) and xi (N,4) or (4,)")
    with torch.no_grad():
        phi_t, gradient_t = wrapper.forward_clusters(
            torch.as_tensor(poses, dtype=wrapper.A.dtype, device=wrapper.A.device))
    phi = phi_t.detach().cpu().numpy().astype(float)
    gradient = gradient_t.detach().cpu().numpy().astype(float)
    if phi.shape[1] == 0 or not np.isfinite(phi).all() or not np.isfinite(gradient).all():
        raise ValueError("Expected at least one obstacle and finite geometry")
    c, s = np.cos(poses[:, None, 2]), np.sin(poses[:, None, 2])
    q = np.stack((c * gradient[..., 0] + s * gradient[..., 1],
                  -s * gradient[..., 0] + c * gradient[..., 1],
                  gradient[..., 2]), axis=-1)
    variance = (np.exp(xi[:, None, 0]) * q[..., 0]) ** 2
    variance += (np.exp(xi[:, None, 1]) * q[..., 1] + xi[:, None, 3] * q[..., 2]) ** 2
    variance += (np.exp(xi[:, None, 2]) * q[..., 2]) ** 2
    sigma = np.sqrt(variance)
    z = np.divide(d_min - phi, sigma, out=np.zeros_like(phi), where=sigma > 0)
    risk = (.5 * torch.special.erfc(-torch.as_tensor(z, dtype=torch.float64)
                                    / np.sqrt(2))).numpy()
    risk[sigma == 0] = (phi[sigma == 0] < d_min).astype(float)
    active = phi.argmin(axis=1)
    return dict(phi=phi, world_gradient=gradient, local_q=q, sigma=sigma,
                probability_surrogate=risk, residual=epsilon - risk,
                active_cluster=active, separated=phi > 0,
                mf_domain_active=phi.min(axis=1) > d_min)


def track_two_pose_clusters(wrapper, pose_before, pose_after, xi1, d_min,
                            epsilon=0.1):
    """Exact old/new obstacle decomposition with ONE fixed xi1 at both poses.

    p_new(after)-p_old(before) = within_old_change + obstacle_selection_change.
    The selection term compares old/new obstacles at the SAME next pose and
    covariance. The before-pose evaluation intentionally uses xi1 as a common
    reference, not the covariance belonging to its previous recorded interval.
    """
    ev = evaluate_cluster_risks(wrapper, np.stack((pose_before, pose_after)),
                                xi1, d_min, epsilon)
    old, new = (int(k) for k in ev['active_cluster'])
    p = ev['probability_surrogate']
    within = float(p[1, old] - p[0, old])
    selection = float(p[1, new] - p[1, old])
    total = float(p[1, new] - p[0, old])
    if not np.isclose(total, within + selection, atol=1e-14, rtol=0):
        raise AssertionError('Probability decomposition failed')
    return dict(old_cluster=old, new_cluster=new, cluster_changed=old != new,
                old_before_probability=float(p[0, old]),
                old_after_probability=float(p[1, old]),
                new_before_probability=float(p[0, new]),
                new_after_probability=float(p[1, new]),
                within_old_probability_change=within,
                obstacle_selection_probability_change=selection,
                total_probability_change=total, evaluation=ev)
