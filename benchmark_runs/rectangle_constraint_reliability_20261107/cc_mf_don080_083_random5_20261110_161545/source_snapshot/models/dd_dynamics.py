"""Shared RK4 differential-drive dynamics with constant inputs per step."""

import numpy as np


def _rk4_pose_components(pose, v, omega, dt, cosine, sine):
    """Classical RK4, simplified for [v*cos(theta), v*sin(theta), omega]."""
    theta = pose[2]
    half_theta = theta + 0.5 * dt * omega
    next_theta = theta + dt * omega
    return (
        pose[0] + dt * v * (
            cosine(theta) + 4.0 * cosine(half_theta) + cosine(next_theta)
        ) / 6.0,
        pose[1] + dt * v * (
            sine(theta) + 4.0 * sine(half_theta) + sine(next_theta)
        ) / 6.0,
        next_theta,
    )


def symbolic_dd_step(pose, v, omega, dt):
    """One CasADi RK4 pose step, shared by CasADi MPC and CC controllers."""
    import casadi as ca

    return ca.vertcat(*_rk4_pose_components(pose, v, omega, dt, ca.cos, ca.sin))


class DifferentialDriveDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep):
        """Advance [x, y, yaw] with one RK4 step and constant [v, omega]."""
        return np.asarray(
            _rk4_pose_components(x, u[0], u[1], timestep, np.cos, np.sin),
            dtype=float,
        )

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return the same RK4 step as a CasADi state/input function."""
        import casadi as ca

        x_symbol = ca.SX.sym("x", 3)
        u_symbol = ca.SX.sym("u", 2)
        state_symbol_next = symbolic_dd_step(
            x_symbol, u_symbol[0], u_symbol[1], timestep,
        )
        return ca.Function(
            "DifferentialDrive_dynamics", [x_symbol, u_symbol], [state_symbol_next],
        )

    @staticmethod
    def nominal_safe_controller(x, timestep, v_last, amin, amax):
        """Apply the bounded braking input through the shared RK4 dynamics."""
        u_nom = np.zeros(2)
        a_nom = np.clip(-v_last / timestep, amin, amax)
        u_nom[0] = v_last + a_nom * timestep
        return DifferentialDriveDynamics.forward_dynamics(x, u_nom, timestep), u_nom

    @staticmethod
    def safe_dist(timestep, v_last, amax, dist_margin):
        """Return a safe distance outside which to ignore obstacles."""
        safe_ratio = 1.25
        brake_min_dist = (abs(v_last) + amax * timestep) ** 2 / (2 * amax) + dist_margin
        return safe_ratio * brake_min_dist + abs(v_last) * timestep + 0.5 * amax * timestep ** 2
