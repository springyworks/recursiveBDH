"""
Double Segmented Pendulum — Lagrangian mechanics simulation.

Two coupled segments with controllable torques. Provides the physical
system that the BDH constellation attempts to orchestrate, creating
a "neuromorphic ballet" of two interacting double pendulums.
"""

import numpy as np


class DoublePendulum:
    """
    A double (two-segment) pendulum with external torque inputs.

    State: [theta1, theta2, omega1, omega2]
    where theta = angle, omega = angular velocity.
    """

    def __init__(self, l1=1.0, l2=0.8, m1=1.5, m2=1.0,
                 g=9.81, damping=0.02, pendulum_id=0):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.damping = damping
        self.pendulum_id = pendulum_id

        # State: [theta1, theta2, omega1, omega2]
        self.state = np.array([
            np.pi / 2 + 0.3 * pendulum_id,  # theta1
            np.pi / 2 - 0.2 * pendulum_id,  # theta2
            0.0,                              # omega1
            0.0,                              # omega2
        ])

        # Trajectory history for visualization
        self.trail_x1 = []
        self.trail_y1 = []
        self.trail_x2 = []
        self.trail_y2 = []

    def derivatives(self, state, tau1=0.0, tau2=0.0):
        """Compute derivatives using Lagrangian mechanics."""
        t1, t2, w1, w2 = state
        m1, m2, l1, l2, g, b = self.m1, self.m2, self.l1, self.l2, self.g, self.damping

        delta = t2 - t1
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)

        den1 = (m1 + m2) * l1 - m2 * l1 * cos_d * cos_d
        den2 = (l2 / l1) * den1

        # Prevent division by zero
        den1 = np.clip(den1, 1e-8, None) if den1 > 0 else np.clip(den1, None, -1e-8)
        den2 = np.clip(den2, 1e-8, None) if den2 > 0 else np.clip(den2, None, -1e-8)

        dw1 = (
            m2 * l1 * w1 * w1 * sin_d * cos_d
            + m2 * g * np.sin(t2) * cos_d
            + m2 * l2 * w2 * w2 * sin_d
            - (m1 + m2) * g * np.sin(t1)
            - b * w1
            + tau1
        ) / den1

        dw2 = (
            -m2 * l2 * w2 * w2 * sin_d * cos_d
            + (m1 + m2) * g * np.sin(t1) * cos_d
            - (m1 + m2) * l1 * w1 * w1 * sin_d
            - (m1 + m2) * g * np.sin(t2)
            - b * w2
            + tau2
        ) / den2

        return np.array([w1, w2, dw1, dw2])

    def step(self, dt, tau1=0.0, tau2=0.0):
        """RK4 integration step with external torques."""
        # Clamp torques for safety
        tau1 = np.clip(tau1, -10.0, 10.0)
        tau2 = np.clip(tau2, -10.0, 10.0)

        k1 = self.derivatives(self.state, tau1, tau2)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, tau1, tau2)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, tau1, tau2)
        k4 = self.derivatives(self.state + dt * k3, tau1, tau2)

        self.state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Record trail
        x1, y1, x2, y2 = self.get_positions()
        self.trail_x2.append(x2)
        self.trail_y2.append(y2)
        # Limit trail length
        max_trail = 300
        if len(self.trail_x2) > max_trail:
            self.trail_x2.pop(0)
            self.trail_y2.pop(0)

    def get_positions(self):
        """Get (x1, y1, x2, y2) Cartesian coordinates of both masses."""
        t1, t2 = self.state[0], self.state[1]
        x1 = self.l1 * np.sin(t1)
        y1 = -self.l1 * np.cos(t1)
        x2 = x1 + self.l2 * np.sin(t2)
        y2 = y1 - self.l2 * np.cos(t2)
        return x1, y1, x2, y2

    def get_state_vector(self):
        """Return state as array [theta1, theta2, omega1, omega2]."""
        return self.state.copy()

    def get_energy(self):
        """Total mechanical energy (kinetic + potential)."""
        t1, t2, w1, w2 = self.state
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        # Kinetic
        T = (0.5 * m1 * (l1 * w1) ** 2
             + 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2
                           + 2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)))
        # Potential
        V = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)

        return T + V
