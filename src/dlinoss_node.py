"""
Damped Linear Oscillatory State-Space (DLINOSS) Node — PyTorch port.

Based on jaredbmit/damped-linoss DampedIMEX1Layer.
Implements a damped harmonic oscillator as a state-space recurrence:
    z_{k+1} = (z_k + dt * (-A*x_k + B*u_{k+1})) / (1 + dt*G)
    x_{k+1} = x_k + dt * z_{k+1}

Provides oscillatory coupling between BDH nodes in the constellation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DampedLinOSSNode(nn.Module):
    """
    A single Damped LinOSS oscillator node for the constellation.
    Maintains position (x) and velocity (z) state vectors,
    producing oscillatory dynamics with damping.
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 damping_range=(0.1, 2.0), freq_range=(0.5, 8.0),
                 dt_init=0.05, node_id: int = 0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.node_id = node_id

        # Learnable SSM parameters
        self.A_diag = nn.Parameter(
            torch.empty(state_dim).uniform_(freq_range[0], freq_range[1])
        )
        self.G_diag = nn.Parameter(
            torch.empty(state_dim).uniform_(damping_range[0], damping_range[1])
        )
        self.dt_raw = nn.Parameter(torch.full((state_dim,), dt_init))

        # Complex-valued B, C matrices (stored as real pairs)
        self.B_re = nn.Parameter(torch.empty(state_dim, hidden_dim).normal_(
            std=1.0 / (hidden_dim ** 0.5)))
        self.B_im = nn.Parameter(torch.empty(state_dim, hidden_dim).normal_(
            std=1.0 / (hidden_dim ** 0.5)))
        self.C_re = nn.Parameter(torch.empty(hidden_dim, state_dim).normal_(
            std=1.0 / (state_dim ** 0.5)))
        self.C_im = nn.Parameter(torch.empty(hidden_dim, state_dim).normal_(
            std=1.0 / (state_dim ** 0.5)))
        self.D = nn.Parameter(torch.randn(hidden_dim))

        # Oscillator state: position and velocity
        self.register_buffer("x_state", torch.zeros(state_dim))
        self.register_buffer("z_state", torch.zeros(state_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def _soft_project(self):
        """Project parameters to valid stability region."""
        dt = torch.sigmoid(self.dt_raw)
        G = F.relu(self.G_diag)
        A = F.relu(self.A_diag)

        # Stability constraint: (G - dt*A)^2 - 4*A < 0
        A_low = (2 + dt * G - 2 * torch.sqrt(1 + dt * G)) / torch.clamp(dt ** 2, min=1e-6)
        A_high = (2 + dt * G + 2 * torch.sqrt(1 + dt * G)) / torch.clamp(dt ** 2, min=1e-6)
        A = A_low + F.relu(A - A_low) - F.relu(A - A_high)

        return A, G, dt

    @torch.no_grad()
    def step(self, u):
        """
        One recurrence step of the damped oscillator.
        Args:
            u: input signal (hidden_dim,)
        Returns:
            output: (hidden_dim,) oscillatory output
        """
        A, G, dt = self._soft_project()

        # Complex B @ u
        Bu = (self.B_re @ u) + 1j * (self.B_im @ u)  # (state_dim,) complex

        # Damped IMEX1 recurrence
        S = 1.0 + dt * G
        z_new = (self.z_state + dt * (-A * self.x_state + Bu.real)) / S
        x_new = self.x_state + dt * z_new

        self.z_state.copy_(z_new)
        self.x_state.copy_(x_new)

        # Output: C @ x + D * u
        y = (self.C_re @ x_new) + self.D * u

        return self.output_proj(y)

    def get_energy(self):
        """Oscillator energy: kinetic + potential."""
        A, _, _ = self._soft_project()
        kinetic = 0.5 * (self.z_state ** 2).sum()
        potential = 0.5 * (A * self.x_state ** 2).sum()
        return (kinetic + potential).item()

    def get_phase(self):
        """Return oscillator phase information for visualization."""
        return self.x_state.detach().clone(), self.z_state.detach().clone()

    def reset(self):
        """Reset oscillator state with slight perturbation."""
        self.x_state.normal_(std=0.1)
        self.z_state.normal_(std=0.1)
