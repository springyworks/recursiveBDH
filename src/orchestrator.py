"""
Orchestrator — the Meta-BDH controller.

Manages the constellation of BDH + DLINOSS nodes and two double pendulums.
Implements the hierarchical control loop:
  Pendulum states → Input hubs → Constellation propagation → Output hubs → Torques

Provides top-down control, inhibiting or exciting child nodes.
"""

import torch
import numpy as np


class Orchestrator:
    """
    Meta-BDH: orchestrates the constellation and double pendulums.
    """

    def __init__(self, constellation, pendulums, cfg, device="cpu"):
        self.constellation = constellation
        self.pendulums = pendulums
        self.cfg = cfg
        self.device = device

        state_dim = cfg["constellation"]["state_dim"]
        self.internal_steps = cfg["orchestrator"]["internal_steps"]

        # Pendulum ↔ constellation interface projections
        # Pendulum state (4D) → node state (state_dim)
        self.pend_to_node = torch.randn(4, state_dim, device=device) * 0.1
        # Node state (state_dim) → torques (2D per pendulum)
        self.node_to_torque = torch.randn(state_dim, 2, device=device) * 0.05

        # Runtime parameters (adjustable via sliders)
        self.coupling_strength = cfg["orchestrator"]["coupling_strength"]
        self.firing_threshold = cfg["orchestrator"]["firing_threshold"]
        self.noise_scale = cfg["orchestrator"]["noise_scale"]
        self.chaos_factor = cfg["orchestrator"]["chaos_factor"]
        self.sim_dt = cfg["pendulum"]["sim_dt"]

        # Simulation clock
        self.tick = 0

        # Metrics history
        self.energy_history = []
        self.pend_energy_history = [[], []]
        self.activation_history = []
        self.hub_phase_history = []

    def step(self):
        """Execute one full orchestration cycle."""
        self.tick += 1
        hub_ids = self.constellation.hub_ids

        # 1. Read pendulum states → inject into input hubs
        for p_idx, pend in enumerate(self.pendulums):
            pstate = torch.tensor(
                pend.get_state_vector(), dtype=torch.float32, device=self.device
            )
            signal = pstate @ self.pend_to_node
            # Inject into designated hub nodes
            hub_a = hub_ids[p_idx * 2]
            hub_b = hub_ids[min(p_idx * 2 + 1, len(hub_ids) - 1)]
            self.constellation.inject(hub_a, signal * 0.5)
            self.constellation.inject(hub_b, signal * 0.5)

        # 2. Run constellation internal steps (emergence!)
        for _ in range(self.internal_steps):
            self.constellation.step(
                coupling_strength=self.coupling_strength,
                firing_threshold=self.firing_threshold,
                noise_scale=self.noise_scale,
                chaos_factor=self.chaos_factor,
            )

        # 3. Read hub outputs → generate torques for pendulums
        hub_outs = self.constellation.get_hub_outputs(n=5)
        for p_idx, pend in enumerate(self.pendulums):
            # Average outputs from designated hubs
            h1 = hub_outs[p_idx * 2]
            h2 = hub_outs[min(p_idx * 2 + 1, len(hub_outs) - 1)]
            combined = (h1 + h2) * 0.5
            torques = combined @ self.node_to_torque
            tau1 = torques[0].item() * 2.0
            tau2 = torques[1].item() * 2.0

            # Step pendulum physics (multiple sub-steps for stability)
            n_substeps = 4
            sub_dt = self.sim_dt / n_substeps
            for _ in range(n_substeps):
                pend.step(sub_dt, tau1, tau2)

        # 4. Record metrics
        self._record_metrics()

    def _record_metrics(self):
        """Record metrics for TensorBoard and visualization."""
        activations = self.constellation.get_all_activations()
        self.activation_history.append(activations.copy())
        if len(self.activation_history) > 500:
            self.activation_history.pop(0)

        total_energy = sum(self.constellation.energies)
        self.energy_history.append(total_energy)
        if len(self.energy_history) > 1000:
            self.energy_history.pop(0)

        for p_idx, pend in enumerate(self.pendulums):
            self.pend_energy_history[p_idx].append(pend.get_energy())
            if len(self.pend_energy_history[p_idx]) > 1000:
                self.pend_energy_history[p_idx].pop(0)

        # Hub phases (for phase portrait)
        hub_phases = []
        for hid in self.constellation.hub_ids[:4]:
            out = self.constellation.outputs[hid]
            hub_phases.append(out[:2].cpu().numpy())
        self.hub_phase_history.append(np.array(hub_phases))
        if len(self.hub_phase_history) > 500:
            self.hub_phase_history.pop(0)

    def get_metrics(self):
        """Return current metrics dict."""
        return {
            "tick": self.tick,
            "constellation_energy": self.energy_history[-1] if self.energy_history else 0,
            "pend_energies": [h[-1] if h else 0 for h in self.pend_energy_history],
            "activations": self.constellation.get_all_activations(),
            "pulse_rates": self.constellation.pulse_rates.copy(),
            "weight_matrix": self.constellation.get_weight_matrix(),
            "energies": self.constellation.energies.copy(),
        }
