"""
Real-Time Visualizer — matplotlib FuncAnimation with sliders.

Panels:
  1. Double pendulums with trails (ballet!)
  2. Constellation DAG with pulsing node colors
  3. Phase portrait of hub oscillators
  4. Node activation waterfall / time series
  5. DLINOSS oscillator phase space
  6. Energy plot

Sliders:
  - Coupling strength
  - Damping / firing threshold
  - Chaos factor
  - Time scale
"""

import os
import numpy as np
import matplotlib

# Pick the best available interactive backend
_backend_set = False
for _be in ("TkAgg", "Qt5Agg", "GTK3Agg", "Agg"):
    try:
        matplotlib.use(_be)
        _backend_set = True
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors


class RealTimeVisualizer:
    """Real-time matplotlib dashboard for the recursive BDH constellation."""

    def __init__(self, orchestrator, cfg):
        self.orch = orchestrator
        self.cfg = cfg
        self.running = True

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(20, 12), facecolor="#0a0a0a")
        self.fig.canvas.manager.set_window_title(
            "Recursive BDH + DLINOSS Constellation — Neuromorphic Ballet"
        )

        # Create grid layout
        gs = self.fig.add_gridspec(
            4, 4, hspace=0.35, wspace=0.3,
            left=0.05, right=0.95, top=0.93, bottom=0.18
        )

        # Panel 1: Double pendulums (top-left, large)
        self.ax_pend = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_pend.set_xlim(-3.5, 3.5)
        self.ax_pend.set_ylim(-3.0, 1.5)
        self.ax_pend.set_aspect("equal")
        self.ax_pend.set_title("Double Pendulum Ballet", fontsize=11, color="#ff6b6b")

        # Panel 2: Constellation DAG (top-right)
        self.ax_dag = self.fig.add_subplot(gs[0:2, 2:4])
        self.ax_dag.set_title("BDH+DLINOSS Constellation", fontsize=11, color="#51cf66")

        # Panel 3: Phase portrait (bottom-left)
        self.ax_phase = self.fig.add_subplot(gs[2, 0])
        self.ax_phase.set_title("Hub Phase Portrait", fontsize=9, color="#74c0fc")

        # Panel 4: Activation waterfall (bottom, wide)
        self.ax_waterfall = self.fig.add_subplot(gs[2, 1:3])
        self.ax_waterfall.set_title("Node Activations", fontsize=9, color="#ffd43b")

        # Panel 5: DLINOSS phase (bottom-right)
        self.ax_osc = self.fig.add_subplot(gs[2, 3])
        self.ax_osc.set_title("Oscillator Phases", fontsize=9, color="#e599f7")

        # Panel 6: Energy plot (bottom row)
        self.ax_energy = self.fig.add_subplot(gs[3, 0:2])
        self.ax_energy.set_title("System Energy", fontsize=9, color="#ff922b")

        # Panel 7: Pulse rates (bottom row right)
        self.ax_pulse = self.fig.add_subplot(gs[3, 2:4])
        self.ax_pulse.set_title("Pulse Rates", fontsize=9, color="#20c997")

        # Pre-compute DAG layout
        self.dag_pos = self.orch.constellation.get_graph_positions()

        # Sliders
        slider_color = "#2a2a3a"
        ax_coupling = self.fig.add_axes([0.08, 0.08, 0.18, 0.02], facecolor=slider_color)
        ax_threshold = self.fig.add_axes([0.08, 0.045, 0.18, 0.02], facecolor=slider_color)
        ax_chaos = self.fig.add_axes([0.35, 0.08, 0.18, 0.02], facecolor=slider_color)
        ax_noise = self.fig.add_axes([0.35, 0.045, 0.18, 0.02], facecolor=slider_color)
        ax_timescale = self.fig.add_axes([0.62, 0.08, 0.18, 0.02], facecolor=slider_color)

        self.s_coupling = Slider(ax_coupling, "Coupling", 0.01, 2.0,
                                  valinit=self.orch.coupling_strength, color="#ff6b6b")
        self.s_threshold = Slider(ax_threshold, "Threshold", 0.01, 1.0,
                                   valinit=self.orch.firing_threshold, color="#51cf66")
        self.s_chaos = Slider(ax_chaos, "Chaos", 0.0, 1.0,
                               valinit=self.orch.chaos_factor, color="#74c0fc")
        self.s_noise = Slider(ax_noise, "Noise", 0.0, 0.5,
                               valinit=self.orch.noise_scale, color="#ffd43b")
        self.s_timescale = Slider(ax_timescale, "Time Scale", 1, 20,
                                   valinit=4, valstep=1, color="#e599f7")

        # Connect sliders
        self.s_coupling.on_changed(self._on_coupling)
        self.s_threshold.on_changed(self._on_threshold)
        self.s_chaos.on_changed(self._on_chaos)
        self.s_noise.on_changed(self._on_noise)

        # Phase portrait history
        self.phase_x = []
        self.phase_y = []

        # Waterfall history
        self.waterfall_data = np.zeros((100, self.orch.constellation.n_total))

    def _on_coupling(self, val):
        self.orch.coupling_strength = val

    def _on_threshold(self, val):
        self.orch.firing_threshold = val

    def _on_chaos(self, val):
        self.orch.chaos_factor = val

    def _on_noise(self, val):
        self.orch.noise_scale = val

    def _update(self, frame):
        """Animation update called each frame."""
        # Run simulation steps
        n_steps = int(self.s_timescale.val)
        for _ in range(n_steps):
            self.orch.step()

        # ---- Draw pendulums ----
        self.ax_pend.clear()
        self.ax_pend.set_xlim(-3.5, 3.5)
        self.ax_pend.set_ylim(-3.0, 1.5)
        self.ax_pend.set_facecolor("#0a0a0a")
        self.ax_pend.set_title("Double Pendulum Ballet", fontsize=11, color="#ff6b6b")

        colors_pend = [("#ff6b6b", "#ff8787", "#ffa8a8"),
                       ("#74c0fc", "#91d5ff", "#b2e0ff")]
        offsets = [-1.2, 1.2]

        for p_idx, pend in enumerate(self.orch.pendulums):
            ox = offsets[p_idx]
            x1, y1, x2, y2 = pend.get_positions()
            c1, c2, c3 = colors_pend[p_idx]

            # Draw trail
            if len(pend.trail_x2) > 2:
                tx = np.array(pend.trail_x2) + ox
                ty = np.array(pend.trail_y2)
                n_trail = len(tx)
                alphas = np.linspace(0.02, 0.6, n_trail)
                for i in range(n_trail - 1):
                    self.ax_pend.plot(
                        [tx[i], tx[i + 1]], [ty[i], ty[i + 1]],
                        color=c3, alpha=alphas[i], linewidth=0.8
                    )

            # Draw pendulum
            self.ax_pend.plot([ox, x1 + ox], [0, y1], color=c1, linewidth=2.5, zorder=5)
            self.ax_pend.plot([x1 + ox, x2 + ox], [y1, y2], color=c2, linewidth=2.0, zorder=5)
            self.ax_pend.scatter([ox], [0], color="white", s=30, zorder=6)
            self.ax_pend.scatter([x1 + ox], [y1], color=c1, s=60, zorder=6, edgecolors="white", linewidths=0.5)
            self.ax_pend.scatter([x2 + ox], [y2], color=c2, s=40, zorder=6, edgecolors="white", linewidths=0.5)

        # ---- Draw constellation DAG ----
        self.ax_dag.clear()
        self.ax_dag.set_facecolor("#0a0a0a")
        self.ax_dag.set_title("BDH+DLINOSS Constellation", fontsize=11, color="#51cf66")

        const = self.orch.constellation
        pos = self.dag_pos
        activations = const.get_all_activations()
        max_act = max(activations.max(), 0.01)
        norm_act = activations / max_act

        # Draw edges with weight-based opacity
        wm = const.get_weight_matrix()
        for idx, (u, v) in enumerate(const.edge_list):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = const.weights[idx].item()
            alpha = min(w / 2.0, 0.8)
            self.ax_dag.plot([x0, x1], [y0, y1], color="#3a3a4a",
                            alpha=alpha, linewidth=0.8)

        # Draw nodes
        for i in range(const.n_total):
            x, y = pos[i]
            if const.node_types[i] == "bdh":
                color = plt.cm.plasma(norm_act[i])
                marker = "o"
                size = 80 + 120 * norm_act[i]
            else:
                color = plt.cm.cool(norm_act[i])
                marker = "D"
                size = 100 + 100 * norm_act[i]

            # Hub nodes get a glow ring
            if i in const.hub_ids:
                self.ax_dag.scatter([x], [y], c=[color], s=size * 2.5,
                                   marker=marker, alpha=0.2, edgecolors="none")
            self.ax_dag.scatter([x], [y], c=[color], s=size,
                               marker=marker, edgecolors="white",
                               linewidths=0.3, alpha=0.9)

            # Pulse flash
            if const.pulse_rates[i] > 0.5:
                self.ax_dag.scatter([x], [y], c="white", s=size * 0.3,
                                   marker="*", alpha=0.7)

        self.ax_dag.set_xlim(-1.3, 1.3)
        self.ax_dag.set_ylim(-1.3, 1.3)
        self.ax_dag.axis("off")

        # ---- Phase portrait ----
        self.ax_phase.clear()
        self.ax_phase.set_facecolor("#0a0a0a")
        self.ax_phase.set_title("Hub Phase Portrait", fontsize=9, color="#74c0fc")

        if self.orch.hub_phase_history:
            hp = np.array(self.orch.hub_phase_history[-200:])
            n_pts = len(hp)
            if n_pts > 1:
                colors_phase = plt.cm.viridis(np.linspace(0.2, 1.0, n_pts))
                for h in range(min(hp.shape[1], 4)):
                    self.ax_phase.scatter(
                        hp[:, h, 0], hp[:, h, 1],
                        c=colors_phase, s=2, alpha=0.6
                    )
        self.ax_phase.set_xlabel("x₁", fontsize=7, color="#666")
        self.ax_phase.set_ylabel("x₂", fontsize=7, color="#666")
        self.ax_phase.tick_params(labelsize=6, colors="#444")

        # ---- Activation waterfall ----
        act_row = norm_act
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1, :] = act_row

        self.ax_waterfall.clear()
        self.ax_waterfall.set_facecolor("#0a0a0a")
        self.ax_waterfall.set_title("Node Activations", fontsize=9, color="#ffd43b")
        self.ax_waterfall.imshow(
            self.waterfall_data.T, aspect="auto", cmap="inferno",
            vmin=0, vmax=1, origin="lower", interpolation="bilinear"
        )
        self.ax_waterfall.set_xlabel("Time", fontsize=7, color="#666")
        self.ax_waterfall.set_ylabel("Node", fontsize=7, color="#666")
        self.ax_waterfall.tick_params(labelsize=6, colors="#444")

        # ---- DLINOSS oscillator phases ----
        self.ax_osc.clear()
        self.ax_osc.set_facecolor("#0a0a0a")
        self.ax_osc.set_title("Oscillator Phases", fontsize=9, color="#e599f7")

        for i in range(const.n_total):
            if const.node_types[i] == "dlinoss":
                node = const.nodes[i]
                x_s, z_s = node.get_phase()
                # Plot first 8 dimensions as phase trajectories
                xv = x_s[:8].cpu().numpy()
                zv = z_s[:8].cpu().numpy()
                self.ax_osc.scatter(xv, zv, s=8, alpha=0.7,
                                   c=plt.cm.cool(np.linspace(0, 1, 8)))
        self.ax_osc.set_xlabel("position", fontsize=7, color="#666")
        self.ax_osc.set_ylabel("velocity", fontsize=7, color="#666")
        self.ax_osc.tick_params(labelsize=6, colors="#444")

        # ---- Energy plot ----
        self.ax_energy.clear()
        self.ax_energy.set_facecolor("#0a0a0a")
        self.ax_energy.set_title("System Energy", fontsize=9, color="#ff922b")

        if len(self.orch.energy_history) > 1:
            self.ax_energy.plot(
                self.orch.energy_history[-300:],
                color="#ff922b", alpha=0.8, linewidth=0.8, label="Constellation"
            )
        for p_idx in range(len(self.orch.pendulums)):
            pe = self.orch.pend_energy_history[p_idx]
            if len(pe) > 1:
                colors_e = ["#ff6b6b", "#74c0fc"]
                self.ax_energy.plot(
                    pe[-300:], color=colors_e[p_idx],
                    alpha=0.7, linewidth=0.8, label=f"Pend {p_idx}"
                )
        self.ax_energy.legend(fontsize=6, loc="upper right", framealpha=0.3)
        self.ax_energy.tick_params(labelsize=6, colors="#444")

        # ---- Pulse rates ----
        self.ax_pulse.clear()
        self.ax_pulse.set_facecolor("#0a0a0a")
        self.ax_pulse.set_title("Pulse Rates", fontsize=9, color="#20c997")

        pr = const.pulse_rates
        bar_colors = ["#20c997" if const.node_types[i] == "bdh" else "#e599f7"
                      for i in range(const.n_total)]
        self.ax_pulse.bar(range(const.n_total), pr, color=bar_colors, alpha=0.8)
        self.ax_pulse.set_ylim(0, 1.05)
        self.ax_pulse.set_xlabel("Node", fontsize=7, color="#666")
        self.ax_pulse.tick_params(labelsize=6, colors="#444")

        # Title with live stats
        self.fig.suptitle(
            f"Recursive BDH Constellation  |  tick={self.orch.tick}  |  "
            f"nodes={const.n_total}  |  edges={len(const.edge_list)}  |  "
            f"coupling={self.orch.coupling_strength:.2f}  chaos={self.orch.chaos_factor:.2f}",
            fontsize=12, color="#aaa", y=0.97
        )

        return []

    def run(self, tb_logger=None):
        """Start the real-time animation."""
        def update_with_logging(frame):
            result = self._update(frame)
            if tb_logger is not None:
                tb_logger.log(self.orch)
            return result

        interval = self.cfg["visualization"]["update_interval_ms"]
        self.anim = FuncAnimation(
            self.fig, update_with_logging,
            interval=interval, blit=False, cache_frame_data=False
        )
        plt.show()
