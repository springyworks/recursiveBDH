"""
TensorBoard Logger — logs constellation metrics to TensorBoard.
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Wraps TensorBoard SummaryWriter for constellation metrics."""

    def __init__(self, log_dir="runs", log_interval=10):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_interval = log_interval
        self.step_count = 0

    def log(self, orchestrator):
        """Log metrics from the orchestrator to TensorBoard."""
        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            return

        t = self.step_count
        metrics = orchestrator.get_metrics()

        # Scalar metrics
        self.writer.add_scalar("energy/constellation", metrics["constellation_energy"], t)
        for i, e in enumerate(metrics["pend_energies"]):
            self.writer.add_scalar(f"energy/pendulum_{i}", e, t)

        # Per-node metrics
        for i, act in enumerate(metrics["activations"]):
            self.writer.add_scalar(f"node_activation/{i}", act, t)
        for i, pr in enumerate(metrics["pulse_rates"]):
            self.writer.add_scalar(f"pulse_rate/{i}", pr, t)
        for i, en in enumerate(metrics["energies"]):
            self.writer.add_scalar(f"node_energy/{i}", en, t)

        # Histograms
        self.writer.add_histogram("activations", metrics["activations"], t)
        self.writer.add_histogram("pulse_rates", metrics["pulse_rates"], t)

        # Weight matrix as image
        wm = metrics["weight_matrix"]
        if wm.max() > 0:
            wm_norm = wm / wm.max()
        else:
            wm_norm = wm
        self.writer.add_image(
            "weight_matrix",
            np.stack([wm_norm, wm_norm * 0.5, wm_norm * 0.2]),
            t,
        )

        # Pendulum states
        for i, pend in enumerate(orchestrator.pendulums):
            s = pend.get_state_vector()
            self.writer.add_scalar(f"pendulum_{i}/theta1", s[0], t)
            self.writer.add_scalar(f"pendulum_{i}/theta2", s[1], t)
            self.writer.add_scalar(f"pendulum_{i}/omega1", s[2], t)
            self.writer.add_scalar(f"pendulum_{i}/omega2", s[3], t)

    def close(self):
        self.writer.close()
