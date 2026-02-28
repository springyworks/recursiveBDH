"""
Continuous BDH Node — adapted from pathwaycom/bdh for real-time dynamics.

Instead of token embeddings, uses continuous state vectors.
Preserves the core BDH mechanics: RoPE attention, sparse ReLU pulsing,
encoder/decoder projections, and gated Hebbian-style interactions.
"""

import math
import dataclasses

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class ContinuousBDHConfig:
    n_layer: int = 1
    n_embd: int = 32
    n_head: int = 2
    mlp_internal_dim_multiplier: int = 4
    dropout: float = 0.0
    buffer_len: int = 16


def _get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class ContinuousAttention(nn.Module):
    """RoPE-based causal self-attention for continuous state sequences."""

    def __init__(self, config: ContinuousBDHConfig):
        super().__init__()
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            _get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def _phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def _rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = ContinuousAttention._phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self._rope(r_phases, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        return scores @ V


class ContinuousBDHNode(nn.Module):
    """
    A single BDH node adapted for continuous dynamics in a constellation.
    Maintains a temporal buffer and processes it with BDH attention.
    """

    def __init__(self, config: ContinuousBDHConfig, node_id: int = 0):
        super().__init__()
        self.config = config
        self.node_id = node_id

        D = config.n_embd
        nh = config.n_head
        N = config.mlp_internal_dim_multiplier * D // nh

        # Input/output projections (continuous, not embedding)
        self.input_proj = nn.Linear(D, D, bias=False)
        self.output_proj = nn.Linear(D, D, bias=False)

        # BDH core parameters
        self.encoder = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
        self.decoder = nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))

        self.attn = ContinuousAttention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)

        # Temporal buffer (circular)
        self.register_buffer(
            "buffer", torch.zeros(1, 1, config.buffer_len, D)
        )
        self.register_buffer("buf_ptr", torch.tensor(0, dtype=torch.long))

        # Node activation state
        self.register_buffer("activation", torch.zeros(D))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def push_to_buffer(self, x):
        """Push a state vector into the circular buffer."""
        ptr = self.buf_ptr.item()
        self.buffer[0, 0, ptr, :] = x
        self.buf_ptr.fill_((ptr + 1) % self.config.buffer_len)

    @torch.no_grad()
    def step(self, neighbor_input):
        """
        Process one timestep — fast path using only encoder/decoder
        with gated sparse activation (no full attention recompute).
        """
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = C.mlp_internal_dim_multiplier * D // nh

        # Project & combine with current state
        x_in = self.input_proj(neighbor_input)
        combined = self.ln(x_in + self.activation)
        self.push_to_buffer(combined)

        # Fast BDH: encode → sparse gate → decode (single vector)
        x = combined.view(1, 1, 1, D)
        x = self.ln(x)

        for _ in range(C.n_layer):
            x_latent = x @ self.encoder             # (1, nh, 1, N)
            x_sparse = F.relu(x_latent)

            # Lightweight self-modulation using buffer statistics
            buf_mean = self.buffer.mean(dim=2, keepdim=True)  # (1, 1, 1, D)
            buf_latent = buf_mean @ self.encoder_v   # (1, nh, 1, N)
            buf_sparse = F.relu(buf_latent)

            xy_sparse = x_sparse * buf_sparse       # gating
            yMLP = (
                xy_sparse.transpose(1, 2)
                .reshape(1, 1, 1, N * nh)
                @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        output = self.output_proj(x.view(D))
        self.activation.copy_(output)
        return output

    def get_pulse(self, threshold):
        """Return a sparse pulse: only fire if activation exceeds threshold."""
        mask = (self.activation.abs() > threshold).float()
        return self.activation * mask

    def get_energy(self):
        """Node's current activation energy."""
        return self.activation.norm().item()
