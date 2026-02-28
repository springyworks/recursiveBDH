"""
Recursive BDH + DLINOSS Constellation — Scale-Free DAG.

Builds a Barabási-Albert scale-free graph of 20 BDH nodes and 5 DLINOSS
oscillator nodes. Hub nodes serve as entry/exit points; DLINOSS nodes
provide oscillatory coupling. Sparse pulsing and Hebbian learning govern
the inter-node communication.
"""

import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np

from src.bdh_node import ContinuousBDHNode, ContinuousBDHConfig
from src.dlinoss_node import DampedLinOSSNode


class Constellation:
    """
    A scale-free DAG of BDH + DLINOSS nodes.
    Nodes communicate via sparse pulses; connections adapt via Hebbian learning.
    """

    def __init__(self, cfg: dict, device="cpu"):
        self.cfg = cfg
        self.device = device

        n_bdh = cfg["constellation"]["num_bdh_nodes"]
        n_dlinoss = cfg["constellation"]["num_dlinoss_nodes"]
        self.n_total = n_bdh + n_dlinoss
        state_dim = cfg["constellation"]["state_dim"]
        ba_m = cfg["constellation"]["ba_attachment"]

        # Build scale-free graph
        self.graph = nx.barabasi_albert_graph(self.n_total, ba_m, seed=42)
        # Convert to directed (DAG-ish) by degree rank
        degree_rank = sorted(
            self.graph.degree, key=lambda x: x[1], reverse=True
        )
        self.hub_ids = [n for n, _ in degree_rank[:5]]
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(self.n_total))
        rank_map = {n: i for i, (n, _) in enumerate(degree_rank)}
        for u, v in self.graph.edges():
            if rank_map[u] <= rank_map[v]:
                self.dag.add_edge(u, v)
            else:
                self.dag.add_edge(v, u)
        # Ensure connectivity by adding some back-edges for cycles (emergence)
        for i in range(0, self.n_total - 1, 5):
            j = (i + 7) % self.n_total
            if not self.dag.has_edge(j, i):
                self.dag.add_edge(j, i)

        # Create nodes
        bdh_cfg = ContinuousBDHConfig(
            n_layer=cfg["bdh"]["n_layer"],
            n_embd=cfg["bdh"]["n_embd"],
            n_head=cfg["bdh"]["n_head"],
            mlp_internal_dim_multiplier=cfg["bdh"]["mlp_internal_dim_multiplier"],
            dropout=cfg["bdh"]["dropout"],
            buffer_len=cfg["bdh"]["buffer_len"],
        )
        dlinoss_cfg = cfg["dlinoss"]

        self.nodes = []
        self.node_types = []  # 'bdh' or 'dlinoss'
        for i in range(n_bdh):
            node = ContinuousBDHNode(bdh_cfg, node_id=i).to(device)
            node.eval()
            self.nodes.append(node)
            self.node_types.append("bdh")
        for i in range(n_dlinoss):
            node = DampedLinOSSNode(
                state_dim=dlinoss_cfg["state_dim"],
                hidden_dim=state_dim,
                damping_range=dlinoss_cfg["damping_range"],
                freq_range=dlinoss_cfg["freq_range"],
                dt_init=dlinoss_cfg["dt_init"],
                node_id=n_bdh + i,
            ).to(device)
            node.eval()
            self.nodes.append(node)
            self.node_types.append("dlinoss")

        # Connection weights (Hebbian, learnable)
        self.edge_list = list(self.dag.edges())
        n_edges = len(self.edge_list)
        self.weights = torch.ones(n_edges, device=device) * cfg["orchestrator"]["coupling_strength"]

        # Node outputs cache
        self.outputs = [torch.zeros(state_dim, device=device) for _ in range(self.n_total)]
        self.pulses = [torch.zeros(state_dim, device=device) for _ in range(self.n_total)]

        # Metrics
        self.energies = np.zeros(self.n_total)
        self.pulse_rates = np.zeros(self.n_total)
        self._pulse_history = [[] for _ in range(self.n_total)]

    def inject(self, node_id, signal):
        """Inject an external signal into a specific node."""
        self.outputs[node_id] = self.outputs[node_id] + signal.to(self.device)

    def step(self, coupling_strength=None, firing_threshold=0.3,
             noise_scale=0.02, chaos_factor=0.1):
        """
        Run one propagation step through the constellation.
        Returns per-node outputs.
        """
        state_dim = self.cfg["constellation"]["state_dim"]

        # Gather inputs for each node from its predecessors
        node_inputs = [torch.zeros(state_dim, device=self.device)
                       for _ in range(self.n_total)]

        for idx, (u, v) in enumerate(self.edge_list):
            w = self.weights[idx] if coupling_strength is None else coupling_strength
            pulse = self.pulses[u]
            node_inputs[v] = node_inputs[v] + w * pulse

        # Add noise for emergence (VCO-style babbling)
        for i in range(self.n_total):
            noise = torch.randn(state_dim, device=self.device) * noise_scale
            chaos = torch.randn(state_dim, device=self.device) * chaos_factor
            # Cross-frequency coupling: inject low-freq oscillation
            t = torch.linspace(0, 2 * np.pi, state_dim, device=self.device)
            vco_signal = chaos_factor * torch.sin(t * (1.0 + 0.3 * i))
            node_inputs[i] = node_inputs[i] + noise + chaos * 0.1 + vco_signal * 0.05

        # Process each node
        for i in range(self.n_total):
            inp = node_inputs[i]
            if self.node_types[i] == "bdh":
                out = self.nodes[i].step(inp)
                pulse = self.nodes[i].get_pulse(firing_threshold)
                self.energies[i] = self.nodes[i].get_energy()
            else:
                out = self.nodes[i].step(inp)
                pulse = out * (out.abs() > firing_threshold).float()
                self.energies[i] = self.nodes[i].get_energy()

            self.outputs[i] = out.detach()
            self.pulses[i] = pulse.detach()

            # Track pulse rate
            fired = (pulse.abs() > 0).any().item()
            self._pulse_history[i].append(1.0 if fired else 0.0)
            if len(self._pulse_history[i]) > 100:
                self._pulse_history[i].pop(0)
            self.pulse_rates[i] = np.mean(self._pulse_history[i])

        # Hebbian learning: strengthen connections that fire together
        self._hebbian_update()

        return self.outputs

    def _hebbian_update(self):
        """Hebbian plasticity: connections between co-active nodes strengthen."""
        lr = self.cfg["orchestrator"]["hebbian_lr"]
        for idx, (u, v) in enumerate(self.edge_list):
            pre = self.pulses[u].norm().item()
            post = self.pulses[v].norm().item()
            # Oja's rule (bounded Hebbian)
            delta = lr * (pre * post - self.weights[idx].item() * post * post)
            self.weights[idx] = torch.clamp(
                self.weights[idx] + delta, min=0.01, max=2.0
            )

    def get_hub_outputs(self, n=5):
        """Return outputs from the top-N hub nodes."""
        return [self.outputs[h] for h in self.hub_ids[:n]]

    def get_all_activations(self):
        """Return stacked activation magnitudes for visualization."""
        return np.array([o.norm().item() for o in self.outputs])

    def get_weight_matrix(self):
        """Return adjacency matrix with current Hebbian weights."""
        mat = np.zeros((self.n_total, self.n_total))
        for idx, (u, v) in enumerate(self.edge_list):
            mat[u, v] = self.weights[idx].item()
        return mat

    def get_graph_positions(self):
        """Return node positions for visualization."""
        return nx.spring_layout(self.graph, seed=42)
