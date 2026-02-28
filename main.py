#!/usr/bin/env python3
"""
Recursive BDH + DLINOSS Constellation — Main Entry Point.

Launches the neuromorphic ballet: a scale-free constellation of 20 BDH nodes
and 5 DLINOSS oscillators orchestrating two double pendulums. Real-time
matplotlib visualization with sliders + TensorBoard logging.

Usage:
    python main.py [--config configs/constellation.yaml]
"""

import argparse
import sys
import os

import yaml
import torch

from src.constellation import Constellation
from src.double_pendulum import DoublePendulum
from src.orchestrator import Orchestrator
from src.visualizer import RealTimeVisualizer
from src.tb_logger import TBLogger


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Recursive BDH + DLINOSS Constellation"
    )
    parser.add_argument(
        "--config", default="configs/constellation.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--no-tb", action="store_true",
        help="Disable TensorBoard logging"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[recBDH] Device: {device}")

    # Build constellation
    print("[recBDH] Building constellation DAG...")
    constellation = Constellation(cfg, device=device)
    print(f"  → {constellation.n_total} nodes, {len(constellation.edge_list)} edges")
    print(f"  → Hub nodes: {constellation.hub_ids}")
    print(f"  → BDH nodes: {sum(1 for t in constellation.node_types if t == 'bdh')}")
    print(f"  → DLINOSS nodes: {sum(1 for t in constellation.node_types if t == 'dlinoss')}")

    # Initialize DLINOSS oscillators with perturbation
    for i, node in enumerate(constellation.nodes):
        if constellation.node_types[i] == "dlinoss":
            node.reset()

    # Build double pendulums
    pcfg = cfg["pendulum"]
    pendulums = []
    for p_id in range(pcfg["num_pendulums"]):
        pend = DoublePendulum(
            l1=pcfg["l1"], l2=pcfg["l2"],
            m1=pcfg["m1"], m2=pcfg["m2"],
            g=pcfg["gravity"], damping=pcfg["damping"],
            pendulum_id=p_id,
        )
        pendulums.append(pend)
    print(f"[recBDH] {len(pendulums)} double pendulums initialized")

    # Build orchestrator
    orchestrator = Orchestrator(constellation, pendulums, cfg, device=device)
    print("[recBDH] Orchestrator ready")

    # TensorBoard
    tb_logger = None
    if not args.no_tb:
        tb_cfg = cfg["tensorboard"]
        os.makedirs(tb_cfg["log_dir"], exist_ok=True)
        tb_logger = TBLogger(
            log_dir=tb_cfg["log_dir"],
            log_interval=tb_cfg["log_interval"],
        )
        print(f"[recBDH] TensorBoard logging to {tb_cfg['log_dir']}/")
        print(f"         Run: tensorboard --logdir {tb_cfg['log_dir']}")

    # Launch visualization
    print("[recBDH] Launching real-time visualization...")
    print("         Close the window to exit.")
    viz = RealTimeVisualizer(orchestrator, cfg)

    try:
        viz.run(tb_logger=tb_logger)
    except KeyboardInterrupt:
        print("\n[recBDH] Interrupted.")
    finally:
        if tb_logger:
            tb_logger.close()
        print("[recBDH] Done.")


if __name__ == "__main__":
    main()
