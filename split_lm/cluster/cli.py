"""`split-cluster` entry point: run a node agent or list the ones on the LAN."""

from __future__ import annotations

import argparse
import logging
import socket

from .discovery import DISCOVERY_PORT, discover_nodes, local_node, serve_discovery
from .plan import ClusterNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("split_lm")

CONTROL_PORT = 29510
_HEADER = f"{'NODE':<16} {'HOSTNAME':<16} {'CONTROL':<22} {'CPU':>4} {'MEM_MIB':>9}"


def format_nodes(nodes: tuple[ClusterNode, ...]) -> str:
    """Render discovered nodes as a plain-text table for the operator.

    Usage example: `print(format_nodes(discover_nodes()))`.
    """
    if not nodes:
        return "nenhum no respondeu ao discovery"
    rows = [
        f"{node.node_id:<16} {node.hostname:<16} "
        f"{node.address + ':' + str(node.control_port):<22} "
        f"{node.cpu_count:>4} {node.available_memory // 1024 // 1024:>9}"
        for node in nodes
    ]
    return "\n".join([_HEADER, *rows])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="split-cluster", description="cluster split LLM")
    parser.add_argument("role", choices=["worker", "nodes"])
    parser.add_argument("--node-id", default=socket.gethostname(),
                        help="worker: identificador anunciado no discovery")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT,
                        help="worker: porta TCP onde o agente aceita jobs vertical")
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT,
                        help="porta UDP de discovery, igual em todos os nos")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="nodes: segundos esperando respostas do discovery")
    # 255.255.255.255 sai pela rota default; com VPN (wg0 /32) na frente da LAN
    # o probe morre no tunel, entao o operador aponta o broadcast da subrede.
    parser.add_argument("--broadcast", default="255.255.255.255",
                        help="nodes: endereco de broadcast da LAN, ex 10.1.2.255")
    return parser


def main() -> None:
    """Run the cluster CLI.

    Usage example: `split-cluster nodes --timeout 3`.
    """
    args = _build_parser().parse_args()
    if args.role == "nodes":
        print(format_nodes(
            discover_nodes(args.discovery_port, args.timeout, args.broadcast)))
        return
    node = local_node(args.node_id, args.control_port)
    log.info("worker %s pronto: %d cpus, %d MiB livres",
             node.node_id, node.cpu_count, node.available_memory // 1024 // 1024)
    serve_discovery(node, args.discovery_port)


if __name__ == "__main__":
    main()
