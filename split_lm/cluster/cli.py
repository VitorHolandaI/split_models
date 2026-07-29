"""`split-cluster` entry point: run a node agent or list the ones on the LAN."""

from __future__ import annotations

import argparse
import logging
import secrets
import socket
import subprocess
import sys
import threading
from collections.abc import Sequence

from transformers import AutoConfig

from ..benchmark_profiles import (
    RUNTIME_PRECISIONS, format_profiles, load_profile, load_profiles,
    profile_overrides)
from ..context_benchmark import parse_context_sizes
from ..model.loader import _check_supported_config, _ensure_precision_available
from .control import serve_control
from .discovery import DISCOVERY_PORT, discover_nodes, local_node, serve_discovery
from .master import run_vertical_cluster
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


def random_node_id() -> str:
    """Return an opaque worker identity for one agent process."""
    return f"node-{secrets.token_hex(4)}"


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-id", default=random_node_id(),
                        help="worker: identificador opcional; por padrao gera um aleatorio")
    parser.add_argument("--control-port", type=int, default=CONTROL_PORT,
                        help="worker: porta TCP onde o agente aceita jobs vertical")
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT,
                        help="porta UDP de discovery, igual em todos os nos")


def _add_discovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT,
                        help="porta UDP de discovery, igual em todos os nos")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="nodes: segundos esperando respostas do discovery")
    # 255.255.255.255 sai pela rota default; com VPN (wg0 /32) na frente da LAN
    # o probe morre no tunel, entao o operador aponta o broadcast da subrede.
    parser.add_argument("--broadcast", default="255.255.255.255",
                        help="nodes: endereco de broadcast da LAN, ex 10.1.2.255")


def _add_master_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--discover", action="store_true",
                        help="master: encontrar workers via UDP")
    parser.add_argument("--head-address", default=None,
                        help="master: IP anunciado aos workers; por padrao detecta pela rota")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--ready-timeout", type=float, default=300)
    parser.add_argument("--context-sizes", default=None,
                        help="benchmark: contextos crescentes, ex 128,256,512")
    parser.add_argument("--metrics-file", default="benchmark-results/generation.jsonl")
    parser.add_argument("--benchmark-output", default="benchmark-results/context-sweep")
    parser.add_argument("--precision", choices=RUNTIME_PRECISIONS, default="bfloat16")
    parser.add_argument("--warmup-runs", type=int, default=0)


def _add_profile_arguments(parser: argparse.ArgumentParser,
                           required: bool = False) -> None:
    parser.add_argument("--benchmark-config", default="benchmarks.toml",
                        help="TOML containing named benchmark profiles")
    parser.add_argument("--benchmark-profile", required=required,
                        help="profile name shown by `split-cluster benchmarks`")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="split-cluster", description="cluster split LLM")
    commands = parser.add_subparsers(dest="role", required=True, title="commands")
    worker = commands.add_parser("worker", help="wait for vertical jobs")
    _add_worker_arguments(worker)
    nodes = commands.add_parser("nodes", help="list discovered workers")
    _add_discovery_arguments(nodes)
    master = commands.add_parser("master", help="run interactive or benchmark split")
    _add_discovery_arguments(master)
    _add_master_arguments(master)
    _add_profile_arguments(master)
    profiles = commands.add_parser("benchmarks", help="list named benchmark profiles")
    profiles.add_argument("--benchmark-config", default="benchmarks.toml")
    local = commands.add_parser(
        "benchmark-local", help="run a profile locally",
        description="Run one named profile on a full local model")
    _add_profile_arguments(local, required=True)
    return parser


def source_address_for(remote_address: str) -> str:
    """Return the local IPv4 address selected for a worker route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((remote_address, DISCOVERY_PORT))
        return str(probe.getsockname()[0])


def local_benchmark_command(config: str, profile: str,
                            executable: str = sys.executable) -> list[str]:
    """Build the local runner command for one shared benchmark profile."""
    return [
        executable, "-m", "tools.benchmark_single_model",
        "--benchmark-config", config, "--benchmark-profile", profile,
    ]


def _run_local_benchmark(args: argparse.Namespace) -> None:
    if not args.benchmark_profile:
        raise ValueError(
            "benchmark-local exige --benchmark-profile; use split-cluster benchmarks")
    load_profile(args.benchmark_config, args.benchmark_profile)
    subprocess.run(local_benchmark_command(
        args.benchmark_config, args.benchmark_profile), check=True)


def _run_worker_agent(args: argparse.Namespace) -> None:
    node = local_node(args.node_id, args.control_port)
    stop = threading.Event()
    discovery = threading.Thread(
        target=serve_discovery, args=(node, args.discovery_port, stop), daemon=True)
    discovery.start()
    log.info("worker %s pronto: discovery udp/%d, controle tcp/%d",
             node.node_id, args.discovery_port, args.control_port)
    try:
        serve_control(node.node_id, node.control_port, stop)
    finally:
        stop.set()
        discovery.join(2)


def _run_master(args: argparse.Namespace) -> None:
    if not args.discover:
        raise ValueError("master exige --discover; esperado discovery UDP dos workers")
    if args.benchmark_profile:
        profile = load_profile(args.benchmark_config, args.benchmark_profile)
        for name, value in profile_overrides(profile).items():
            setattr(args, name, value)
    _ensure_precision_available(args.precision)
    workers = discover_nodes(args.discovery_port, args.timeout, args.broadcast)
    if not workers:
        raise RuntimeError(f"nenhum worker encontrado via {args.broadcast}:{args.discovery_port}")
    config = AutoConfig.from_pretrained(args.model)
    _check_supported_config(config)
    head_address = args.head_address or source_address_for(workers[0].address)
    context_sizes = ()
    if args.context_sizes:
        max_context = int(config.max_position_embeddings) - args.max_new_tokens
        context_sizes = parse_context_sizes(args.context_sizes, max_context)
    print(format_nodes(workers), flush=True)
    run_vertical_cluster(
        args.model, int(config.num_hidden_layers), head_address, workers,
        args.max_new_tokens, args.ready_timeout, context_sizes=context_sizes,
        metrics_file=args.metrics_file, benchmark_output=args.benchmark_output,
        precision=args.precision, warmup_runs=args.warmup_runs)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the cluster CLI.

    Usage example: `split-cluster nodes --timeout 3`.
    """
    args = _build_parser().parse_args(argv)
    if args.role == "benchmarks":
        print(format_profiles(load_profiles(args.benchmark_config)))
        return
    if args.role == "benchmark-local":
        _run_local_benchmark(args)
        return
    if args.role == "nodes":
        print(format_nodes(
            discover_nodes(args.discovery_port, args.timeout, args.broadcast)))
        return
    if args.role == "master":
        _run_master(args)
        return
    _run_worker_agent(args)


if __name__ == "__main__":
    main()
