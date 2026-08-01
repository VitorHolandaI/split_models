"""`split-llama` entry point: RPC worker, fleet listing, and master run.

Separate from `split-cluster` on purpose. That command drives the PyTorch
pipeline split; this one drives llama.cpp over its RPC backend. They share a
LAN but not a protocol, and merging the two would make every error message
ambiguous about which engine failed.

The RPC backend has no authentication and no encryption: whoever reaches the
port runs code on the node. Isolated LAN only.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import subprocess
import sys
from collections.abc import Sequence

from ..errors import ClusterError
from .context_sweep import (
    DEFAULT_CONTEXT_SIZES, format_sweep, run_context_sweep)
from .examples import format_llama_examples
from .discovery import (
    DISCOVERY_PORT, RPC_PORT, RpcWorker, discover_workers, local_worker,
    rpc_endpoints)
from .master import DEFAULT_GGUF_REPO, list_devices_command, llama_cli_command
from .process import (
    RPC_CACHE_DIR, assert_clean_exit, lan_address, serve_until_server_exits,
    start_rpc_server)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("split_lm")

_HEADER = f"{'NODE':<16} {'HOSTNAME':<16} {'RPC':<22}"
_INSECURE = ("AVISO: o backend RPC do llama.cpp nao tem autenticacao nem "
             "criptografia. Use somente em LAN isolada.")


def random_node_id() -> str:
    """Return an opaque worker identity for one RPC process."""
    return f"node-{secrets.token_hex(4)}"


def format_workers(workers: tuple[RpcWorker, ...]) -> str:
    """Render discovered RPC workers as a plain-text table for the operator.

    Usage example: `print(format_workers(discover_workers()))`.
    """
    if not workers:
        return "nenhum worker RPC respondeu ao discovery"
    rows = [f"{worker.node_id:<16} {worker.hostname:<16} {worker.endpoint:<22}"
            for worker in workers]
    return "\n".join([_HEADER, *rows])


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--discovery-port", type=int, default=DISCOVERY_PORT,
                        help="porta UDP de discovery, igual em todos os nos")
    parser.add_argument("--broadcast", default="255.255.255.255",
                        help="endereco de broadcast da LAN")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="split-llama", description=f"llama.cpp distribuido por RPC. {_INSECURE}",
        epilog="comandos prontos, com modelos GGUF: split-llama exemplos")
    parser.add_argument("--comandos-exemplos", action="store_true",
                        help="imprime comandos prontos e sai")
    commands = parser.add_subparsers(dest="command", required=False)

    commands.add_parser("exemplos", help="imprime comandos prontos com modelos GGUF")

    worker = commands.add_parser("worker", help="serve este no como worker RPC")
    _add_shared_arguments(worker)
    worker.add_argument("--node-id", default=random_node_id(),
                        help="identificador opcional; por padrao gera um aleatorio")
    worker.add_argument("--bind", default="",
                        help="IPv4 onde o RPC escuta; por padrao o da rota da LAN")
    worker.add_argument("--rpc-port", type=int, default=RPC_PORT,
                        help="porta TCP do ggml-rpc-server")
    worker.add_argument("--cache-dir", default=RPC_CACHE_DIR,
                        help="cache de modelos do RPC, em RAM por padrao")

    nodes = commands.add_parser("nodes", help="lista os workers RPC da LAN")
    _add_shared_arguments(nodes)
    nodes.add_argument("--timeout", type=float, default=2.0,
                       help="segundos aguardando respostas de discovery")

    master = commands.add_parser(
        "master", help="roda llama-cli usando os workers descobertos",
        epilog="argumentos apos -- vao direto para o llama-cli")
    _add_shared_arguments(master)
    master.add_argument("--discover", action="store_true",
                        help="descobre os workers em vez de receber --rpc pronto")
    master.add_argument("--hf", default=DEFAULT_GGUF_REPO,
                        help="repositorio GGUF no HuggingFace")
    master.add_argument("--timeout", type=float, default=3.0,
                        help="segundos aguardando respostas de discovery")
    master.add_argument("--rpc", default="",
                        help="lista host:porta explicita, quando nao usar --discover")
    master.add_argument("llama_args", nargs="*",
                        help="argumentos extras repassados ao llama-cli")

    sweep = commands.add_parser(
        "contexto", help="mede o maior contexto que cada configuracao aguenta")
    _add_shared_arguments(sweep)
    sweep.add_argument("--model", required=True, help="caminho do arquivo GGUF")
    sweep.add_argument("--rpc", default="",
                       help="host:porta dos workers; vazio mede so esta maquina")
    sweep.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_CONTEXT_SIZES),
                       help="tamanhos de contexto separados por virgula")
    sweep.add_argument("--timeout", type=float, default=900.0,
                       help="segundos por ponto antes de considerar travado")
    sweep.add_argument("--max-context", type=int, default=0,
                       help="limite do modelo (n_ctx_train); 0 nao corta")
    return parser


def run_worker(options: argparse.Namespace) -> int:
    """Serve this machine as an RPC worker until the server exits."""
    address = options.bind or lan_address()
    worker = RpcWorker(options.node_id, local_worker(options.node_id).hostname,
                       address, options.rpc_port)
    print(_INSECURE, file=sys.stderr)
    server = start_rpc_server(worker, options.cache_dir)
    log.info("worker %s pronto em %s", worker.node_id, worker.endpoint)
    exit_code = serve_until_server_exits(worker, server, options.discovery_port)
    assert_clean_exit(worker, server, exit_code)
    return exit_code


def run_nodes(options: argparse.Namespace) -> int:
    """Print the RPC workers answering on the LAN."""
    workers = discover_workers(options.discovery_port, options.timeout,
                               options.broadcast)
    print(format_workers(workers))
    return 0 if workers else 1


def _resolve_endpoints(options: argparse.Namespace) -> str:
    if options.rpc:
        return options.rpc
    workers = discover_workers(options.discovery_port, options.timeout,
                               options.broadcast)
    print(format_workers(workers), file=sys.stderr)
    return rpc_endpoints(workers)


def run_master(options: argparse.Namespace) -> int:
    """Run llama-cli against the discovered workers."""
    endpoints = _resolve_endpoints(options)
    print(_INSECURE, file=sys.stderr)
    # Devices first: it is the only cheap way to see that a worker contributed
    # its accelerator instead of silently falling back to CPU.
    subprocess.run(list_devices_command(endpoints), check=False)
    # Sem terminal nao ha quem responda ao chat do llama-cli, e ele espera para
    # sempre: um turno so e o unico modo que termina sozinho.
    command = llama_cli_command(options.hf, endpoints, options.llama_args,
                                interactive=sys.stdin.isatty())
    log.info("executando %s", " ".join(command))
    return subprocess.run(command, check=False).returncode


def run_examples(_options: argparse.Namespace) -> int:
    """Print runnable commands and the GGUF repositories worth trying."""
    print(format_llama_examples())
    return 0


def _parse_sizes(specification: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part) for part in specification.split(","))
    except ValueError as exc:
        raise ValueError(
            f"sizes={specification!r}; esperado inteiros separados por virgula") from exc
    if not sizes or min(sizes) <= 0:
        raise ValueError(f"sizes={sizes}; esperado inteiros positivos")
    return sizes


def run_context(options: argparse.Namespace) -> int:
    """Grow the context until it stops loading, and report where it stopped."""
    results = run_context_sweep(options.model, options.rpc,
                                _parse_sizes(options.sizes), options.timeout,
                                options.max_context)
    print(format_sweep(results))
    return 0 if any(result.loaded for result in results) else 1


_COMMANDS = {"worker": run_worker, "nodes": run_nodes, "master": run_master,
             "exemplos": run_examples, "contexto": run_context}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one `split-llama` subcommand.

    Usage example: `main(["nodes", "--broadcast", "192.168.0.255"])`.
    """
    parser = _build_parser()
    options = parser.parse_args(argv)
    if options.comandos_exemplos:
        return run_examples(options)
    if options.command is None:
        parser.print_help()
        return 1
    try:
        return _COMMANDS[options.command](options)
    except ClusterError as error:
        log.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
