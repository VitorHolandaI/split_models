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
import os
import secrets
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ..errors import ClusterError
from .bench import (
    DEFAULT_DEPTHS, DEFAULT_RATIOS, bench_command, device_speed_command)
from .capacity_load import (
    HttpCompletionClient, HttpSlotsSampler, HttpTokenCounter, exact_prompt,
    run_point)
from .capacity_metrics import (
    detect_saturation, summarize_point, write_request_jsonl,
    write_summary_csv)
from .capacity_profiles import (
    PROFILES as CAPACITY_PROFILES, CapacityProfileError, get_profile)
from .controller import ModeController, ModeTransitionError
from .examples import format_llama_examples
from .discovery import (
    DISCOVERY_PORT, RPC_PORT, RpcWorker, discover_workers, local_worker,
    rpc_endpoints)
from .fleet_coordinator import FleetCoordinator, FleetError
from .fleet_inventory import InventoryError, NodeInventory
from .fleet_ssh import FLEET_KEY, SshError, SshKeyRunner
from .health import TcpPortProbe
from .master import DEFAULT_GGUF_REPO, list_devices_command, llama_cli_command
from .mode_state import (
    MODEL_PROFILES, Mode, ModeConfig, ModeConfigError, NodeState)
from .mode_store import (
    RUNTIME_DIR, ModeLockedError, StateError, StateStore)
from .process import (
    RPC_CACHE_DIR, assert_clean_exit, lan_address, serve_until_server_exits,
    start_rpc_server)
from .units import SystemdUnitManager, UnitCommandError

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

    bench = commands.add_parser(
        "bench", help="varre proporcoes e contextos com llama-bench")
    bench.add_argument("--model", required=True, help="caminho do arquivo GGUF")
    bench.add_argument("--rpc", default="",
                       help="host:porta dos workers; vazio mede so esta maquina")
    bench.add_argument("--ratios", default=",".join(DEFAULT_RATIOS),
                       help="razoes -ts, dispositivos separados por barra")
    bench.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS),
                       help="profundidades de contexto separadas por virgula")
    bench.add_argument("--repetitions", type=int, default=3,
                       help="repeticoes por ponto")

    speed = commands.add_parser(
        "velocidade", help="mede rapido esta maquina, para escolher o master")
    speed.add_argument("--model", required=True, help="caminho do arquivo GGUF")
    speed.add_argument("--rpc", default="", help="mede atraves de um worker")

    mode = commands.add_parser(
        "mode", help="troca o papel supervisionado deste no sem reboot")
    mode.add_argument("target",
                      choices=[*(m.value for m in Mode), "status"],
                      help="papel destino; status so le o estado gravado")
    mode.add_argument("--model", choices=sorted(MODEL_PROFILES), default=None,
                      help="perfil oficial do GGUF (master e replica)")
    mode.add_argument("--parallel", type=int, default=4,
                      help="slots/sequencias vivas do llama-server")
    mode.add_argument("--rpc", default="",
                      help="endpoints host:porta separados por virgula; "
                           "obrigatorio no distributed-master")
    mode.add_argument("--port", type=int, default=8080,
                      help="porta HTTP do llama-server")
    mode.add_argument("--keep-model-cache", action="store_true",
                      help="mantem o GGUF em tmpfs ao sair de master/replica")
    mode.add_argument("--health-timeout", type=float, default=900.0,
                      help="segundos esperando GET /health antes do rollback")
    mode.add_argument("--brief", action="store_true",
                      help="status em uma linha, para a barra do tmux")
    _add_runtime_arguments(mode)

    serve = commands.add_parser(
        "serve", help="executa o comando efetivo gravado (uso da unidade "
                      "split-llama-server.service)")
    _add_runtime_arguments(serve)

    fleet = commands.add_parser(
        "fleet", help="coordena a frota por SSH por chave",
        epilog="cenarios: inventory | distributed | replicated | reset")
    fleet.add_argument("--key", default=FLEET_KEY,
                       help="chave privada dedicada da frota")
    fleet.add_argument("--user", default="root",
                       help="usuario SSH nos nos")
    fleet.add_argument("--run-dir", type=Path, default=Path("fleet-runs"),
                       help="onde ficam inventario e registro de cada run")
    fleet.add_argument("--health-timeout", type=float, default=900.0,
                       help="segundos esperando GET /health nos nos")
    fleet.add_argument("--rpc-port", type=int, default=RPC_PORT,
                       help="porta RPC dos workers")
    fleet_sub = fleet.add_subparsers(dest="fleet_command", required=True)

    inventory = fleet_sub.add_parser(
        "inventory", help="coleta o inventario exato dos nos")
    inventory.add_argument("--nodes", required=True,
                           help="hosts SSH separados por virgula")

    distributed = fleet_sub.add_parser(
        "distributed", help="promove um no a distributed-master; os demais "
                            "viram workers RPC")
    distributed.add_argument("--master", required=True,
                             help="host do no que vira master")
    distributed.add_argument("--workers", required=True,
                             help="hosts worker separados por virgula")
    distributed.add_argument("--model", required=True,
                             choices=sorted(MODEL_PROFILES),
                             help="perfil oficial do GGUF")
    distributed.add_argument("--parallel", type=int, default=4,
                             help="slots/sequencias vivas do llama-server")
    distributed.add_argument("--port", type=int, default=8080,
                             help="porta HTTP do master")

    replicated = fleet_sub.add_parser(
        "replicated", help="poe N nos em replica local e gera o HAProxy")
    replicated.add_argument("--replicas", required=True,
                            help="hosts replica separados por virgula")
    replicated.add_argument("--model", required=True,
                            choices=sorted(MODEL_PROFILES),
                            help="perfil oficial do GGUF")
    replicated.add_argument("--parallel", type=int, default=4,
                            help="slots/sequencias vivas do llama-server")
    replicated.add_argument("--port", type=int, default=8080,
                            help="porta HTTP das replicas")
    replicated.add_argument("--haproxy-file", type=Path, default=Path("haproxy.cfg"),
                            help="onde escrever a configuracao do HAProxy")

    reset = fleet_sub.add_parser(
        "reset", help="devolve todos os nos ao modo worker e limpa o estado")
    reset.add_argument("--nodes", required=True,
                       help="hosts SSH separados por virgula")

    bench = fleet_sub.add_parser(
        "bench", help="inventario + promocao + HAProxy + bench-capacity em um comando")
    bench.add_argument("--scenario", required=True,
                       choices=("distributed", "replicated"),
                       help="topologia do benchmark")
    bench.add_argument("--model", required=True,
                       choices=sorted(MODEL_PROFILES),
                       help="perfil oficial do GGUF")
    bench.add_argument("--profile", required=True,
                       choices=sorted(CAPACITY_PROFILES),
                       help="perfil fixo do bench-capacity")
    bench.add_argument("--nodes", default="",
                       help="hosts SSH separados por virgula (ignorado com --auto-discover)")
    bench.add_argument("--master", default="",
                       help="host do master (obrigatorio no cenario distributed)")
    bench.add_argument("--parallel", type=int, default=4,
                       help="slots/sequencias vivas do llama-server")
    bench.add_argument("--port", type=int, default=8080,
                       help="porta HTTP dos servidores inferencia")
    bench.add_argument("--output-dir", type=Path, default=Path("benchmark-results"),
                       help="onde gravar requests.jsonl e summary.csv")
    bench.add_argument("--auto-discover", action="store_true",
                       help="descobre nos via UDP broadcast em vez de --nodes")
    bench.add_argument("--broadcast-address", default="255.255.255.255",
                       help="endereco de broadcast para discovery")
    bench.add_argument("--discovery-timeout", type=float, default=10.0,
                       help="segundos escutando respostas de discovery")
    bench.add_argument("--health-timeout", type=float, default=900.0,
                       help="segundos esperando GET /health")

    bench_capacity = commands.add_parser(
        "bench-capacity",
        help="executa um perfil de capacidade contra llama-server(s)")
    bench_capacity.add_argument(
        "--profile", required=True,
        choices=sorted(CAPACITY_PROFILES),
        help="perfil fixo: users-2k, context-2users, nodes-8users-2k, "
             "capacity-envelope")
    bench_capacity.add_argument(
        "--scenario", required=True, choices=("distributed", "replicated"),
        help="cenario da frota; muda o que os endpoints significam")
    bench_capacity.add_argument(
        "--endpoint", action="append", required=True,
        help="URL base http://host:porta de um llama-server (repetivel)")
    bench_capacity.add_argument(
        "--output-dir", type=Path, default=Path("benchmark-results"),
        help="onde gravar requests.jsonl e summary.csv")
    bench_capacity.add_argument(
        "--health-timeout", type=float, default=900.0,
        help="segundos esperando GET /health antes de medir")
    bench_capacity.add_argument(
        "--runtime-dir", type=Path, default=RUNTIME_DIR,
        help="raiz do estado, para ler o commit do build")
    return parser


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-dir", type=Path, default=RUNTIME_DIR,
                        help="raiz do estado, /run/split-models na appliance")


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


def _parse_depths(specification: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(part) for part in specification.split(","))
    except ValueError as exc:
        raise ValueError(
            f"depths={specification!r}; esperado inteiros separados por virgula") from exc
    if not depths or min(depths) < 0:
        raise ValueError(f"depths={depths}; esperado inteiros nao negativos")
    return depths


def run_bench(options: argparse.Namespace) -> int:
    """Sweep split ratios and context depths with llama-bench."""
    command = bench_command(options.model, options.rpc,
                            tuple(options.ratios.split(",")),
                            _parse_depths(options.depths),
                            repetitions=options.repetitions)
    log.info("executando %s", " ".join(command))
    return subprocess.run(command, check=False).returncode


def run_speed(options: argparse.Namespace) -> int:
    """Measure this machine quickly, to decide which node should be master."""
    command = device_speed_command(options.model, options.rpc)
    log.info("executando %s", " ".join(command))
    return subprocess.run(command, check=False).returncode


def _mode_config_from(options: argparse.Namespace) -> ModeConfig:
    mode = Mode.from_text(options.target)
    model = MODEL_PROFILES[options.model] if options.model else None
    endpoints = tuple(part.strip() for part in options.rpc.split(",")
                      if part.strip())
    return ModeConfig(mode, model, options.parallel, endpoints, options.port)


def _mode_controller(options: argparse.Namespace) -> ModeController:
    # O probe da porta RPC precisa do endereco que o worker usou de bind; sem
    # rota de LAN o loopback ainda distingue porta aberta de fechada.
    try:
        bind = lan_address()
    except OSError:
        bind = "127.0.0.1"
    return ModeController(StateStore(options.runtime_dir), SystemdUnitManager(),
                          port_probe=TcpPortProbe(bind),
                          health_timeout=options.health_timeout)


def _status_line(state: NodeState) -> str:
    try:
        address = lan_address()
    except OSError:
        address = "unavailable"
    return (f"role={state.mode.value} | node=node-{socket.gethostname()} | "
            f"ip={address}")


def _print_status(store: StateStore, brief: bool) -> None:
    state = store.read_state()
    if brief:
        print(_status_line(state))
        return
    lines = [f"modo: {state.mode.value}"]
    if state.model:
        lines.append(f"modelo: {state.model}")
        lines.append(f"parallel: {state.parallel}")
        lines.append(f"porta http: {state.port}")
    if state.rpc_endpoints:
        lines.append(f"rpc: {','.join(state.rpc_endpoints)}")
    try:
        lines.append(f"comando efetivo: {' '.join(store.read_command())}")
    except StateError:
        pass
    print("\n".join(lines))


def run_mode(options: argparse.Namespace) -> int:
    """Drive one locked mode transition, or print the recorded state."""
    store = StateStore(options.runtime_dir)
    if options.target == "status":
        _print_status(store, options.brief)
        return 0
    try:
        config = _mode_config_from(options)
        state = _mode_controller(options).apply(
            config, keep_model_cache=options.keep_model_cache)
    except (ModeConfigError, ModeLockedError, ModeTransitionError,
            UnitCommandError) as error:
        log.error("%s", error)
        return 1
    log.info("modo atual: %s", state.mode.value)
    return 0


def run_serve(options: argparse.Namespace) -> int:
    """Exec the recorded effective command; the server unit's only job.

    os.execvp replaces this process, so llama-server becomes the unit's main
    PID and systemd watches the right process.
    """
    try:
        argv = StateStore(options.runtime_dir).read_command()
    except (StateError, ValueError) as error:
        log.error("%s", error)
        return 1
    log.info("executando %s", " ".join(argv))
    try:
        os.execvp(argv[0], argv)
    except OSError as error:
        log.error("exec %s: %s", argv[0], error)
        return 1
    return 0  # inalcancavel: execvp substitui o processo


def _split_hosts(specification: str) -> tuple[str, ...]:
    hosts = tuple(part.strip() for part in specification.split(",")
                  if part.strip())
    if not hosts:
        raise FleetError(
            f"hosts={specification!r}; esperado lista separada por virgula")
    return hosts


_INVENTORY_HEADER = f"{'NODE':<16} {'HOSTNAME':<16} {'IP':<16} {'MODE':<18} BUILD"


def format_inventory(nodes: tuple[NodeInventory, ...]) -> str:
    """Render the collected inventory as a plain-text table for the operator.

    Usage example: `print(format_inventory(nodes))`.
    """
    rows = [f"{node.node_id:<16} {node.hostname:<16} {node.ip:<16} "
            f"{node.mode:<18} {node.build_commit[:12]}"
            for node in nodes]
    return "\n".join([_INVENTORY_HEADER, *rows])


def run_fleet(options: argparse.Namespace) -> int:
    """Drive one fleet scenario over SSH and report the run artifacts."""
    coordinator = FleetCoordinator(
        SshKeyRunner(options.key, user=options.user),
        run_dir=options.run_dir, health_timeout=options.health_timeout,
        rpc_port=options.rpc_port)
    try:
        if options.fleet_command == "inventory":
            inventory_path, nodes = coordinator.collect_inventory(
                _split_hosts(options.nodes))
            print(format_inventory(nodes))
            print(f"inventario: {inventory_path}")
            return 0
        if options.fleet_command == "distributed":
            result = coordinator.promote_master(
                options.master, _split_hosts(options.workers), options.model,
                parallel=options.parallel, port=options.port)
            print(f"master pronto; inventario: {result.inventory_path}")
            print(f"registro: {result.run_path}")
            for endpoint, node_id, hostname in result.contributions:
                print(f"{endpoint} <- {node_id} ({hostname})")
            return 0
        if options.fleet_command == "replicated":
            result = coordinator.configure_replicas(
                _split_hosts(options.replicas), options.model,
                parallel=options.parallel, port=options.port,
                haproxy_path=options.haproxy_file)
            print(f"replicas prontas; inventario: {result.inventory_path}")
            print(f"haproxy: {result.haproxy_path}")
            return 0
        if options.fleet_command == "bench":
            return _run_fleet_bench(coordinator, options)
        result = coordinator.reset(_split_hosts(options.nodes))
        print("nos em modo worker")
        print(f"registro: {result}")
        return 0
    except (FleetError, SshError, InventoryError, ModeConfigError,
            OSError, ValueError) as error:
        log.error("%s", error)
        return 1


def _run_fleet_bench(coordinator: FleetCoordinator,
                     options: argparse.Namespace) -> int:
    """Inventory -> promote -> HAProxy -> bench-capacity -> cleanup."""
    from .discovery import discover_workers
    from .fleet_ssh import SshKeyRunner
    from .health import HttpHealthProbe, wait_until
    from .mode_state import Mode, ModeConfig, MODEL_PROFILES

    if options.auto_discover:
        found = discover_workers(timeout=options.discovery_timeout,
                                 broadcast_address=options.broadcast_address)
        nodes = tuple(w.address for w in found)
        if not nodes:
            log.error("nenhum no descoberto via UDP broadcast em %ss",
                      options.discovery_timeout)
            return 1
        log.info("descobertos: %s", nodes)
        runner = SshKeyRunner(FLEET_KEY, user="root")
        model_repo = MODEL_PROFILES[options.model]
        for node in nodes:
            remote = f"root@{node}"
            log.info("promovendo %s a %s", remote, options.scenario)
            runner.run(remote, "systemctl stop split-llama-worker.service")
            if options.scenario == "distributed":
                rpc = ",".join(f"{h}:{options.rpc_port or 50052}" for h in nodes if h != node)
                runner.run(remote,
                           f"split-llama mode distributed-master "
                           f"--model {options.model} --rpc {rpc} "
                           f"--parallel {options.parallel} --port {options.port}")
            else:
                runner.run(remote,
                           f"split-llama mode replica "
                           f"--model {options.model} "
                           f"--parallel {options.parallel} --port {options.port}")
        if options.scenario == "replicated":
            servers = tuple((f"node-{i}", f"{h}:{options.port}")
                            for i, h in enumerate(nodes))
            from .haproxy import haproxy_config
            haproxy_path = options.output_dir / "haproxy.cfg"
            haproxy_path.parent.mkdir(parents=True, exist_ok=True)
            haproxy_path.write_text(haproxy_config(servers, options.port))
            proxy = subprocess.Popen(
                ["haproxy", "-f", str(haproxy_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            endpoint = f"http://127.0.0.1:{options.port}"
        else:
            proxy = None
            endpoint = f"http://{nodes[0]}:{options.port}"
    else:
        if not options.nodes:
            log.error("informe --nodes ou --auto-discover")
            return 1
        nodes = _split_hosts(options.nodes)
        proxy = None
        if options.scenario == "distributed":
            master = options.master or nodes[0]
            workers = tuple(h for h in nodes if h != master)
            result = coordinator.promote_master(
                master, workers, options.model,
                parallel=options.parallel, port=options.port)
            endpoint = f"http://{master}:{options.port}"
        else:
            result = coordinator.configure_replicas(
                nodes, options.model,
                parallel=options.parallel, port=options.port,
                haproxy_path=options.output_dir / "haproxy.cfg")
            proxy = subprocess.Popen(
                ["haproxy", "-f", str(result.haproxy_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            endpoint = f"http://127.0.0.1:{options.port}"

    probe = HttpHealthProbe()
    ready = wait_until(
        lambda: probe.ready(f"{endpoint}/health"),
        timeout=options.health_timeout)
    if not ready:
        log.error("%s nao ficou pronto apos %ss", endpoint, options.health_timeout)
        return 1

    bench_opts = argparse.Namespace(
        profile=options.profile, scenario=options.scenario,
        endpoint=[endpoint], output_dir=options.output_dir,
        health_timeout=options.health_timeout)
    exit_code = run_bench_capacity(bench_opts)

    if proxy is not None:
        proxy.terminate()
        try:
            proxy.wait(5)
        except subprocess.TimeoutExpired:
            proxy.kill()
    return exit_code


def run_bench_capacity(options: argparse.Namespace) -> int:
    """Run one fixed capacity profile against the fleet and write artifacts."""
    from .health import HttpHealthProbe, wait_until

    try:
        profile = get_profile(options.profile)
        clients = [HttpCompletionClient(endpoint)
                   for endpoint in options.endpoint]
        probe = HttpHealthProbe()
        ready = wait_until(
            lambda: all(probe.ready(f"{endpoint.rstrip('/')}/health")
                        for endpoint in options.endpoint),
            timeout=options.health_timeout)
        if not ready:
            log.error("endpoints nao ficaram prontos: %s", options.endpoint)
            return 1
        token_counter = HttpTokenCounter(options.endpoint[0])
        results = []
        for point in profile.points:
            prompt = exact_prompt(point.context, token_counter)
            result = run_point(
                profile, options.scenario, point, clients,
                sampler=HttpSlotsSampler(options.endpoint[0]),
                sleep=lambda _seconds: None)
            results.append(result)
            log.info("ponto %s/%s: %s ok, %s falhas, %s tok/s",
                     point, len(profile.points),
                     len([r for r in result.records if r.ok]),
                     len([r for r in result.records if not r.ok]),
                     round(summarize_point(result).total_tok_s, 2))
        flags = detect_saturation(
            [summarize_point(result) for result in results])
        summaries = [summarize_point(result, flag)
                     for result, flag in zip(results, flags)]
        options.output_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summaries, options.output_dir / "capacity-summary.csv")
        for result in results:
            write_request_jsonl(
                result.records,
                options.output_dir / f"capacity-{result.profile}-requests.jsonl")
        print(f"resumo: {options.output_dir / 'capacity-summary.csv'}")
        return 0
    except (CapacityProfileError, OSError, ValueError) as error:
        log.error("%s", error)
        return 1


_COMMANDS = {"worker": run_worker, "nodes": run_nodes, "master": run_master,
             "exemplos": run_examples, "bench": run_bench,
             "velocidade": run_speed, "mode": run_mode, "serve": run_serve,
             "fleet": run_fleet, "bench-capacity": run_bench_capacity}


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
