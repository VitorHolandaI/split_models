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
from .distributed_planner import (
    CapacityPlan, CapacityPlanError, RankedNode, allocate_model,
    build_capacity_plan, known_model_gigabytes, known_model_layers)
from .examples import format_llama_examples
from .discovery import (
    DISCOVERY_PORT, RPC_PORT, RpcWorker, discover_workers, local_worker,
    rpc_endpoints)
from .fleet_coordinator import FleetCoordinator, FleetError
from .fleet_discovery import (
    FLEET_DISCOVERY_PORT, FleetNode, discover_fleet_nodes,
    serve_fleet_discovery)
from .fleet_inventory import (
    InventoryError, NodeInventory, inventory_sections_command,
    parse_node_inventory)
from .fleet_ssh import FLEET_KEY, SshError, SshKeyRunner, SshRunner
from .health import TcpPortProbe
from .hardware_benchmark import (
    HardwareBenchmarkError, HardwareScore, format_hardware_score,
    run_hardware_benchmark)
from .master import DEFAULT_GGUF_REPO, list_devices_command, llama_cli_command
from .model_cache import select_cached_model
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
_MODEL_CACHE_HELPER = Path("/usr/local/bin/split-llama-model-cache")


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
        epilog="""Exemplo de planejamento sem efeitos colaterais:
  split-llama master --discover --dry-run \\
    --hf ggml-org/gpt-oss-20b-GGUF:MXFP4

Para modelos conhecidos, as camadas sao detectadas automaticamente.
--model-layers serve apenas como override para modelos fora do catalogo.

Argumentos apos -- vao direto para o llama-cli.""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_shared_arguments(master)
    master.add_argument("--discover", action="store_true",
                        help="descobre os workers em vez de receber --rpc pronto")
    master.add_argument("--hf", default=DEFAULT_GGUF_REPO,
                        help="repositorio GGUF no HuggingFace")
    master.add_argument("--timeout", type=float, default=3.0,
                        help="segundos aguardando respostas de discovery")
    master.add_argument("--rpc", default="",
                        help="lista host:porta explicita, quando nao usar --discover")
    master.add_argument("--usb-device", default="",
                        help="particao USB ext4 do cache, por exemplo /dev/sda1")
    master.add_argument("--model-size-gb", type=float, default=None,
                        help="tamanho medido para GGUF fora do catalogo")
    master.add_argument("--machines", type=int, default=None,
                        help="total incluindo master; sem valor pergunta no TTY")
    master.add_argument("--dry-run", action="store_true",
                        help="calcula o plano sem mudar nos nem iniciar inferencia")
    master.add_argument("--model-layers", type=int, default=None,
                        help="override de camadas para modelo fora do catalogo")
    master.add_argument("--hardware-timeout", type=int, default=900,
                        help="timeout por no para hardware-bench small")
    master.add_argument("llama_args", nargs="*",
                        help="argumentos extras repassados ao llama-cli")
    _add_runtime_arguments(master)

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

    hardware = commands.add_parser(
        "hardware-bench", help="mede CPU, iGPU e llama.cpp small nesta maquina")
    hardware.add_argument("--json", action="store_true",
                          help="emite um registro JSON para coleta da frota")
    hardware.add_argument("--cache-dir", type=Path,
                          default=RUNTIME_DIR / "hardware-bench-cache",
                          help="cache temporario do modelo small")
    _add_runtime_arguments(hardware)

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
                           "obrigatorio no distributed-master sem autodiscovery")
    mode.add_argument("--auto-discover", action="store_true",
                      help="descobre a frota e monta os endpoints RPC")
    mode.add_argument("--broadcast", default="255.255.255.255",
                      help="broadcast da LAN para autodiscovery")
    mode.add_argument("--discovery-timeout", type=float, default=5.0,
                      help="segundos aguardando respostas de autodiscovery")
    mode.add_argument("--usb-device", default="",
                      help="particao USB ext4 do cache, por exemplo /dev/sda1")
    mode.add_argument("--port", type=int, default=8080,
                      help="porta HTTP do llama-server")
    mode.add_argument("--keep-model-cache", action="store_true",
                      help="mantem o cache RAM/USB selecionado ao sair do modo")
    mode.add_argument("--health-timeout", type=float, default=900.0,
                      help="segundos esperando GET /health antes do rollback")
    mode.add_argument("--brief", action="store_true",
                      help="status em uma linha, para a barra do tmux")
    _add_runtime_arguments(mode)

    become_master = commands.add_parser(
        "become_master",
        help="para inferencia local e prepara este no como coordenador/master")
    become_master.add_argument("--health-timeout", type=float, default=900.0,
                               help="timeout do controlador de modo")
    _add_runtime_arguments(become_master)

    serve = commands.add_parser(
        "serve", help="executa o comando efetivo gravado (uso da unidade "
                      "split-llama-server.service)")
    _add_runtime_arguments(serve)

    fleet_agent = commands.add_parser(
        "fleet-agent", help="anuncia este no para controle em qualquer modo")
    fleet_agent.add_argument("--node-id", default=f"node-{socket.gethostname()}",
                             help="identificador estavel deste no")
    fleet_agent.add_argument("--discovery-port", type=int,
                             default=FLEET_DISCOVERY_PORT,
                             help="porta UDP do plano de controle")
    _add_runtime_arguments(fleet_agent)

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

    fleet_nodes = fleet_sub.add_parser(
        "nodes", help="lista todos os nos pelo discovery de controle")
    fleet_nodes.add_argument("--broadcast-address", default="255.255.255.255",
                             help="endereco de broadcast para discovery")
    fleet_nodes.add_argument("--discovery-timeout", type=float, default=5.0,
                             help="segundos escutando respostas de discovery")
    fleet_nodes.add_argument("--discovery-port", type=int,
                             default=FLEET_DISCOVERY_PORT,
                             help="porta UDP do plano de controle")

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
    reset.add_argument("--nodes", default="",
                       help="hosts SSH separados por virgula")
    reset.add_argument("--auto-discover", action="store_true",
                       help="descobre todos os nos pelo plano de controle")
    reset.add_argument("--broadcast-address", default="255.255.255.255",
                       help="endereco de broadcast para discovery")
    reset.add_argument("--discovery-timeout", type=float, default=5.0,
                       help="segundos escutando respostas de discovery")
    reset.add_argument("--discovery-port", type=int,
                       default=FLEET_DISCOVERY_PORT,
                       help="porta UDP do plano de controle")

    reboot = fleet_sub.add_parser(
        "reboot", help="reinicia todos os nos descobertos via UDP broadcast")
    reboot.add_argument("--broadcast-address", default="255.255.255.255",
                        help="endereco de broadcast para discovery")
    reboot.add_argument("--discovery-timeout", type=float, default=10.0,
                        help="segundos escutando respostas de discovery")
    reboot.add_argument("--discovery-port", type=int,
                        default=FLEET_DISCOVERY_PORT,
                        help="porta UDP do plano de controle")

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
    bench.add_argument("--haproxy-port", type=int, default=8088,
                       help="porta local do HAProxy no cenario replicated")
    bench.add_argument("--output-dir", type=Path, default=Path("benchmark-results"),
                       help="onde gravar requests.jsonl e summary.csv")
    bench.add_argument("--auto-discover", action="store_true",
                       help="descobre nos via UDP broadcast em vez de --nodes")
    bench.add_argument("--broadcast-address", default="255.255.255.255",
                       help="endereco de broadcast para discovery")
    bench.add_argument("--discovery-timeout", type=float, default=10.0,
                       help="segundos escutando respostas de discovery")
    bench.add_argument("--discovery-port", type=int,
                       default=FLEET_DISCOVERY_PORT,
                       help="porta UDP do plano de controle")
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


def _restore_discovered_workers(options: argparse.Namespace,
                                runner: SshRunner | None = None
                                ) -> tuple[str, ...]:
    """Return every control-discovered node to RPC worker mode over SSH."""
    found = discover_fleet_nodes(FLEET_DISCOVERY_PORT, options.timeout,
                                 options.broadcast)
    hosts = tuple(node.address for node in found)
    if not hosts:
        log.warning("nenhum no respondeu ao discovery de controle")
        return ()
    ssh = runner or SshKeyRunner(_resolve_fleet_key(), user="root")
    _return_all_to_worker(ssh, hosts)
    return hosts


def _discover_master_hosts(options: argparse.Namespace) -> tuple[str, ...]:
    found = discover_fleet_nodes(FLEET_DISCOVERY_PORT, options.timeout,
                                 options.broadcast)
    if not found:
        log.warning("nenhum no respondeu ao discovery de controle")
        return ()
    runner = SshKeyRunner(_resolve_fleet_key(), user="root")
    return _select_master_hosts(options, found, runner)


def _build_master_capacity_plan(options: argparse.Namespace,
                                found: tuple[FleetNode, ...],
                                runner: SshRunner,
                                restore_workers: bool = True
                                ) -> CapacityPlan:
    hosts = tuple(node.address for node in found)
    if restore_workers:
        _return_all_to_worker(runner, hosts)
    ranked = tuple(_measure_ranked_node(host, runner, options.hardware_timeout)
                   for host in hosts)
    model_gb = (known_model_gigabytes(options.hf)
                if options.model_size_gb is None else options.model_size_gb)
    return build_capacity_plan(model_gb, ranked, lan_address())


def _select_master_hosts(options: argparse.Namespace,
                         found: tuple[FleetNode, ...],
                         runner: SshRunner) -> tuple[str, ...]:
    plan = _build_master_capacity_plan(options, found, runner)
    total = _machine_total(plan, options.machines)
    selected = plan.select(total)
    _log_capacity_plan(plan, selected)
    return tuple(node.inventory.host for node in selected)


def _measure_ranked_node(host: str, runner: SshRunner,
                         timeout: int) -> RankedNode:
    inventory = parse_node_inventory(
        runner.run(host, inventory_sections_command()).stdout, host)
    command = ["timeout", str(timeout), "split-llama", "hardware-bench", "--json"]
    score = HardwareScore.from_json(runner.run(host, command).stdout)
    log.info("%s: %.2f TOPS, %d MIPS, %.2f Vulkan tok/s", host,
             score.int8_tops, score.cpu_mips, score.vulkan_tokens_per_second)
    return RankedNode(inventory, score)


def _machine_total(plan: CapacityPlan, requested: int | None) -> int:
    if requested is not None:
        return requested
    maximum = len(plan.ordered_nodes)
    if not sys.stdin.isatty() or plan.minimum_total == maximum:
        return plan.minimum_total
    answer = input(
        f"Modelo {plan.model_gigabytes:.2f} GB: minimo {plan.minimum_total}, "
        f"disponiveis {maximum}. Quantas maquinas no total? "
        f"[{plan.minimum_total}]: ").strip()
    return int(answer) if answer else plan.minimum_total


def _log_capacity_plan(plan: CapacityPlan,
                       selected: tuple[RankedNode, ...]) -> None:
    """Log the safe selection and print the measured fleet table."""
    log.info("modelo %.2f GB; minimo seguro=%d; selecionadas=%d",
             plan.model_gigabytes, plan.minimum_total, len(selected))
    print(format_capacity_table(selected))


def format_capacity_table(selected: tuple[RankedNode, ...]) -> str:
    """Render RAM, TOPS, MIPS, and tok/s per selected node for the operator."""
    rows = [f"{'NODE':<14} {'RAM GiB':>8} {'TOPS':>7} {'MIPS':>7} "
            f"{'tok/s CPU':>10} {'tok/s Vulkan':>12}"]
    for node in selected:
        hardware = node.hardware
        rows.append(
            f"{node.inventory.host:<14} "
            f"{node.inventory.mem_total_kb / (1024 * 1024):>8.1f} "
            f"{hardware.int8_tops:>7.2f} {hardware.cpu_mips:>7d} "
            f"{hardware.cpu_tokens_per_second:>10.2f} "
            f"{hardware.vulkan_tokens_per_second:>12.2f}")
    return "\n".join(rows)


def format_capacity_plan(plan: CapacityPlan,
                         selected: tuple[RankedNode, ...],
                         model_layers: int | None
                         ) -> str:
    """Render the dry-run model budget and estimated contiguous layer slices."""
    allocations = allocate_model(plan, selected, model_layers)
    safe_total = sum(node.safe_model_gib for node in selected)
    selected_hosts = {node.inventory.host for node in selected}
    unused_hosts = [node.inventory.host for node in plan.ordered_nodes
                    if node.inventory.host not in selected_hosts]
    lines = [
        "DRY-RUN: nenhum no sera alterado, nenhum download sera feito",
        f"modelo: {plan.model_gigabytes:.2f} GB | capacidade segura: "
        f"{safe_total:.2f} GiB | minimo: {plan.minimum_total} maquinas",
        f"plano: {len(selected)} de {len(plan.ordered_nodes)} maquinas "
        "(master incluso)",
        f"{'PAPEL':<8} {'HOST':<16} {'RAM GiB':>8} {'SEGURA':>8} "
        f"{'MODELO':>8} {'CAMADAS':>12}",
    ]
    for index, allocation in enumerate(allocations):
        layer_range = ("n/d" if allocation.layer_start is None else
                       f"{allocation.layer_start}-{allocation.layer_end}")
        lines.append(
            f"{'master' if index == 0 else 'worker':<8} "
            f"{allocation.node.inventory.host:<16} "
            f"{allocation.node.inventory.mem_total_kb / (1024 * 1024):>8.1f} "
            f"{allocation.node.safe_model_gib:>8.2f} "
            f"{allocation.model_gigabytes:>8.2f} {layer_range:>12}")
    if unused_hosts:
        lines.append(f"fora do plano minimo: {', '.join(unused_hosts)}")
    if model_layers is None:
        lines.append("camadas: use --model-layers N para exibir faixas; "
                     "MODELO e uma estimativa por capacidade")
    else:
        lines.append("camadas: faixas proporcionais estimadas; o llama.cpp "
                     "pode ajustar pelos tamanhos reais dos tensores")
    return "\n".join(lines)


def _dry_run_model_layers(options: argparse.Namespace) -> int | None:
    if options.model_layers is not None:
        return options.model_layers
    try:
        return known_model_layers(options.hf)
    except CapacityPlanError:
        return None


def _run_master_dry_run(options: argparse.Namespace) -> int:
    if not options.discover:
        log.error("--dry-run exige --discover para calcular a frota")
        return 1
    found = discover_fleet_nodes(options.discovery_port, options.timeout,
                                 options.broadcast)
    if not found:
        log.error("nenhum no respondeu ao discovery de controle")
        return 1
    runner = SshKeyRunner(_resolve_fleet_key(), user="root")
    try:
        plan = _build_master_capacity_plan(options, found, runner, False)
        total = options.machines or plan.minimum_total
        selected = plan.select(total)
        print(format_capacity_plan(plan, selected,
                                   _dry_run_model_layers(options)))
        return 0
    except (CapacityPlanError, SshError, OSError, ValueError) as error:
        log.error("dry-run: %s", error)
        return 1


def _resolve_endpoints(options: argparse.Namespace) -> str:
    if options.rpc:
        return options.rpc
    selected_hosts: tuple[str, ...] = ()
    if options.discover:
        selected_hosts = _discover_master_hosts(options)
    workers = discover_workers(options.discovery_port, options.timeout,
                               options.broadcast)
    if selected_hosts:
        workers = tuple(worker for worker in workers
                        if worker.address in selected_hosts)
    print(format_workers(workers), file=sys.stderr)
    return rpc_endpoints(workers)


def _set_model_cache_device(runtime_dir: Path, device: str) -> None:
    selector = runtime_dir / "model-cache-device"
    if not device:
        if runtime_dir.exists():
            selector.unlink(missing_ok=True)
        return
    runtime_dir.mkdir(parents=True, exist_ok=True)
    selector.write_text(device + "\n", encoding="utf-8")


def _configure_direct_mode_discovery(options: argparse.Namespace) -> None:
    if options.target != Mode.DISTRIBUTED_MASTER.value:
        raise ModeConfigError(
            f"--auto-discover com target={options.target!r}; esperado "
            "distributed-master")
    found = discover_fleet_nodes(FLEET_DISCOVERY_PORT,
                                 options.discovery_timeout, options.broadcast)
    local_id = _local_node_id()
    remote_hosts = tuple(node.address for node in found
                         if node.node_id != local_id)
    if not remote_hosts:
        raise ModeConfigError(
            "autodiscovery encontrou nenhum worker remoto; esperado pelo "
            "menos um outro no")
    runner = SshKeyRunner(_resolve_fleet_key(), user="root")
    _return_all_to_worker(runner, remote_hosts)
    workers = discover_workers(DISCOVERY_PORT, options.discovery_timeout,
                               options.broadcast)
    remote_ids = {node.node_id for node in found if node.node_id != local_id}
    endpoints = tuple(worker.endpoint for worker in workers
                      if worker.node_id in remote_ids)
    if not endpoints:
        raise ModeConfigError(
            f"autodiscovery nao encontrou RPC dos nos {remote_hosts!r}")
    options.rpc = rpc_endpoints(tuple(
        worker for worker in workers if worker.node_id in remote_ids))


def _prepare_master_model_cache(model: str,
                                model_gigabytes: float | None = None,
                                runtime_dir: Path = RUNTIME_DIR,
                                usb_device: str = ""
                                ) -> dict[str, str] | None:
    _set_model_cache_device(runtime_dir, usb_device)
    cache_setting = os.environ.get("LLAMA_CACHE")
    if cache_setting:
        cache_dir = Path(cache_setting)
    elif _MODEL_CACHE_HELPER.exists():
        subprocess.run([str(_MODEL_CACHE_HELPER)], check=True)
        cache_dir = StateStore().model_cache_dir
    else:
        return None
    select_cached_model(cache_dir, model, model_gigabytes)
    return {**os.environ, "LLAMA_CACHE": str(cache_dir)}


def _optional_model_gigabytes(options: argparse.Namespace) -> float | None:
    override = getattr(options, "model_size_gb", None)
    if override is not None:
        return float(override)
    try:
        return known_model_gigabytes(options.hf)
    except CapacityPlanError:
        return None


def run_master(options: argparse.Namespace) -> int:
    """Run llama-cli against the discovered workers."""
    if getattr(options, "dry_run", False):
        return _run_master_dry_run(options)
    try:
        endpoints = _resolve_endpoints(options)
        environment = _prepare_master_model_cache(
            options.hf, _optional_model_gigabytes(options),
            getattr(options, "runtime_dir", RUNTIME_DIR),
            getattr(options, "usb_device", ""))
    except (OSError, SshError, ValueError, subprocess.CalledProcessError) as error:
        log.error("%s", error)
        return 1
    print(_INSECURE, file=sys.stderr)
    # Devices first: it is the only cheap way to see that a worker contributed
    # its accelerator instead of silently falling back to CPU.
    subprocess.run(list_devices_command(endpoints), check=False)
    # Sem terminal nao ha quem responda ao chat do llama-cli, e ele espera para
    # sempre: um turno so e o unico modo que termina sozinho.
    command = llama_cli_command(options.hf, endpoints, options.llama_args,
                                interactive=sys.stdin.isatty())
    log.info("executando %s", " ".join(command))
    return subprocess.run(command, check=False, env=environment).returncode


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


def run_hardware(options: argparse.Namespace) -> int:
    """Measure synthetic CPU/iGPU peaks and effective small-model throughput.

    The score is also recorded under the runtime dir so this node's own tmux
    panel can show the result on its screen.
    """
    try:
        score = run_hardware_benchmark(cache_dir=options.cache_dir)
    except (HardwareBenchmarkError, OSError) as error:
        log.error("%s", error)
        return 1
    try:
        StateStore(options.runtime_dir).write_hardware_score(score.to_record())
    except OSError as error:
        log.warning("score nao gravado para o painel: %s", error)
    print(score.to_json() if options.json else format_hardware_score(score))
    return 0


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


def _local_node_id(machine_id_path: Path = Path("/etc/machine-id")) -> str:
    """Return the same stable ID used by systemd's `node-%m` service argument."""
    try:
        machine_id = machine_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        machine_id = ""
    return f"node-{machine_id or socket.gethostname()}"


def _status_line(state: NodeState) -> str:
    try:
        address = lan_address()
    except OSError:
        address = "unavailable"
    return (f"role={state.mode.value} | node={_local_node_id()} | "
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
        if options.auto_discover:
            _configure_direct_mode_discovery(options)
        _set_model_cache_device(options.runtime_dir, options.usb_device)
        config = _mode_config_from(options)
        state = _mode_controller(options).apply(
            config, keep_model_cache=options.keep_model_cache)
    except (ModeConfigError, ModeLockedError, ModeTransitionError,
            UnitCommandError) as error:
        log.error("%s", error)
        return 1
    log.info("modo atual: %s", state.mode.value)
    return 0


def run_become_master(options: argparse.Namespace) -> int:
    """Stop local inference roles while leaving fleet discovery available."""
    try:
        _set_model_cache_device(options.runtime_dir, "")
        _mode_controller(options).apply(ModeConfig(Mode.IDLE))
    except (ModeLockedError, ModeTransitionError, UnitCommandError,
            OSError, subprocess.CalledProcessError) as error:
        log.error("%s", error)
        return 1
    log.info("no pronto como coordenador; discovery UDP de controle permanece ativo")
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


def run_fleet_agent(options: argparse.Namespace) -> int:
    """Advertise this node for fleet control independently of its RPC role."""
    serve_fleet_discovery(options.node_id, StateStore(options.runtime_dir),
                          port=options.discovery_port)
    return 0


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


def format_fleet_nodes(nodes: tuple[FleetNode, ...]) -> str:
    """Render always-on control nodes and their current inference modes."""
    if not nodes:
        return "nenhum no respondeu ao discovery de controle"
    header = f"{'NODE':<20} {'HOSTNAME':<20} {'IP':<16} MODE"
    rows = [f"{node.node_id:<20} {node.hostname:<20} "
            f"{node.address:<16} {node.mode}" for node in nodes]
    return "\n".join([header, *rows])


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
        if options.fleet_command == "nodes":
            return _run_fleet_nodes(options)
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
        if options.fleet_command == "reboot":
            return _run_fleet_reboot(coordinator, options)
        result = coordinator.reset(_reset_hosts(options))
        print("nos em modo worker")
        print(f"registro: {result}")
        return 0
    except (FleetError, SshError, InventoryError, ModeConfigError,
            OSError, ValueError) as error:
        log.error("%s", error)
        return 1


def _resolve_fleet_key() -> str:
    """Pick the fleet key: repo copy (operator machine) or ISO copy.

    The operator runs from the cloned repo, where the private key lives at
    `deploy/keys/split-models-appliance`. A node booting the ISO has no repo
    and uses the key copied to `FLEET_KEY`. Prefer the repo one so running
    from the operator machine needs zero preparation.
    """
    repo_key = Path("deploy/keys/split-models-appliance")
    if repo_key.exists():
        return str(repo_key)
    return FLEET_KEY


def _reset_hosts(options: argparse.Namespace) -> tuple[str, ...]:
    if not options.auto_discover:
        if not options.nodes:
            raise FleetError("informe --nodes ou --auto-discover")
        return _split_hosts(options.nodes)
    found = discover_fleet_nodes(options.discovery_port,
                                 options.discovery_timeout,
                                 options.broadcast_address)
    hosts = tuple(node.address for node in found)
    if not hosts:
        raise FleetError("nenhum no descoberto via UDP de controle")
    return hosts


def _run_fleet_nodes(options: argparse.Namespace) -> int:
    nodes = discover_fleet_nodes(options.discovery_port,
                                 options.discovery_timeout,
                                 options.broadcast_address)
    print(format_fleet_nodes(nodes))
    return 0 if nodes else 1


def _run_fleet_reboot(coordinator: FleetCoordinator,
                      options: argparse.Namespace) -> int:
    """Discover every control agent and reboot it over SSH."""
    from .fleet_ssh import SshKeyRunner

    found = discover_fleet_nodes(options.discovery_port,
                                 options.discovery_timeout,
                                 options.broadcast_address)
    nodes = tuple(node.address for node in found)
    if not nodes:
        log.error("nenhum no descoberto via UDP broadcast em %ss",
                  options.discovery_timeout)
        return 1
    runner = SshKeyRunner(_resolve_fleet_key(), user="root")
    log.info("reiniciando %s", nodes)
    for node in nodes:
        try:
            runner.run(node, ["reboot"])
            log.info("%s: reboot enviado", node)
        except SshError as error:
            log.error("%s", error)
            return 1
    return 0


def _return_all_to_worker(runner: SshRunner,
                          nodes: tuple[str, ...]) -> None:
    for node in nodes:
        log.info("%s: voltando a worker", node)
        runner.run(node, ["split-llama", "mode", "worker"])


def _configure_distributed(runner: SshRunner, nodes: tuple[str, ...],
                           options: argparse.Namespace) -> str:
    if len(nodes) < 2:
        raise FleetError(
            f"nodes={nodes!r}; cenario distributed espera master e worker")
    local = lan_address()
    master = options.master or (local if local in nodes else nodes[0])
    if master not in nodes:
        raise FleetError(
            f"master={master!r}; esperado um dos nos descobertos {nodes!r}")
    _return_all_to_worker(runner, nodes)
    workers = tuple(node for node in nodes if node != master)
    rpc = ",".join(f"{node}:{options.rpc_port}" for node in workers)
    runner.run(master, [
        "split-llama", "mode", "distributed-master",
        "--model", options.model, "--rpc", rpc,
        "--parallel", str(options.parallel), "--port", str(options.port)])
    return f"http://{master}:{options.port}"


def _start_replicated(runner: SshRunner, nodes: tuple[str, ...],
                      options: argparse.Namespace) -> tuple[str, subprocess.Popen[bytes]]:
    from .haproxy import haproxy_config

    _return_all_to_worker(runner, nodes)
    for node in nodes:
        runner.run(node, [
            "split-llama", "mode", "replica", "--model", options.model,
            "--parallel", str(options.parallel), "--port", str(options.port)])
    servers = tuple((f"node-{index}", f"{host}:{options.port}")
                    for index, host in enumerate(nodes))
    haproxy_path = options.output_dir.resolve() / "haproxy.cfg"
    haproxy_path.parent.mkdir(parents=True, exist_ok=True)
    haproxy_path.write_text(
        haproxy_config(servers, options.haproxy_port), encoding="utf-8")
    proxy = subprocess.Popen(["haproxy", "-f", str(haproxy_path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    return f"http://127.0.0.1:{options.haproxy_port}", proxy


def _run_fleet_bench(coordinator: FleetCoordinator,
                     options: argparse.Namespace) -> int:
    """Inventory -> promote -> HAProxy -> bench-capacity -> cleanup."""
    from .fleet_ssh import SshKeyRunner
    from .health import HttpHealthProbe, wait_until

    if options.auto_discover:
        found = discover_fleet_nodes(options.discovery_port,
                                     options.discovery_timeout,
                                     options.broadcast_address)
        nodes = tuple(node.address for node in found)
        if not nodes:
            log.error("nenhum no descoberto via UDP de controle em %ss",
                      options.discovery_timeout)
            return 1
        log.info("descobertos: %s", nodes)
        runner = SshKeyRunner(_resolve_fleet_key(), user="root")
        if options.scenario == "replicated":
            endpoint, proxy = _start_replicated(runner, nodes, options)
        else:
            proxy = None
            endpoint = _configure_distributed(runner, nodes, options)
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
                haproxy_path=options.output_dir / "haproxy.cfg",
                haproxy_port=options.haproxy_port)
            proxy = subprocess.Popen(
                ["haproxy", "-f", str(result.haproxy_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            endpoint = f"http://127.0.0.1:{options.haproxy_port}"

    try:
        probe = HttpHealthProbe()
        ready = wait_until(
            lambda: probe.ready(f"{endpoint}/health"),
            timeout=options.health_timeout)
        if not ready:
            log.error("%s nao ficou pronto apos %ss",
                      endpoint, options.health_timeout)
            return 1
        bench_opts = argparse.Namespace(
            profile=options.profile, scenario=options.scenario,
            endpoint=[endpoint], output_dir=options.output_dir,
            health_timeout=options.health_timeout)
        return run_bench_capacity(bench_opts)
    finally:
        _stop_proxy(proxy)


def _stop_proxy(proxy: subprocess.Popen[bytes] | None) -> None:
    if proxy is None:
        return
    proxy.terminate()
    try:
        proxy.wait(5)
    except subprocess.TimeoutExpired:
        proxy.kill()


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
        resolved_dir = options.output_dir.resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summaries, resolved_dir / "capacity-summary.csv")
        for result in results:
            write_request_jsonl(
                result.records,
                resolved_dir / f"capacity-{result.profile}-requests.jsonl")
        print(f"resumo: {resolved_dir / 'capacity-summary.csv'}")
        return 0
    except (CapacityProfileError, OSError, ValueError) as error:
        log.error("%s", error)
        return 1


_COMMANDS = {"worker": run_worker, "nodes": run_nodes, "master": run_master,
              "exemplos": run_examples, "bench": run_bench,
              "velocidade": run_speed, "hardware-bench": run_hardware,
              "mode": run_mode,
              "become_master": run_become_master, "serve": run_serve,
              "fleet-agent": run_fleet_agent,
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
