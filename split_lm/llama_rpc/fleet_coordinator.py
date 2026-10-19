"""Coordenador da frota por SSH: inventario, promocao, replicas e reset.

Cada cenario coleta o inventario exato dos nos usados, grava um JSON por run e
grava um registro que referencia esse mesmo arquivo, para que qualquer run
futura possa ser reproduzida a partir do inventario que a decidiu. A promocao
segue o controlador de modos do no (Phase 2): o coordenador escreve o
ModeConfig do master remoto e espera `GET /health` no IP de LAN do inventario.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .discovery import RPC_PORT
from .examples import KNOWN_MODELS
from .fleet_inventory import (
    FleetInventory, InventoryError, NodeInventory, inventory_sections_command,
    parse_node_inventory)
from .fleet_ssh import SshRunner
from .haproxy import haproxy_config
from .health import HttpHealthProbe, wait_until
from .mode_state import MODEL_PROFILES, Mode, ModeConfig

# O GGUF inteiro mora no tmpfs do master: a fraqueza do orcamento e o arquivo
# no master, nao o total de workers. Overhead identico ao do examples.py.
_MASTER_OVERHEAD_GB = 2.0


class FleetError(RuntimeError):
    """Um cenario de frota foi recusado; a mensagem nomeia o valor e o esperado."""


@dataclass(frozen=True)
class FleetRunRecord:
    """Registro de um cenario concluido, referenciando o inventario exato."""

    scenario: str
    timestamp: str
    inventory_file: str
    nodes: tuple[str, ...]
    params: tuple[tuple[str, str], ...] = ()

    def to_json(self) -> str:
        return json.dumps({"scenario": self.scenario, "timestamp": self.timestamp,
                           "inventory_file": self.inventory_file,
                           "nodes": list(self.nodes),
                           "params": dict(self.params)}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> FleetRunRecord:
        record = json.loads(text)
        if not isinstance(record, dict) or set(record) != {
                "scenario", "timestamp", "inventory_file", "nodes", "params"}:
            raise ValueError(
                f"registro={record!r}; esperado objeto com scenario, timestamp, "
                "inventory_file, nodes e params")
        params = record["params"]
        if not isinstance(params, dict):
            raise ValueError(f"params={params!r}; esperado objeto de textos")
        return cls(
            _text(record["scenario"], "scenario"),
            _text(record["timestamp"], "timestamp"),
            _text(record["inventory_file"], "inventory_file"),
            tuple(_text(item, "nodes") for item in record["nodes"]),
            tuple((str(key), str(value)) for key, value in params.items()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


@dataclass(frozen=True)
class PromotionResult:
    """The outcome of a distributed promotion, for the operator's report."""

    inventory_path: Path
    run_path: Path
    endpoints: tuple[str, ...]
    contributions: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ReplicaResult:
    """The outcome of a replicated scenario, for the operator's report."""

    inventory_path: Path
    run_path: Path
    haproxy_path: Path


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}={value!r}; esperado texto")
    return value


def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_safe(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace("+", "-")


class FleetCoordinator:
    """Drives one fleet scenario over SSH, recording the exact inventory.

    Every dependency is constructor-injected (ssh runner, health probe, clock,
    clock-aware sleeper, run directory, timestamp) so tests run full scenarios
    against named fakes with no ssh, HTTP, or sleeping involved.

    Usage example:
        `coordinator.promote_master("node-a", ("node-b", "node-c"), "large", 8)`.
    """

    def __init__(self, ssh: SshRunner, *,
                 health: HttpHealthProbe | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 health_timeout: float = 900.0,
                 run_dir: Path = Path("fleet-runs"),
                 rpc_port: int = RPC_PORT,
                 now: Callable[[], str] = _default_now) -> None:
        self._ssh = ssh
        self._health = health or HttpHealthProbe()
        self._sleep = sleep
        self._clock = clock
        self._health_timeout = health_timeout
        self._run_dir = run_dir
        self._rpc_port = rpc_port
        self._now = now

    def collect_inventory(self, hosts: Sequence[str]
                          ) -> tuple[Path, tuple[NodeInventory, ...]]:
        """Collect and save the exact inventory of `hosts`, one SSH run each.

        Usage example: `path, nodes = coordinator.collect_inventory(["h1"])`.
        """
        hosts = self._assert_unique(hosts, "hosts")
        timestamp = self._now()
        nodes, inventory_path = self._collect(hosts, timestamp)
        return inventory_path, nodes

    def promote_master(self, master_host: str,
                       worker_hosts: Sequence[str], model_profile: str, *,
                       parallel: int = 4, port: int = 8080) -> PromotionResult:
        """Promote one node to distributed-master over the other nodes' RPC.

        The coordinator builds the master's ModeConfig, applies it remotely
        (`split-llama mode distributed-master`), and waits for `GET /health`
        on the master's LAN IP before reporting which endpoint each worker
        contributes.

        Usage example: `result = coordinator.promote_master("m", ("w1", "w2"), "large")`.
        """
        worker_hosts = self._assert_unique(worker_hosts, "workers")
        if not worker_hosts:
            raise FleetError(
                "workers vazio; esperado ao menos um no worker")
        if master_host in worker_hosts:
            raise FleetError(
                f"master_host={master_host!r} tambem listado em "
                f"workers={worker_hosts!r}; esperado nos distintos")
        if model_profile not in MODEL_PROFILES:
            raise FleetError(
                f"model_profile={model_profile!r}; esperado um de "
                f"{sorted(MODEL_PROFILES)}")
        hosts = (master_host, *worker_hosts)
        timestamp = self._now()
        nodes, inventory_path = self._collect(hosts, timestamp)
        master, workers = nodes[0], nodes[1:]
        self._assert_workers_are_workers(workers)
        self._assert_master_budget(master, model_profile)
        endpoints, contributions = self._worker_endpoints(workers)
        ModeConfig(Mode.DISTRIBUTED_MASTER, MODEL_PROFILES[model_profile],
                   parallel, endpoints, port)
        self._ssh.run(master_host, self._mode_argv(
            "distributed-master", model_profile, parallel, port, endpoints))
        self._wait_health(master, port)
        run_path = self._record_run("distributed", hosts, inventory_path, (
            ("master", master_host), ("workers", ",".join(worker_hosts)),
            ("model", model_profile), ("parallel", str(parallel)),
            ("port", str(port))), timestamp)
        return PromotionResult(inventory_path, run_path, endpoints, contributions)

    def configure_replicas(self, replica_hosts: Sequence[str],
                           model_profile: str, *,
                           parallel: int = 4, port: int = 8080,
                           haproxy_path: Path = Path("haproxy.cfg"),
                           haproxy_port: int | None = None
                           ) -> ReplicaResult:
        """Put every replica in local-server mode and write the HAProxy file.

        Each replica runs `split-llama mode replica` (no `--rpc`) and must
        answer `GET /health` before the HAProxy config points at it.

        Usage example: `result = coordinator.configure_replicas(["r1", "r2"], "small")`.
        """
        replica_hosts = self._assert_unique(replica_hosts, "replicas")
        if not replica_hosts:
            raise FleetError(
                "replicas vazio; esperado ao menos um no")
        if model_profile not in MODEL_PROFILES:
            raise FleetError(
                f"model_profile={model_profile!r}; esperado um de "
                f"{sorted(MODEL_PROFILES)}")
        timestamp = self._now()
        nodes, inventory_path = self._collect(replica_hosts, timestamp)
        ModeConfig(Mode.REPLICA, MODEL_PROFILES[model_profile],
                   parallel, (), port)
        argv = self._mode_argv("replica", model_profile, parallel, port)
        for node in nodes:
            self._ssh.run(node.host, argv)
        for node in nodes:
            self._wait_health(node, port)
        self._write_haproxy(haproxy_path, nodes, port,
                            haproxy_port if haproxy_port is not None else port)
        run_path = self._record_run("replicated", replica_hosts, inventory_path, (
            ("model", model_profile), ("parallel", str(parallel)),
            ("port", str(port)), ("haproxy_file", str(haproxy_path))), timestamp)
        return ReplicaResult(inventory_path, run_path, haproxy_path)

    def reset(self, hosts: Sequence[str]) -> Path:
        """Return every node to worker mode and record the exact inventory.

        `split-llama mode worker` stops the server unit, wipes the full GGUF
        from tmpfs, clears the recorded command, and restarts the RPC worker;
        that is the sanctioned cleanup path of the node controller.

        Usage example: `path = coordinator.reset(["h1", "h2"])`.
        """
        hosts = self._assert_unique(hosts, "hosts")
        if not hosts:
            raise FleetError(
                "hosts vazio; esperado ao menos um no")
        timestamp = self._now()
        nodes, inventory_path = self._collect(hosts, timestamp)
        for node in nodes:
            self._ssh.run(node.host, ["split-llama", "mode", "worker"])
        return self._record_run("reset", hosts, inventory_path, (), timestamp)

    def _collect(self, hosts: tuple[str, ...], timestamp: str
                 ) -> tuple[tuple[NodeInventory, ...], Path]:
        nodes = tuple(self._inventory_one(host) for host in hosts)
        inventory_path = self._run_dir / f"inventory-{_file_safe(timestamp)}.json"
        FleetInventory(timestamp, nodes).save(inventory_path)
        return nodes, inventory_path

    def _inventory_one(self, host: str) -> NodeInventory:
        result = self._ssh.run(host, inventory_sections_command())
        return parse_node_inventory(result.stdout, host)

    def _record_run(self, scenario: str, hosts: tuple[str, ...],
                    inventory_path: Path,
                    params: tuple[tuple[str, str], ...], timestamp: str
                    ) -> Path:
        run_path = self._run_dir / f"run-{_file_safe(timestamp)}.json"
        FleetRunRecord(scenario, timestamp, str(inventory_path), hosts,
                       params).save(run_path)
        return run_path

    def _assert_unique(self, hosts: Sequence[str], label: str) -> tuple[str, ...]:
        checked = tuple(hosts)
        duplicates = {host for host in checked if checked.count(host) > 1}
        if duplicates:
            raise FleetError(
                f"{label}={checked!r} com repetido {sorted(duplicates)}; "
                "esperado lista sem duplicatas")
        return checked

    def _assert_workers_are_workers(self, workers: tuple[NodeInventory, ...]
                                    ) -> None:
        for node in workers:
            if node.mode != Mode.WORKER.value:
                raise FleetError(
                    f"worker {node.node_id} em modo {node.mode!r}; esperado "
                    "'worker' antes de promover o master")

    def _assert_master_budget(self, master: NodeInventory,
                              model_profile: str) -> None:
        # Somente modelos com tamanho medido entram no preflight; perfil novo
        # sem medida nao pode chutar gigabyte.
        repo = MODEL_PROFILES[model_profile].split(":", 1)[0]
        model = next((item for item in KNOWN_MODELS if item.repo == repo), None)
        if model is None:
            return
        ram_gb = master.mem_total_kb / (1024 * 1024)
        needed_gb = model.gigabytes + _MASTER_OVERHEAD_GB
        if not model.fits_master(ram_gb):
            raise FleetError(
                f"master {master.node_id} com RAM={ram_gb:.1f} GB nao comporta "
                f"{model.repo} de {model.gigabytes:.2f} GB; esperado RAM >= "
                f"{needed_gb:.1f} GB (GGUF em tmpfs mais overhead)")

    def _worker_endpoints(self, workers: tuple[NodeInventory, ...]
                          ) -> tuple[tuple[str, ...],
                                     tuple[tuple[str, str, str], ...]]:
        endpoints: list[str] = []
        contributions: list[tuple[str, str, str]] = []
        for node in workers:
            if node.ip == "unknown":
                raise FleetError(
                    f"worker {node.node_id} sem IP no inventario; esperado "
                    "endereco alcancavel pelo master")
            endpoint = f"{node.ip}:{self._rpc_port}"
            endpoints.append(endpoint)
            contributions.append((endpoint, node.node_id, node.hostname))
        return tuple(endpoints), tuple(contributions)

    def _mode_argv(self, target: str, model_profile: str, parallel: int,
                   port: int, endpoints: tuple[str, ...] = ()) -> list[str]:
        argv = ["split-llama", "mode", target, "--model", model_profile,
                "--parallel", str(parallel), "--port", str(port)]
        if endpoints:
            argv += ["--rpc", ",".join(endpoints)]
        return argv

    def _wait_health(self, node: NodeInventory, port: int) -> None:
        if node.ip == "unknown":
            raise FleetError(
                f"no {node.node_id} sem IP no inventario; esperado endereco "
                "para o health check")
        url = f"http://{node.ip}:{port}/health"
        ready = wait_until(lambda: self._health.ready(url),
                           self._health_timeout, sleep=self._sleep,
                           clock=self._clock)
        if not ready:
            raise FleetError(
                f"GET {url} sem 200 apos {self._health_timeout}s; esperado "
                "readiness do llama-server")

    def _write_haproxy(self, path: Path, nodes: tuple[NodeInventory, ...],
                       backend_port: int, listen_port: int) -> None:
        servers = tuple((node.node_id, f"{node.ip}:{backend_port}")
                        for node in nodes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(haproxy_config(servers, listen_port), encoding="utf-8")
