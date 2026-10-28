"""Locked mode transitions with readiness, cleanup, and rollback to worker.

The sequence stops the worker unit and its RPC-only announcement, waits for the
RPC port to close, restarts the server with its new command, waits for
`GET /health`, and only then records the mode. The separate fleet discovery
service remains active throughout. Any failure cleans state and restores
`worker`, so a node never ends up role-less or with a dead server holding a
full model in RAM.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .discovery import RPC_PORT
from .health import HttpHealthProbe, TcpPortProbe, wait_until
from .mode_state import Mode, ModeConfig, NodeState
from .mode_store import StateStore
from .model_cache import record_cached_model, select_cached_model
from .server import llama_server_command
from .units import SERVER_UNIT, WORKER_UNIT, UnitManager

log = logging.getLogger("split_lm")
_MODEL_CACHE_HELPER = Path("/usr/local/bin/split-llama-model-cache")


class ModeTransitionError(RuntimeError):
    """A transition failed after rollback; the message names mode and stage."""


class ModeController:
    """Drives this node between worker, distributed-master, replica, and idle.

    Every dependency is constructor-injected (store, units, probes, clock) so
    tests run the full transition against named fakes with no systemd, HTTP,
    or sleeping involved.

    Usage example: `state = ModeController(store, units).apply(config)`.
    """

    def __init__(self, store: StateStore, units: UnitManager, *,
                 health: HttpHealthProbe | None = None,
                 port_probe: TcpPortProbe | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 health_timeout: float = 900.0,
                 port_timeout: float = 30.0) -> None:
        self._store = store
        self._units = units
        self._health = health or HttpHealthProbe()
        self._port_probe = port_probe or TcpPortProbe()
        self._sleep = sleep
        self._clock = clock
        # Downloads of the 4B profile on a slow LAN fit in the default window;
        # /health stays 503 for the whole download+load, by design.
        self._health_timeout = health_timeout
        self._port_timeout = port_timeout

    def apply(self, config: ModeConfig, *,
              keep_model_cache: bool = False) -> NodeState:
        """Move the node to `config.mode`, rolling back to worker on failure.

        Usage example: `controller.apply(ModeConfig(Mode.IDLE))`.
        """
        with self._store.lock():
            if config.mode is Mode.WORKER:
                return self._to_worker(config, keep_model_cache)
            if config.mode is Mode.IDLE:
                return self._to_idle(config, keep_model_cache)
            return self._to_server(config, keep_model_cache)

    def status(self) -> NodeState:
        """The recorded mode, or the boot default before the first transition."""
        return self._store.read_state()

    def _to_worker(self, config: ModeConfig, keep_cache: bool) -> NodeState:
        self._units.stop(SERVER_UNIT)
        if not keep_cache:
            self._clear_model_cache()
        self._units.start(WORKER_UNIT)
        return self._record(config)

    def _to_idle(self, config: ModeConfig, keep_cache: bool) -> NodeState:
        self._units.stop(SERVER_UNIT)
        self._units.stop(WORKER_UNIT)
        if not keep_cache:
            self._clear_model_cache()
        return self._record(config)

    def _to_server(self, config: ModeConfig, keep_cache: bool) -> NodeState:
        stage = "worker.stop"
        try:
            self._units.stop(WORKER_UNIT)
            stage = "worker.port-close"
            self._wait_worker_port_closed()
            # `systemctl start` nao reinicia um servidor ja ativo. Parar aqui
            # garante que replica -> distributed-master aplique o novo argv.
            stage = "server.stop"
            self._units.stop(SERVER_UNIT)
            stage = "server.prepare"
            self._prepare_model_cache(config, keep_cache)
            self._store.write_command(llama_server_command(config))
            stage = "server.start"
            self._units.start(SERVER_UNIT)
            stage = "server.health"
            self._wait_health(config)
            stage = "server.cache-record"
            if config.model:
                record_cached_model(self._store.model_cache_dir, config.model)
        except Exception as exc:
            raise self._rollback(config, stage, exc) from exc
        return self._record(config)

    def _record(self, config: ModeConfig) -> NodeState:
        state = NodeState.from_config(config)
        if state.mode in (Mode.WORKER, Mode.IDLE):
            self._store.clear()
        self._store.write_state(state)
        log.info("modo registrado: %s", state.mode.value)
        return state

    def _wait_worker_port_closed(self) -> None:
        closed = wait_until(lambda: not self._port_probe.open(RPC_PORT),
                            self._port_timeout, sleep=self._sleep,
                            clock=self._clock)
        if not closed:
            raise TimeoutError(
                f"porta RPC {RPC_PORT} aberta apos {self._port_timeout}s; "
                "esperado ggml-rpc-server parado antes de servir")

    def _wait_health(self, config: ModeConfig) -> None:
        url = f"http://127.0.0.1:{config.port}/health"
        ready = wait_until(lambda: self._health.ready(url),
                           self._health_timeout, sleep=self._sleep,
                           clock=self._clock)
        if not ready:
            raise TimeoutError(
                f"GET {url} sem 200 apos {self._health_timeout}s; esperado "
                "readiness do llama-server (health)")

    def _rollback(self, config: ModeConfig, stage: str, exc: Exception
                  ) -> ModeTransitionError:
        log.warning("transicao para %s falhou em %s; restaurando worker",
                    config.mode.value, stage)
        self._units.stop(SERVER_UNIT)
        self._store.clear()
        # A partial download keeps holding tmpfs RAM; rollback always wipes it,
        # even under --keep-model-cache, because a half file is never reusable.
        self._clear_model_cache()
        self._units.start(WORKER_UNIT)
        self._store.write_state(NodeState.boot_default())
        return ModeTransitionError(
            f"transicao para {config.mode.value} falhou em {stage}: {exc}; "
            "no restaurado para worker")

    def _clear_model_cache(self) -> None:
        cache = self._store.model_cache_dir
        if not cache.exists():
            return
        if _MODEL_CACHE_HELPER.exists():
            subprocess.run([str(_MODEL_CACHE_HELPER), "--unmount"], check=True)
        for child in cache.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _prepare_model_cache(self, config: ModeConfig, keep_cache: bool) -> None:
        if keep_cache and config.model:
            select_cached_model(self._store.model_cache_dir, config.model)
            return
        self._clear_model_cache()
