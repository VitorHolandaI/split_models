"""Validated TCP control for vertical worker processes."""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from ..errors import ClusterError
from ..io.framing import _recv_msg, _send_msg
from ..benchmark_profiles import RUNTIME_PRECISIONS


log = logging.getLogger("split_lm")
_JOB_FIELDS = {
    "model", "layer_start", "layer_end", "total_layers", "listen_port",
    "control_to", "node_id", "connect_to", "token_back_to", "max_new_tokens",
    "precision",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")


class JobValidationError(ValueError):
    """A control request does not describe a safe vertical worker job."""


class WorkerProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


ProcessFactory = Callable[[Sequence[str]], WorkerProcess]


def _integer(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise JobValidationError(f"{field}={value!r}; esperado inteiro")
    return value


def _text(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise JobValidationError(f"{field}={value!r}; esperado texto nao vazio")
    return value


def _optional_text(payload: dict[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise JobValidationError(f"{field}={value!r}; esperado endereco ou null")
    return value


def _validate_address(value: str, field: str) -> None:
    host, separator, port_text = value.rpartition(":")
    try:
        port = int(port_text)
        socket.inet_aton(host)
    except (OSError, ValueError) as exc:
        raise JobValidationError(
            f"{field}={value!r}; esperado endereco IPv4:porta"
        ) from exc
    if not separator or not 1 <= port <= 65535:
        raise JobValidationError(f"{field}={value!r}; esperado endereco IPv4:porta")


@dataclass(frozen=True)
class VerticalJob:
    model: str
    layer_start: int
    layer_end: int
    total_layers: int
    listen_port: int
    control_to: str
    node_id: str
    connect_to: str | None
    token_back_to: str | None
    max_new_tokens: int
    precision: str

    def to_payload(self) -> dict[str, object]:
        """Serialize this validated job for a control request."""
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> VerticalJob:
        """Validate an exact worker-job schema and return its typed form."""
        unknown = set(payload) - _JOB_FIELDS
        missing = _JOB_FIELDS - set(payload)
        if unknown or missing:
            raise JobValidationError(
                f"campos unknown={sorted(unknown)}, missing={sorted(missing)}; "
                f"esperado exatamente {sorted(_JOB_FIELDS)}"
            )
        job = cls(
            _text(payload, "model"), _integer(payload, "layer_start"),
            _integer(payload, "layer_end"), _integer(payload, "total_layers"),
            _integer(payload, "listen_port"), _text(payload, "control_to"),
            _text(payload, "node_id"), _optional_text(payload, "connect_to"),
            _optional_text(payload, "token_back_to"),
            _integer(payload, "max_new_tokens"),
            _text(payload, "precision"),
        )
        job.validate()
        return job

    def validate(self) -> None:
        """Reject values outside the fixed vertical-worker contract."""
        if not 0 <= self.layer_start < self.layer_end <= self.total_layers:
            raise JobValidationError(
                f"layer_start={self.layer_start}, layer_end={self.layer_end}, "
                f"total_layers={self.total_layers}; esperado 0 <= start < end <= total"
            )
        if not 1 <= self.listen_port <= 65535 or self.max_new_tokens <= 0:
            raise JobValidationError(
                f"listen_port={self.listen_port}, max_new_tokens={self.max_new_tokens}; "
                "esperado porta valida e tokens positivos"
            )
        if not _SAFE_NAME.fullmatch(self.node_id) or not _SAFE_MODEL.fullmatch(self.model):
            raise JobValidationError(
                f"model={self.model!r}, node_id={self.node_id!r}; esperado nomes seguros"
            )
        if self.precision not in RUNTIME_PRECISIONS:
            raise JobValidationError(
                f"precision={self.precision!r}; esperado um de {RUNTIME_PRECISIONS}")
        for field in ("control_to", "connect_to", "token_back_to"):
            value = getattr(self, field)
            if value is not None:
                _validate_address(value, field)


def worker_command(job: VerticalJob, executable: str = sys.executable) -> list[str]:
    """Build a shell-free `split-lm worker` command from a validated job."""
    command = [
        executable, "-m", "split_lm", "--model", job.model,
        "--total-layers", str(job.total_layers), "--max-new-tokens",
        str(job.max_new_tokens), "worker", "--layer-start", str(job.layer_start),
        "--layer-end", str(job.layer_end), "--port", str(job.listen_port),
        "--control-to", job.control_to, "--node-id", job.node_id,
        "--precision", job.precision,
    ]
    if job.connect_to:
        command.extend(["--connect", job.connect_to])
    if job.token_back_to:
        command.extend(["--token-back-to", job.token_back_to])
    return command


def _spawn_worker(command: Sequence[str]) -> WorkerProcess:
    return subprocess.Popen(list(command))


def _stop_worker(process: WorkerProcess | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _handle_request(node_id: str, request: object,
                    process: WorkerProcess | None,
                    factory: ProcessFactory) -> tuple[dict[str, object], WorkerProcess | None]:
    if not isinstance(request, dict):
        raise JobValidationError(f"request={request!r}; esperado objeto JSON")
    request_type = request.get("type")
    if request_type == "STOP":
        _stop_worker(process)
        return {"type": "STOPPED", "node": node_id}, None
    if request_type != "START" or set(request) != {"type", "job"}:
        raise JobValidationError(f"request={request!r}; esperado START com job ou STOP")
    if process is not None and process.poll() is None:
        raise JobValidationError(f"node={node_id!r} ja executa pid={process.pid}; esperado ocioso")
    payload = request["job"]
    if not isinstance(payload, dict):
        raise JobValidationError(f"job={payload!r}; esperado objeto JSON")
    job = VerticalJob.from_payload(payload)
    started = factory(worker_command(job))
    return {"type": "STARTED", "node": node_id, "pid": started.pid}, started


def _agent_error(node_id: str, hostname: str, address: str,
                 stage: str, exc: Exception) -> ClusterError:
    return ClusterError.from_exception(
        "agent", node_id, hostname, address, stage, exc)


def _connection_response(connection: socket.socket, node_id: str,
                         process: WorkerProcess | None, factory: ProcessFactory,
                         hostname: str, address: str
                         ) -> tuple[dict[str, object], WorkerProcess | None]:
    try:
        request = _recv_msg(connection)
    except (OSError, ValueError) as exc:
        error = _agent_error(node_id, hostname, address, "agent.receive_request", exc)
        return error.to_payload(), process
    try:
        return _handle_request(node_id, request, process, factory)
    except (JobValidationError, OSError, ValueError) as exc:
        stage = ("agent.validate_request" if isinstance(exc, JobValidationError)
                 else "agent.handle_request")
        return _agent_error(node_id, hostname, address, stage, exc).to_payload(), process


def _serve_connection(connection: socket.socket, node_id: str,
                       process: WorkerProcess | None,
                       factory: ProcessFactory) -> WorkerProcess | None:
    hostname = socket.gethostname()
    address = str(connection.getsockname()[0])
    response, process = _connection_response(
        connection, node_id, process, factory, hostname, address)
    try:
        _send_msg(connection, response)
    except OSError as exc:
        error = _agent_error(node_id, hostname, address, "agent.send_response", exc)
        log.exception("%s", error)
    return process


def serve_control(node_id: str, port: int, stop: StopSignal | None = None,
                  process_factory: ProcessFactory = _spawn_worker) -> None:
    """Serve validated START/STOP requests until a stop event is set."""
    process: WorkerProcess | None = None
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", port))
        server.listen()
        server.settimeout(0.2)
        while stop is None or not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                process = _serve_connection(connection, node_id, process, process_factory)
    _stop_worker(process)


def _connect_agent(address: tuple[str, int], timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection(address, timeout=min(1.0, timeout))
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise ConnectionError(
        f"agente={address} nao conectou em {timeout}s; ultimo erro={last_error}"
    ) from last_error


def request_agent(address: tuple[str, int], message: dict[str, object],
                  timeout: float = 5.0) -> dict[str, object]:
    """Send one control request and return its response."""
    with _connect_agent(address, timeout) as connection:
        _send_msg(connection, message)
        response = _recv_msg(connection)
    if not isinstance(response, dict):
        raise ConnectionError(f"resposta={response!r} de {address}; esperado objeto JSON")
    return response
