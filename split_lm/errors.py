"""Structured cluster errors shared by agents, workers, heads, and masters."""

from __future__ import annotations

import errno as errno_module
from dataclasses import dataclass


_PAYLOAD_FIELDS = {"type", "version", "origin", "machine", "stage", "cause"}
_MACHINE_FIELDS = {"node_id", "hostname", "address"}
_CAUSE_FIELDS = {"code", "type", "message", "errno"}


def _error_code(exc: Exception) -> str:
    if isinstance(exc, OSError) and exc.errno == errno_module.EADDRINUSE:
        return "address_in_use"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection_failed"
    if isinstance(exc, ImportError):
        return "dependency_missing"
    if isinstance(exc, ValueError):
        return "invalid_value"
    if isinstance(exc, OSError):
        return "operating_system"
    return "internal_error"


def _mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(
            f"{label}={value!r}; esperado objeto com campos {sorted(fields)}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}={value!r}; esperado texto nao vazio")
    return value


@dataclass(frozen=True)
class ClusterError(Exception):
    """Carry one failure with its component, machine, stage, and root cause."""

    origin: str
    node_id: str
    hostname: str
    address: str
    stage: str
    code: str
    cause_type: str
    message: str
    errno: int | None = None

    @classmethod
    def from_exception(cls, origin: str, node_id: str, hostname: str,
                       address: str, stage: str, exc: Exception) -> ClusterError:
        """Classify an exception at the boundary where its context is known."""
        return cls(
            origin, node_id, hostname, address, stage, _error_code(exc),
            type(exc).__name__, str(exc), getattr(exc, "errno", None))

    def to_payload(self) -> dict[str, object]:
        """Serialize the error for a control-channel ERROR message."""
        machine = {"node_id": self.node_id, "hostname": self.hostname,
                   "address": self.address}
        cause = {"code": self.code, "type": self.cause_type,
                 "message": self.message, "errno": self.errno}
        return {"type": "ERROR", "version": 1, "origin": self.origin,
                "machine": machine, "stage": self.stage, "cause": cause}

    @classmethod
    def from_payload(cls, payload: object) -> ClusterError:
        """Validate and reconstruct one remote ERROR payload."""
        record = _mapping(payload, _PAYLOAD_FIELDS, "error")
        if record["type"] != "ERROR" or record["version"] != 1:
            raise ValueError(
                f"type={record['type']!r} version={record['version']!r}; "
                "esperado ERROR version=1")
        machine = _mapping(record["machine"], _MACHINE_FIELDS, "machine")
        cause = _mapping(record["cause"], _CAUSE_FIELDS, "cause")
        error_number = cause["errno"]
        if error_number is not None and not isinstance(error_number, int):
            raise ValueError(f"cause.errno={error_number!r}; esperado inteiro ou null")
        return cls(
            _text(record["origin"], "origin"), _text(machine["node_id"], "node_id"),
            _text(machine["hostname"], "hostname"), _text(machine["address"], "address"),
            _text(record["stage"], "stage"), _text(cause["code"], "cause.code"),
            _text(cause["type"], "cause.type"), _text(cause["message"], "cause.message"),
            error_number)

    def __str__(self) -> str:
        machine = f"{self.node_id}/{self.hostname}@{self.address}"
        error_number = f" errno={self.errno}" if self.errno is not None else ""
        return (f"ERROR origin={self.origin} machine={machine} stage={self.stage} "
                f"cause={self.code}/{self.cause_type}{error_number}: {self.message}")
