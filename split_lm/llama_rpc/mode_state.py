"""Typed node modes, their configuration, and their serialized state.

The mode of a node decides which supervised unit runs on it: the RPC worker,
a llama-server spanning the fleet, a full local replica, or nothing. Keeping
the four modes as one typed enum is what lets the CLI, the controller and the
status panel agree on the same names instead of re-parsing free text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class ModeConfigError(ValueError):
    """A mode was asked for with a configuration it cannot run."""


class Mode(Enum):
    """The four roles a node can assume without a reboot."""

    WORKER = "worker"
    DISTRIBUTED_MASTER = "distributed-master"
    REPLICA = "replica"
    IDLE = "idle"

    @classmethod
    def from_text(cls, text: str) -> Mode:
        """Parse a CLI mode name, naming the valid set on refusal.

        Usage example: `mode = Mode.from_text("distributed-master")`.
        """
        try:
            return cls(text)
        except ValueError as exc:
            expected = [mode.value for mode in cls]
            raise ModeConfigError(
                f"modo={text!r}; esperado um de {expected}") from exc


# Official profiles from plan.md: only these repository:quantization pairs are
# valid benchmark models, so the CLI never takes a free-form GGUF name.
MODEL_PROFILES = {
    "small": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "large": "unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
}

SERVER_MODES = (Mode.DISTRIBUTED_MASTER, Mode.REPLICA)

_STATE_FIELDS = ("mode", "model", "parallel", "rpc_endpoints", "port")


@dataclass(frozen=True)
class ModeConfig:
    """What the operator asked this node to become.

    `model` is the resolved HuggingFace repository, not the profile name; the
    CLI maps small/large through MODEL_PROFILES before building this.
    """

    mode: Mode
    model: str | None = None
    parallel: int = 4
    rpc_endpoints: tuple[str, ...] = ()
    port: int = 8080

    def __post_init__(self) -> None:
        if self.parallel < 1:
            raise ModeConfigError(
                f"parallel={self.parallel}; esperado inteiro >= 1")
        if not 1 <= self.port <= 65535:
            raise ModeConfigError(
                f"port={self.port}; esperado inteiro entre 1 e 65535")
        if self.mode in SERVER_MODES:
            self._validate_server()
            return
        self._validate_not_server()

    def _validate_server(self) -> None:
        if not self.model:
            raise ModeConfigError(
                f"modo={self.mode.value} sem model; esperado um perfil de "
                f"{sorted(MODEL_PROFILES)}")
        # A master without --rpc would download the whole GGUF and serve
        # alone, looking distributed while using no worker at all.
        if self.mode is Mode.DISTRIBUTED_MASTER and not self.rpc_endpoints:
            raise ModeConfigError(
                "modo=distributed-master sem rpc_endpoints; esperado "
                "--rpc host:porta[,host:porta...] explicito")

    def _validate_not_server(self) -> None:
        if self.model is not None or self.rpc_endpoints:
            raise ModeConfigError(
                f"modo={self.mode.value} com model={self.model!r} "
                f"rpc_endpoints={self.rpc_endpoints!r}; esperado nenhum dos dois")


@dataclass(frozen=True)
class NodeState:
    """The recorded role of this node, persisted under /run/split-models."""

    mode: Mode
    model: str | None
    parallel: int
    rpc_endpoints: tuple[str, ...]
    port: int

    @classmethod
    def from_config(cls, config: ModeConfig) -> NodeState:
        """Record the config that just became effective.

        Usage example: `state = NodeState.from_config(config)`.
        """
        return cls(config.mode, config.model, config.parallel,
                   config.rpc_endpoints, config.port)

    @classmethod
    def boot_default(cls) -> NodeState:
        """The role of a fresh boot: a worker that has not downloaded anything.

        The state file may not exist yet when status runs for the first time,
        and the panel still needs an answer.
        """
        return cls(Mode.WORKER, None, 4, (), 8080)

    def to_json(self) -> str:
        """Serialize for the atomic state file.

        Usage example: `store.write_text(state.to_json())`.
        """
        return json.dumps({
            "mode": self.mode.value, "model": self.model,
            "parallel": self.parallel,
            "rpc_endpoints": list(self.rpc_endpoints), "port": self.port},
            indent=2)

    @classmethod
    def from_json(cls, text: str) -> NodeState:
        """Rebuild a state, refusing anything but the exact recorded shape.

        Usage example: `state = NodeState.from_json(path.read_text())`.
        """
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"estado={text!r}; esperado JSON com campos {_STATE_FIELDS}") from exc
        if not isinstance(record, dict) or set(record) != set(_STATE_FIELDS):
            raise ValueError(
                f"estado={record!r}; esperado objeto com campos {_STATE_FIELDS}")
        return cls(_parse_mode(record["mode"]), _parse_model(record["model"]),
                   _parse_int(record["parallel"], "parallel"),
                   _parse_endpoints(record["rpc_endpoints"]),
                   _parse_int(record["port"], "port"))


def _parse_mode(value: object) -> Mode:
    if not isinstance(value, str):
        raise ValueError(f"mode={value!r}; esperado texto")
    return Mode.from_text(value)


def _parse_model(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"model={value!r}; esperado texto ou null")


def _parse_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label}={value!r}; esperado inteiro")
    return value


def _parse_endpoints(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value):
        raise ValueError(
            f"rpc_endpoints={value!r}; esperado lista de textos host:porta")
    return tuple(value)
