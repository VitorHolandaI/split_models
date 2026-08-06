"""Atomic state, effective command, and the mode lock under /run/split-models.

Everything here lives in RAM (tmpfs): the appliance has no persistent disk, so
state vanishing on reboot is a feature. Writes go through a temporary file and
os.replace so a status read never catches a half-written JSON.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import TextIO

from .mode_state import NodeState

RUNTIME_DIR = Path("/run/split-models")
STATE_FILE = "state.json"
COMMAND_FILE = "effective-command.json"
LOCK_FILE = "mode.lock"
# Separate caches on purpose: the RPC worker only holds the tensors it is
# handed, while master/replica hold a full GGUF that must be wiped to give the
# RAM back when the node returns to worker.
MODEL_CACHE = "model-cache"
RPC_CACHE = "rpc-cache"


class ModeLockedError(RuntimeError):
    """Another mode transition is in progress on this node."""


class StateError(RuntimeError):
    """State that should have been recorded was not found."""


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


class ModeLock:
    """Non-blocking flock serializing mode transitions on this node.

    Usage example: `with store.lock(): controller.apply(config)`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: TextIO | None = None

    def __enter__(self) -> ModeLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ModeLockedError(
                f"lock {self._path} ocupado; esperado uma transicao de modo "
                "por vez neste no") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class StateStore:
    """Reads and writes the node's recorded mode under one runtime directory.

    Usage example: `state = StateStore(Path("/run/split-models")).read_state()`.
    """

    def __init__(self, root: Path = RUNTIME_DIR) -> None:
        self._root = root

    @property
    def state_path(self) -> Path:
        return self._root / STATE_FILE

    @property
    def command_path(self) -> Path:
        return self._root / COMMAND_FILE

    @property
    def model_cache_dir(self) -> Path:
        return self._root / MODEL_CACHE

    @property
    def rpc_cache_dir(self) -> Path:
        return self._root / RPC_CACHE

    def lock(self) -> ModeLock:
        return ModeLock(self._root / LOCK_FILE)

    def write_state(self, state: NodeState) -> None:
        _atomic_write(self.state_path, state.to_json())

    def read_state(self) -> NodeState:
        if not self.state_path.exists():
            return NodeState.boot_default()
        return NodeState.from_json(self.state_path.read_text(encoding="utf-8"))

    def write_command(self, argv: Sequence[str]) -> None:
        _atomic_write(self.command_path, json.dumps(list(argv)))

    def read_command(self) -> list[str]:
        """The effective llama-server argv recorded by the last transition.

        Usage example: `os.execvp(*[store.read_command()])` in the server unit.
        """
        if not self.command_path.exists():
            raise StateError(
                f"comando efetivo ausente em {self.command_path}; esperado "
                "arquivo gravado por `split-llama mode` antes do start da unidade")
        record = json.loads(self.command_path.read_text(encoding="utf-8"))
        if not isinstance(record, list) or not record or not all(
                isinstance(item, str) for item in record):
            raise ValueError(
                f"comando={record!r}; esperado lista nao vazia de textos")
        return list(record)

    def clear(self) -> None:
        for path in (self.state_path, self.command_path):
            path.unlink(missing_ok=True)
