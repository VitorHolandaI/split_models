"""The supervised systemd units behind worker and server modes.

tmux no longer owns process lifetimes: the worker and the inference server
are proper units with Restart=on-failure, and the controller drives them
through this thin interface so tests substitute a named fake.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Protocol

WORKER_UNIT = "split-llama-worker.service"
SERVER_UNIT = "split-llama-server.service"


class UnitCommandError(RuntimeError):
    """A systemctl action failed; the message carries the unit and output."""


class UnitManager(Protocol):
    """The two actions a mode transition needs from the supervisor."""

    def start(self, unit: str) -> None: ...
    def stop(self, unit: str) -> None: ...


class SystemdUnitManager:
    """systemctl start/stop, with the runner injectable for tests.

    Usage example: `SystemdUnitManager().stop(WORKER_UNIT)`.
    """

    def __init__(self, run: Callable[..., subprocess.CompletedProcess[str]]
                 = subprocess.run) -> None:
        self._run = run

    def start(self, unit: str) -> None:
        self._systemctl("start", unit)

    def stop(self, unit: str) -> None:
        self._systemctl("stop", unit)

    def _systemctl(self, action: str, unit: str) -> None:
        result = self._run(["systemctl", action, unit], check=False,
                           capture_output=True, text=True)
        if result.returncode != 0:
            raise UnitCommandError(
                f"systemctl {action} {unit} saiu {result.returncode}; "
                f"esperado 0. stderr: {result.stderr.strip()!r}")
