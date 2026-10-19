"""SSH por chave para a orquestracao da frota, com runner injetavel.

Nenhum teste toca em ssh de verdade: o coordenador recebe um `SshRunner` que
os testes substituem por um fake nomeado. A implementacao real usa somente a
stdlib (subprocess) e a chave privada dedicada que fica fora da imagem; login
por senha nunca e tentado (`BatchMode=yes`).
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

# A chave privada vive fora da imagem PXE e nao e versionada; a publica esta
# no authorized_keys (deploy/archiso-llamacpp-rpc/.../sshd_config.d).
FLEET_KEY = "/root/.ssh/fleet_ed25519"


class SshError(RuntimeError):
    """Uma execucao SSH falhou; a mensagem nomeia host, comando e saida."""


@dataclass(frozen=True)
class SshResult:
    """Saida de um comando SSH bem-sucedido (returncode 0 garantido)."""

    stdout: str
    stderr: str


class SshRunner(Protocol):
    """The two SSH facts a coordinator needs: run a remote argv by host."""

    def run(self, host: str, argv: Sequence[str]) -> SshResult: ...


class SshKeyRunner:
    """`ssh -i <chave>` com opcoes de batch, subprocess.run injetavel.

    Falha fechado: returncode diferente de 0 ou OSError viram SshError com o
    host, o comando e a saida que o operador precisa para agir.

    Usage example: `result = SshKeyRunner("/keys/fleet").run("node-a", ["hostname"])`.
    """

    def __init__(self, key: str, user: str = "root", connect_timeout: int = 10,
                 run: Callable[..., subprocess.CompletedProcess[str]]
                 = subprocess.run) -> None:
        self._key = key
        self._user = user
        self._connect_timeout = connect_timeout
        self._run = run

    def run(self, host: str, argv: Sequence[str]) -> SshResult:
        remote_command = shlex.join(argv)
        command = ["ssh", "-i", self._key, "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=accept-new",
                   "-o", f"ConnectTimeout={self._connect_timeout}",
                   f"{self._user}@{host}", remote_command]
        try:
            result = self._run(command, check=False, capture_output=True,
                               text=True)
        except OSError as exc:
            raise SshError(
                f"ssh {host} nao executou: {exc}; esperado chave "
                f"{self._key} presente e host alcancavel") from exc
        if result.returncode != 0:
            # stderr curto o suficiente para a mensagem nao virar um dump.
            tail = result.stderr.strip()[-400:]
            raise SshError(
                f"ssh {host} {' '.join(argv)} saiu {result.returncode}; "
                f"esperado 0. stderr: {tail!r}")
        return SshResult(result.stdout, result.stderr)
