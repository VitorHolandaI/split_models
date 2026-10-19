"""Build the llama-server invocation for the distributed-master and replica modes."""

from __future__ import annotations

from .mode_state import SERVER_MODES, Mode, ModeConfig, ModeConfigError

LLAMA_SERVER = "llama-server"


def llama_server_command(config: ModeConfig) -> list[str]:
    """Build the supervised server command for a master or replica config.

    `--host 0.0.0.0` because an external HAProxy must reach the replicas; the
    llama-server default of 127.0.0.1 would hide the node from the LAN.
    `--cont-batching` is llama.cpp's default but goes explicitly: the benchmark
    confirms real batching via `llamacpp:n_busy_slots_per_decode`, and an
    implicit default changing upstream would silently break that reading.
    `--no-mmproj` keeps the visual projector out: the benchmark is text-only.

    Usage example: `argv = llama_server_command(ModeConfig(Mode.REPLICA, repo))`.
    """
    if config.mode not in SERVER_MODES:
        raise ModeConfigError(
            f"modo={config.mode.value}; esperado um de "
            f"{[mode.value for mode in SERVER_MODES]} para montar llama-server")
    if config.model is None:
        raise ModeConfigError(
            f"modo={config.mode.value} sem model; esperado repositorio GGUF")
    command = [LLAMA_SERVER, "-hf", config.model,
               "--host", "0.0.0.0", "--port", str(config.port),
               "--parallel", str(config.parallel),
               "--cont-batching", "--metrics", "--no-mmproj"]
    if config.mode is Mode.DISTRIBUTED_MASTER:
        # Only the master spans the fleet; a replica is a full local copy and
        # must not reference workers that are not there.
        command += ["--rpc", ",".join(config.rpc_endpoints),
                    "--split-mode", "layer"]
    return command
