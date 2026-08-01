"""Run ggml-rpc-server and keep discovery honest about whether it is up."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading

from ..errors import ClusterError
from .discovery import RPC_PORT, RpcWorker, serve_discovery

log = logging.getLogger("split_lm")

RPC_SERVER = "ggml-rpc-server"
# The model cache lives in RAM with the rest of the appliance state, so nothing
# survives a reboot and no disk is ever written.
RPC_CACHE_DIR = "/run/split-models/llama-cache"


def lan_address(remote_address: str = "8.8.8.8") -> str:
    """Return the local IPv4 the kernel would use to leave this machine.

    Connecting a UDP socket sends nothing; it only asks the routing table which
    interface would be chosen, which is the address the other nodes can reach.

    Usage example: `address = lan_address()`.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((remote_address, 53))
        return str(probe.getsockname()[0])


def rpc_server_command(bind_address: str, rpc_port: int = RPC_PORT) -> list[str]:
    """Build the ggml-rpc-server command line.

    `--cache` is a boolean switch, not an option taking a path: the location
    comes from the LLAMA_CACHE environment variable, read by
    fs_get_cache_directory in tools/rpc/rpc-server.cpp. Passing a directory
    after it makes the server refuse to start with "unknown argument".

    `--device` is deliberately absent: the server enumerates whatever backends
    it finds and falls back to CPU on a host without Vulkan, which is what lets
    one image serve both kinds of machine.

    That holds because the image carries one backend per card. A machine built
    with CUDA as well lists the same GPU twice, once per backend, and serving
    both makes the master add the two entries and believe it has double the
    VRAM. Those hosts need `--device`, which is why the bare-metal build says so
    rather than this appliance path.

    Usage example: `command = rpc_server_command("192.168.0.11")`.
    """
    return [RPC_SERVER, "--host", bind_address, "--port", str(rpc_port),
            "--cache"]


def rpc_server_environment(cache_dir: str = RPC_CACHE_DIR) -> dict[str, str]:
    """Environment placing the RPC cache in RAM with the rest of the state.

    Usage example: `environment = rpc_server_environment()`.
    """
    return {**os.environ, "LLAMA_CACHE": cache_dir}


def _wait_for_port(address: str, port: int, timeout: float) -> bool:
    deadline = threading.Event()
    for _ in range(int(timeout * 10)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((address, port)) == 0:
                return True
        if deadline.wait(0.1):
            break
    return False


def _rpc_failure(worker: RpcWorker, stage: str, exc: Exception) -> ClusterError:
    return ClusterError.from_exception(
        "llama-rpc-worker", worker.node_id, worker.hostname, worker.address,
        stage, exc)


def start_rpc_server(worker: RpcWorker, cache_dir: str = RPC_CACHE_DIR,
                     ready_timeout: float = 30.0) -> subprocess.Popen[str]:
    """Launch ggml-rpc-server and return once its port accepts connections.

    Usage example: `server = start_rpc_server(local_worker("node-a"))`.
    """
    command = rpc_server_command(worker.address, worker.rpc_port)
    log.info("iniciando %s com LLAMA_CACHE=%s", " ".join(command), cache_dir)
    try:
        server = subprocess.Popen(command, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  env=rpc_server_environment(cache_dir))
    except OSError as exc:
        raise _rpc_failure(worker, "rpc_server.spawn", exc) from exc
    _assert_server_listening(worker, server, ready_timeout)
    return server


# Discovery must not advertise a node whose RPC port is not accepting yet: a
# master that believed the reply would fail at the handshake instead, naming a
# socket rather than the node that never came up.
def _assert_server_listening(worker: RpcWorker, server: subprocess.Popen[str],
                             ready_timeout: float) -> None:
    if _wait_for_port(worker.address, worker.rpc_port, ready_timeout):
        return
    server.terminate()
    output = server.stdout.read() if server.stdout else ""
    raise _rpc_failure(worker, "rpc_server.listen", TimeoutError(
        f"{RPC_SERVER} nao aceitou conexao em {worker.endpoint} apos "
        f"{ready_timeout}s; saida: {output.strip()!r}"))


def serve_until_server_exits(worker: RpcWorker, server: subprocess.Popen[str],
                             discovery_port: int) -> int:
    """Answer discovery while the RPC server lives, and stop when it dies.

    A worker that kept replying after its server exited would keep being handed
    layers it cannot serve, so the two lifetimes are tied together here.

    Usage example: `code = serve_until_server_exits(worker, server, 29520)`.
    """
    stop, ready = threading.Event(), threading.Event()
    responder = threading.Thread(
        target=serve_discovery, args=(worker, discovery_port, stop, ready),
        daemon=True)
    responder.start()
    ready.wait(5)
    try:
        return server.wait()
    finally:
        stop.set()
        responder.join(2)


def assert_clean_exit(worker: RpcWorker, server: subprocess.Popen[str],
                      exit_code: int) -> None:
    """Turn a non-zero ggml-rpc-server exit into a located ClusterError.

    Usage example: `assert_clean_exit(worker, server, server.wait())`.
    """
    if exit_code == 0:
        return
    output = server.stdout.read() if server.stdout else ""
    raise _rpc_failure(worker, "rpc_server.run", RuntimeError(
        f"{RPC_SERVER} terminou com codigo {exit_code}; saida: {output.strip()!r}"))
