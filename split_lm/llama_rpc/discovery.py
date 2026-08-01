"""UDP discovery of llama.cpp RPC workers on a trusted LAN.

Separate port and payload from `split_lm.cluster.discovery` on purpose. The two
fleets answer different questions, and a PyTorch worker replying to a llama
probe would be accepted here and then fail at the RPC handshake, far from the
cause. The probing loop itself is shared, in `split_lm.io.udp`.

The reply carries only what stays true for the life of the process. Devices and
free memory are deliberately absent: the RPC backend reports those to llama.cpp
after the connection, so duplicating them here would only let the two disagree.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass

from ..io.udp import ReplyDecodeError, broadcast_probes, udp_socket

log = logging.getLogger("split_lm")

DISCOVERY_PORT = 29520
RPC_PORT = 50052
PROTOCOL = 1
_PROBE = b"SPLIT_LLAMA_DISCOVER"
_REPLY_FIELDS = ("protocol", "node_id", "hostname", "rpc_port")


class RpcReplyError(ReplyDecodeError):
    """A worker answered discovery with a payload this version cannot read."""


@dataclass(frozen=True)
class RpcWorker:
    """One llama.cpp RPC worker reachable on the LAN."""

    node_id: str
    hostname: str
    address: str
    rpc_port: int

    @property
    def endpoint(self) -> str:
        """Address in the form llama-cli expects after `--rpc`."""
        return f"{self.address}:{self.rpc_port}"


def local_worker(node_id: str, rpc_port: int = RPC_PORT) -> RpcWorker:
    """Describe this machine for its discovery replies.

    The address stays empty: it is filled in by the receiver from the UDP
    source, so a worker never has to guess which of its interfaces was used.

    Usage example: `worker = local_worker("node-abcd1234")`.
    """
    return RpcWorker(node_id, socket.gethostname(), "", rpc_port)


def encode_reply(worker: RpcWorker) -> bytes:
    """Serialize a worker for the wire; the address comes from the UDP source.

    Usage example: `payload = encode_reply(local_worker("node-abcd1234"))`.
    """
    return json.dumps({"protocol": PROTOCOL, "node_id": worker.node_id,
                       "hostname": worker.hostname,
                       "rpc_port": worker.rpc_port}).encode()


def decode_reply(payload: bytes, address: str) -> RpcWorker:
    """Rebuild a worker from a reply, tagging it with its source address.

    Usage example: `worker = decode_reply(payload, "192.168.0.11")`.
    """
    try:
        fields = json.loads(payload)
        _check_protocol(fields, address)
        return RpcWorker(str(fields["node_id"]), str(fields["hostname"]),
                         address, int(fields["rpc_port"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise RpcReplyError(
            f"resposta invalida de {address}: {payload!r}; "
            f"esperado JSON com {_REPLY_FIELDS}"
        ) from exc


# A mismatched protocol is refused here rather than at the RPC handshake, where
# the error would name a socket instead of the node that needs reinstalling.
def _check_protocol(fields: object, address: str) -> None:
    if not isinstance(fields, dict):
        raise RpcReplyError(f"resposta de {address}={fields!r}; esperado objeto JSON")
    if fields.get("protocol") != PROTOCOL:
        raise RpcReplyError(
            f"protocolo {fields.get('protocol')!r} de {address}; "
            f"esperado {PROTOCOL}. Reconstrua a ISO desse no")


def serve_discovery(worker: RpcWorker, port: int = DISCOVERY_PORT,
                    stop: threading.Event | None = None,
                    ready: threading.Event | None = None) -> None:
    """Answer discovery probes for this worker until `stop` is set.

    `ready` is set once the port is bound, so a caller can sequence startup
    instead of racing the bind.

    Usage example: `serve_discovery(local_worker("node-a"), stop=stop)`.
    """
    reply = encode_reply(worker)
    with udp_socket() as sock:
        sock.bind(("", port))
        sock.settimeout(0.2)
        if ready is not None:
            ready.set()
        log.info("discovery llama de %s ouvindo em udp/%d", worker.node_id, port)
        while stop is None or not stop.is_set():
            try:
                probe, sender = sock.recvfrom(1024)
            except socket.timeout:
                continue
            if probe == _PROBE:
                sock.sendto(reply, sender)


def discover_workers(port: int = DISCOVERY_PORT, timeout: float = 2.0,
                     broadcast_address: str = "255.255.255.255",
                     probes: int = 4) -> tuple[RpcWorker, ...]:
    """Broadcast probes and return the answering workers, sorted by node id.

    Sorting is what makes the resulting `--rpc` list reproducible across runs,
    so the same fleet always yields the same layer assignment.

    Usage example: `workers = discover_workers(timeout=3.0)`.
    """
    found = broadcast_probes(_PROBE, port, timeout, broadcast_address, probes,
                             decode_reply, lambda worker: worker.node_id)
    return tuple(sorted(found.values(), key=lambda worker: worker.node_id))


def rpc_endpoints(workers: tuple[RpcWorker, ...]) -> str:
    """Join workers into the comma-separated list `llama-cli --rpc` expects.

    Usage example: `endpoints = rpc_endpoints(discover_workers())`.
    """
    if not workers:
        raise ValueError("nenhum worker RPC encontrado; esperado ao menos um")
    return ",".join(worker.endpoint for worker in workers)
