"""Always-on UDP discovery for controlling every llama.cpp appliance role."""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass

from ..io.udp import ReplyDecodeError, broadcast_probes, udp_socket
from .mode_store import StateStore

log = logging.getLogger("split_lm")

FLEET_DISCOVERY_PORT = 29521
PROTOCOL = 1
_PROBE = b"SPLIT_LLAMA_FLEET_DISCOVER"
_REPLY_FIELDS = ("protocol", "node_id", "hostname", "mode")


class FleetReplyError(ReplyDecodeError):
    """A control agent answered with an invalid fleet payload."""


@dataclass(frozen=True)
class FleetNode:
    """One appliance reachable through the always-on control plane."""

    node_id: str
    hostname: str
    address: str
    mode: str


def encode_reply(node: FleetNode) -> bytes:
    """Serialize a control node without trusting its view of its own address."""
    return json.dumps({"protocol": PROTOCOL, "node_id": node.node_id,
                       "hostname": node.hostname, "mode": node.mode}).encode()


def decode_reply(payload: bytes, address: str) -> FleetNode:
    """Decode a fleet reply and attach the UDP source address."""
    try:
        fields = json.loads(payload)
        _check_protocol(fields, address)
        return FleetNode(str(fields["node_id"]), str(fields["hostname"]),
                         address, str(fields["mode"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise FleetReplyError(
            f"resposta de controle invalida de {address}: {payload!r}; "
            f"esperado JSON com {_REPLY_FIELDS}") from exc


def _check_protocol(fields: object, address: str) -> None:
    if not isinstance(fields, dict):
        raise FleetReplyError(
            f"resposta de controle de {address}={fields!r}; esperado objeto JSON")
    if fields.get("protocol") != PROTOCOL:
        raise FleetReplyError(
            f"protocolo de controle {fields.get('protocol')!r} de {address}; "
            f"esperado {PROTOCOL}. Reconstrua a ISO desse no")


def _local_node(node_id: str, store: StateStore) -> FleetNode:
    return FleetNode(node_id, socket.gethostname(), "", store.read_state().mode.value)


def serve_fleet_discovery(node_id: str, store: StateStore, *,
                          port: int = FLEET_DISCOVERY_PORT,
                          stop: threading.Event | None = None,
                          ready: threading.Event | None = None) -> None:
    """Answer control probes regardless of the current inference mode."""
    with udp_socket() as sock:
        sock.bind(("", port))
        sock.settimeout(0.2)
        if ready is not None:
            ready.set()
        log.info("discovery de controle de %s ouvindo em udp/%d", node_id, port)
        _answer_probes(sock, node_id, store, stop)


def _answer_probes(sock: socket.socket, node_id: str, store: StateStore,
                   stop: threading.Event | None) -> None:
    while stop is None or not stop.is_set():
        try:
            probe, sender = sock.recvfrom(1024)
        except socket.timeout:
            continue
        if probe == _PROBE:
            sock.sendto(encode_reply(_local_node(node_id, store)), sender)


def discover_fleet_nodes(port: int = FLEET_DISCOVERY_PORT, timeout: float = 2.0,
                         broadcast_address: str = "255.255.255.255",
                         probes: int = 4) -> tuple[FleetNode, ...]:
    """Discover control agents, including replicas and distributed masters."""
    found = broadcast_probes(_PROBE, port, timeout, broadcast_address, probes,
                             decode_reply, lambda node: node.node_id)
    return tuple(sorted(found.values(), key=lambda node: node.node_id))
