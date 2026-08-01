"""UDP broadcast discovery of cluster nodes on a trusted LAN."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading

from ..io.udp import ReplyDecodeError, broadcast_probes, udp_socket
from .plan import ClusterNode

log = logging.getLogger("split_lm")

DISCOVERY_PORT = 29500
_PROBE = b"SPLIT_DISCOVER"
_REPLY_FIELDS = ("node_id", "hostname", "control_port", "cpu_count", "available_memory")


class DiscoveryReplyError(ReplyDecodeError):
    """A node answered discovery with a payload this version cannot read."""


def _available_memory() -> int:
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo sem linha MemAvailable: <kB>")


def local_node(node_id: str, control_port: int) -> ClusterNode:
    """Describe this machine so other nodes can plan a split for it.

    Usage example: `node = local_node("node-a", 29510)`.
    """
    return ClusterNode(node_id, socket.gethostname(), "", control_port,
                       os.cpu_count() or 1, _available_memory())


def encode_reply(node: ClusterNode) -> bytes:
    """Serialize a node for the wire; the address comes from the UDP source.

    Usage example: `payload = encode_reply(local_node("node-a", 29510))`.
    """
    return json.dumps({field: getattr(node, field) for field in _REPLY_FIELDS}).encode()


def decode_reply(payload: bytes, address: str) -> ClusterNode:
    """Rebuild a node from a discovery reply, tagging it with its source address.

    Usage example: `node = decode_reply(payload, "192.168.0.11")`.
    """
    try:
        fields = json.loads(payload)
        return ClusterNode(str(fields["node_id"]), str(fields["hostname"]), address,
                           int(fields["control_port"]), int(fields["cpu_count"]),
                           int(fields["available_memory"]))
    except (ValueError, KeyError, TypeError) as exc:
        raise DiscoveryReplyError(
            f"resposta invalida de {address}: {payload!r}; "
            f"esperado JSON com {_REPLY_FIELDS}"
        ) from exc


def serve_discovery(node: ClusterNode, port: int = DISCOVERY_PORT,
                    stop: threading.Event | None = None,
                    ready: threading.Event | None = None) -> None:
    """Answer discovery probes for this node until `stop` is set.

    `ready` is set once the port is bound. A caller that starts this in a
    thread and probes immediately would otherwise race the bind, and UDP sent
    to an unbound port is discarded with no queue to catch it up.

    Usage example: `serve_discovery(local_node("node-a", 29510), stop=stop)`.
    """
    reply = encode_reply(node)
    with udp_socket() as sock:
        sock.bind(("", port))
        sock.settimeout(0.2)
        if ready is not None:
            ready.set()
        log.info("discovery de %s ouvindo em udp/%d", node.node_id, port)
        while stop is None or not stop.is_set():
            try:
                probe, sender = sock.recvfrom(1024)
            except socket.timeout:
                continue
            if probe == _PROBE:
                sock.sendto(reply, sender)


def discover_nodes(port: int = DISCOVERY_PORT, timeout: float = 2.0,
                   broadcast_address: str = "255.255.255.255",
                   probes: int = 2) -> tuple[ClusterNode, ...]:
    """Broadcast probes and return the answering nodes, sorted by node id.

    Usage example: `nodes = discover_nodes(timeout=3.0)`.
    """
    found = broadcast_probes(_PROBE, port, timeout, broadcast_address, probes,
                             decode_reply, lambda node: node.node_id)
    return tuple(sorted(found.values(), key=lambda node: node.node_id))
