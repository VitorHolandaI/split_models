"""UDP broadcast discovery shared by the PyTorch and llama.cpp fleets.

The two fleets speak different payloads on different ports, but the probing
loop is the same and its timing was subtle enough to get wrong once already
(probes sent as a burst never reach a node that is still binding). One copy.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Callable, TypeVar

log = logging.getLogger("split_lm")

Reply = TypeVar("Reply")


class ReplyDecodeError(ValueError):
    """A node answered discovery with a payload this version cannot read."""


def udp_socket() -> socket.socket:
    """Create a UDP socket ready for broadcast discovery on a LAN.

    Usage example: `with udp_socket() as sock: sock.bind(("", 29500))`.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    return sock


def collect_replies(sock: socket.socket, deadline: float,
                    decode: Callable[[bytes, str], Reply],
                    identify: Callable[[Reply], str]) -> dict[str, Reply]:
    """Read replies until `deadline`, keeping one entry per identity."""
    found: dict[str, Reply] = {}
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return found
        sock.settimeout(remaining)
        try:
            payload, sender = sock.recvfrom(4096)
        except socket.timeout:
            return found
        # A malformed reply is one bad node, not a failed discovery: log the
        # offending payload and keep listening for the healthy ones.
        try:
            reply = decode(payload, sender[0])
        except ReplyDecodeError as exc:
            log.warning("%s", exc)
            continue
        found[identify(reply)] = reply


# Probes spread across the window, never as a burst. Sent back to back the N
# probes leave in the same instant and a node still binding its port misses all
# of them: UDP to an unbound port is discarded with no queue, so repeating only
# helps when the repeats are spaced out.
def _probe_window(deadline: float, timeout: float, attempt: int, probes: int) -> float:
    if attempt == probes - 1:
        return deadline
    return min(deadline, time.monotonic() + timeout / probes)


def broadcast_probes(probe: bytes, port: int, timeout: float,
                     broadcast_address: str, probes: int,
                     decode: Callable[[bytes, str], Reply],
                     identify: Callable[[Reply], str]) -> dict[str, Reply]:
    """Broadcast `probe` repeatedly and gather the answers by identity.

    Usage example:
        `found = broadcast_probes(b"PROBE", 29500, 2.0, "255.255.255.255", 2,
                                  decode_reply, lambda node: node.node_id)`.
    """
    found: dict[str, Reply] = {}
    with udp_socket() as sock:
        sock.bind(("", 0))
        deadline = time.monotonic() + timeout
        for attempt in range(probes):
            sock.sendto(probe, (broadcast_address, port))
            window = _probe_window(deadline, timeout, attempt, probes)
            found.update(collect_replies(sock, window, decode, identify))
            if time.monotonic() >= deadline:
                break
    return found
