"""Readiness probes for the supervised server and the RPC port it replaces.

Both transitions poll: the server until `GET /health` answers 200 (llama-server
answers 503 while the model downloads and loads, which is exactly the window a
transition must wait through), and the worker port until it closes, so a node
never advertises itself as worker and master at the same time.
"""

from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable


class HttpHealthProbe:
    """`GET /health` against the local llama-server, with an injectable opener.

    Usage example: `ready = HttpHealthProbe().ready("http://127.0.0.1:8080/health")`.
    """

    def __init__(self, opener: Callable[[str], object] = urllib.request.urlopen
                 ) -> None:
        self._opener = opener

    def ready(self, url: str) -> bool:
        try:
            self._opener(url)
        except (urllib.error.URLError, OSError):
            # HTTPError (a URLError) covers 503 "loading model"; URLError and
            # OSError cover the port not accepting connections yet.
            return False
        return True


class TcpPortProbe:
    """Whether a TCP port on this node still accepts connections.

    Usage example: `still_up = TcpPortProbe("192.168.0.11").open(50052)`.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._host = host

    def open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            return probe.connect_ex((self._host, port)) == 0


def wait_until(check: Callable[[], bool], timeout: float, *,
               sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic,
               interval: float = 0.5) -> bool:
    """Poll `check` until it holds or `timeout` seconds pass.

    Clock and sleeper are parameters so tests advance a fake clock instead of
    waiting real seconds.

    Usage example: `ok = wait_until(probe.ready, 30.0)`.
    """
    deadline = clock() + timeout
    while True:
        if check():
            return True
        if clock() >= deadline:
            return False
        sleep(min(interval, max(deadline - clock(), 0.0)))
