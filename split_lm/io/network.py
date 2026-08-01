import logging, socket, threading, time

from .framing import _recv_msg, _send_msg
from .protocol import _recv

log = logging.getLogger("split_lm")


def _tcp_socket():
    sock = socket.socket()
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def _configure_connected_socket(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


def _connect(addr, label, attempts=120):
    last = None
    for _ in range(attempts):
        s = None
        try:
            s = _tcp_socket()
            s.connect(addr)
            return s
        except OSError as exc:
            if s is not None:
                s.close()
            last = exc; time.sleep(1)
    raise ConnectionError(f"falha conectando {label} em {addr}: {last}")


def _probe(addr, label, timeout=3):
    try:
        s = _tcp_socket()
        s.settimeout(timeout)
        s.connect(addr)
        s.close()
    except OSError as exc:
        raise ConnectionError(f"probe falhou {label} em {addr}: {exc}") from exc


def _control_server(port, ready, errors, controls, lock):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen()
    log.info("HEAD control escutando em :%d...", port)
    while True:
        conn = _configure_connected_socket(srv.accept()[0])
        threading.Thread(target=_control_reader,
                         args=(conn, ready, errors, controls, lock), daemon=True).start()


def _control_reader(conn, ready, errors, controls, lock):
    while True:
        try:
            msg = _recv_msg(conn)
        except Exception:
            return
        node = msg.get("node", "unknown")
        with lock:
            controls[node] = conn
            if msg.get("type") == "READY": ready.add(node)
            if msg.get("type") == "ERROR": errors.append(msg)
        log.info("HEAD control: %s", msg)


def _wait_ready(worker_count, ready, errors, lock, timeout):
    deadline = time.time() + timeout if timeout else None
    while True:
        with lock:
            if errors: raise RuntimeError(errors[-1])
            if len(ready) >= worker_count: return
        if deadline and time.time() > deadline:
            raise TimeoutError(f"esperava {worker_count} READY, recebeu {len(ready)}")
        time.sleep(0.05)


def _send_stop(controls):
    for control in controls.values():
        try: _send_msg(control, {"type": "STOP", "node": "head"})
        except OSError: pass


def _recv_token(srv_tok, tok, device, net_stats=None):
    while True:
        try:
            token, _ = _recv(tok, device, net_stats)
            return token, tok
        except Exception:
            tok.close()
            tok = _configure_connected_socket(srv_tok.accept()[0])
