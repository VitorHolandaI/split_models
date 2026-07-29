import logging, socket, time

from ..io.network import _connect, _probe, _configure_connected_socket
from ..io.protocol import _send, _recv, _send_msg, NetworkStats, SocketClosedError
from ..model.loader import _build
from ..model.runner import _forward, _new_cache_state
from .profiling import _stat_add, _stat_inc, _profile_report

log = logging.getLogger("split_lm")


def _send_error(control, node_id, stage, exc):
    msg = {"type": "ERROR", "node": node_id, "stage": stage, "message": str(exc)}
    try: _send_msg(control, msg)
    except Exception: log.exception("WORKER falhou mandando erro: %s", msg)


def run_worker(model_name, layer_start, layer_end, n_total, device,
                tokenizer, max_new, listen_port, send_to, token_back_to,
                control_to, node_id, fake_error, profiling, precision="auto"):
    is_first = layer_start == 0
    is_last = layer_end == n_total
    log.info("WORKER %s testando head control %s", node_id, control_to)
    control = _connect(control_to, "head control")
    try:
        if fake_error == "startup":
            raise RuntimeError(f"erro fake de startup em {node_id}")
        if token_back_to:
            log.info("WORKER %s testando token-back %s", node_id, token_back_to)
            _probe(token_back_to, "head token-back")
        model = _build(
            model_name, layer_start, layer_end, n_total, device, precision)
        log.info("WORKER %s layers %d..%d (%d params)", node_id, layer_start,
                 layer_end - 1, sum(p.numel() for p in model.parameters()))

        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", listen_port)); srv.listen(1)
        log.info("WORKER %s escutando em :%d...", node_id, listen_port)

        if send_to:
            downstream = _connect(send_to, "downstream")
            log.info("WORKER %s conectou ao downstream %s", node_id, send_to)
        else:
            downstream = None

        if token_back_to:
            token_back = _connect(token_back_to, "token-back")
            log.info("WORKER %s token-back conectou ao head %s", node_id, token_back_to)
        else:
            token_back = None

        _send_msg(control, {"type": "READY", "node": node_id})
        upstream = _configure_connected_socket(srv.accept()[0])
        log.info("WORKER %s: upstream conectou", node_id)
    except Exception as exc:
        _send_error(control, node_id, "startup", exc)
        raise

    try:
        step = 0
        stats = {"total": 0.0} if profiling else None
        net_stats = NetworkStats() if profiling else None
        cache_state = _new_cache_state()
        total_start = time.perf_counter()
        while True:
            step += 1
            if fake_error == "run":
                raise RuntimeError(f"erro fake de run em {node_id}")
            start = time.perf_counter()
            if is_first:
                ids, reset_cache = _recv(upstream, device, net_stats)
                _stat_add(stats, "recv", time.perf_counter() - start)
                if reset_cache:
                    cache_state = _new_cache_state()
                start = time.perf_counter()
                h = _forward(model, None, device, True, is_last, ids,
                             cache_state)
                cache_state["pos"] += ids.shape[1]
            else:
                h, reset_cache = _recv(upstream, device, net_stats)
                recv_len = h.shape[1]
                _stat_add(stats, "recv", time.perf_counter() - start)
                if reset_cache:
                    cache_state = _new_cache_state()
                start = time.perf_counter()
                h = _forward(model, h, device, False, is_last,
                             cache_state=cache_state)
                cache_state["pos"] += recv_len
            _stat_add(stats, "forward", time.perf_counter() - start)

            if is_last:
                nid = h[0, -1].argmax()
                start = time.perf_counter()
                _send(token_back, nid.unsqueeze(0), net_stats)
                _stat_add(stats, "send_token", time.perf_counter() - start)
                _stat_inc(stats, "tokens")
                if not profiling:
                    log.info("WORKER %s: step %d token %d", node_id, step, nid.item())
            else:
                start = time.perf_counter()
                _send(downstream, h, net_stats, reset_cache)
                _stat_add(stats, "send_downstream", time.perf_counter() - start)
                _stat_inc(stats, "tokens")
                if not profiling:
                    log.info("WORKER %s: step %d encaminhado", node_id, step)
    except SocketClosedError:
        if stats is not None:
            stats["total"] = time.perf_counter() - total_start
            _profile_report(node_id, stats, net_stats)
        log.info("WORKER %s: upstream fechou", node_id)
    except ConnectionError as exc:
        _send_error(control, node_id, "run", exc)
        raise
    except Exception as exc:
        _send_error(control, node_id, "run", exc)
        raise

    srv.close()
    if downstream: downstream.close()
    if token_back: token_back.close()
    control.close()
