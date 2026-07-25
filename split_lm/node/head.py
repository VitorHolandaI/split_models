import logging, socket, sys, threading, time
import torch

from ..io.network import _connect, _control_server, _wait_ready, _send_stop, _recv_token, _configure_connected_socket
from ..io.protocol import _send, NetworkStats
from ..model.loader import _build
from ..model.runner import _forward, _new_cache_state
from .profiling import _stat_add, _stat_inc, _profile_report

log = logging.getLogger("split_lm")


def _stream_token(tokenizer, token_id):
    sys.stdout.write(tokenizer.decode(token_id))
    sys.stdout.flush()


def _prompt_ids(tokenizer, prompt, device, messages=None):
    encoded = tokenizer.apply_chat_template(
        messages or [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ) if getattr(tokenizer, "chat_template", None) else None
    if encoded is not None:
        return encoded["input_ids"].to(device)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def _send_head_state(conn, ids, head_model, device, stats, cache_state,
                     net_stats=None, reset_cache=False):
    chunk = ids if cache_state is None else ids[:, cache_state["pos"]:]
    if head_model is None:
        start = time.perf_counter()
        _send(conn, chunk, net_stats, reset_cache)
        _stat_add(stats, "send", time.perf_counter() - start)
        if cache_state is not None:
            cache_state["pos"] += chunk.shape[1]
        return
    start = time.perf_counter()
    hidden = _forward(head_model, None, device, True, False, chunk,
                      cache_state)
    _stat_add(stats, "forward", time.perf_counter() - start)
    start = time.perf_counter()
    _send(conn, hidden, net_stats, reset_cache)
    _stat_add(stats, "send", time.perf_counter() - start)
    if cache_state is not None:
        cache_state["pos"] += chunk.shape[1]


def run_head(model_name, n_total, device, tokenizer, max_new, interactive,
             worker_addr, token_port, control_port, worker_count, ready_timeout,
             head_layer_start, head_layer_end, profiling, prompt_text):
    ready, errors, controls, lock = set(), [], {}, threading.Lock()
    threading.Thread(target=_control_server,
                     args=(control_port, ready, errors, controls, lock), daemon=True).start()

    srv_tok = socket.socket(); srv_tok.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_tok.bind(("0.0.0.0", token_port)); srv_tok.listen(8)
    log.info("HEAD aguardando token em :%d...", token_port)

    head_model = None
    if head_layer_start is not None:
        head_model = _build(model_name, head_layer_start, head_layer_end, n_total, device)
        log.info("HEAD slice layers %d..%d", head_layer_start, head_layer_end - 1)

    _wait_ready(worker_count, ready, errors, lock, ready_timeout)
    log.info("HEAD conectando ao worker %s", worker_addr)
    conn = _connect(worker_addr, "worker")
    log.info("HEAD conectou ao worker %s", worker_addr)
    tok = _configure_connected_socket(srv_tok.accept()[0])
    log.info("HEAD: token-back conectou")

    stats = {"total": 0.0} if profiling else None
    net_stats = NetworkStats() if profiling else None
    if interactive:
        print("\nChat. /exit pra sair.\n")
        history = []
        while True:
            prompt = input(">>> ")
            if prompt.strip() == "/exit": break
            history.append({"role": "user", "content": prompt})
            ids = _prompt_ids(tokenizer, prompt, device, history)
            log.info("HEAD: %d tokens", ids.shape[1])
            stats = {"total": 0.0} if profiling else None
            cache_state = _new_cache_state()
            out_tokens = []
            total_start = time.perf_counter()
            for step in range(max_new):
                _send_head_state(conn, ids, head_model, device, stats, cache_state,
                                 net_stats, step == 0)
                if errors: raise RuntimeError(errors[-1])
                start = time.perf_counter()
                token, tok = _recv_token(srv_tok, tok, device, net_stats)
                _stat_add(stats, "recv_token", time.perf_counter() - start)
                nid = token.item()
                if nid == tokenizer.eos_token_id: break
                out_tokens.append(nid)
                if not profiling:
                    _stream_token(tokenizer, nid)
                ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
                _stat_inc(stats, "tokens")
            if profiling:
                stats["total"] = time.perf_counter() - total_start
                print(tokenizer.decode(out_tokens))
                _profile_report("head", stats, net_stats)
            answer = tokenizer.decode(out_tokens, skip_special_tokens=True)
            history.append({"role": "assistant", "content": answer})
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        prompt = prompt_text
        ids = _prompt_ids(tokenizer, prompt, device)
        log.info("HEAD: %r -> %d tokens", prompt, ids.shape[1])
        stats = {"total": 0.0} if profiling else None
        cache_state = _new_cache_state()
        out_tokens = []
        total_start = time.perf_counter()
        for step in range(max_new):
            _send_head_state(conn, ids, head_model, device, stats, cache_state,
                             net_stats, step == 0)
            if errors: raise RuntimeError(errors[-1])
            start = time.perf_counter()
            token, tok = _recv_token(srv_tok, tok, device, net_stats)
            _stat_add(stats, "recv_token", time.perf_counter() - start)
            nid = token.item()
            if nid == tokenizer.eos_token_id: break
            out_tokens.append(nid)
            if not profiling:
                _stream_token(tokenizer, nid)
            ids = torch.cat([ids, torch.tensor([[nid]], device=device)], dim=1)
            _stat_inc(stats, "tokens")
        if profiling:
            stats["total"] = time.perf_counter() - total_start
            print(tokenizer.decode(out_tokens))
            _profile_report("head", stats, net_stats)
        sys.stdout.write("\n")
        sys.stdout.flush()

    _send_stop(controls)
    conn.close(); srv_tok.close()
