import logging, socket, sys, threading, time
from dataclasses import dataclass
import torch

from ..context_benchmark import build_context_ids, write_context_outputs
from ..io.network import _connect, _control_server, _wait_ready, _send_stop, _recv_token, _configure_connected_socket
from ..io.protocol import _send, NetworkStats
from ..metrics import (
    GenerationMetrics, append_metrics, build_generation_metrics, format_metrics)
from ..model.loader import _build
from ..model.runner import _forward, _new_cache_state
from .profiling import _stat_add, _stat_inc, _profile_report

log = logging.getLogger("split_lm")


def _stream_token(tokenizer, token_id):
    sys.stdout.write(tokenizer.decode(token_id))
    sys.stdout.flush()


def _read_prompt() -> str | None:
    try:
        return input(">>> ")
    except EOFError:
        return None


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


@dataclass
class GeneratedResponse:
    tokens: list[int]
    token_times: tuple[float, ...]
    started_at: float
    finished_at: float
    stats: dict[str, float | int] | None
    net_stats: NetworkStats | None
    token_socket: socket.socket


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


def _generate_response(tokenizer, ids, head_model, device, max_new, conn,
                       srv_tok, tok, errors, profiling, stream,
                       ignore_eos=False) -> GeneratedResponse:
    stats = {"total": 0.0} if profiling else None
    net_stats = NetworkStats() if profiling else None
    cache_state = _new_cache_state()
    out_tokens, token_times = [], []
    started_at = time.perf_counter()
    for step in range(max_new):
        _send_head_state(conn, ids, head_model, device, stats, cache_state,
                         net_stats, step == 0)
        if errors:
            raise RuntimeError(errors[-1])
        wait_started = time.perf_counter()
        token, tok = _recv_token(srv_tok, tok, device, net_stats)
        arrived_at = time.perf_counter()
        _stat_add(stats, "recv_token", arrived_at - wait_started)
        token_id = token.item()
        if token_id == tokenizer.eos_token_id and not ignore_eos:
            break
        out_tokens.append(token_id); token_times.append(arrived_at)
        if stream:
            _stream_token(tokenizer, token_id)
        ids = torch.cat([ids, torch.tensor([[token_id]], device=device)], dim=1)
        _stat_inc(stats, "tokens")
    finished_at = time.perf_counter()
    if stats is not None:
        stats["total"] = finished_at - started_at
    return GeneratedResponse(
        out_tokens, tuple(token_times), started_at, finished_at,
        stats, net_stats, tok)


def _record_response(model_name: str, prompt_tokens: int, worker_count: int,
                     metrics_file: str, response: GeneratedResponse) -> GenerationMetrics:
    metrics = build_generation_metrics(
        "split", model_name, prompt_tokens, response.started_at,
        response.finished_at, response.token_times, worker_count)
    append_metrics(metrics_file, metrics)
    print(format_metrics(metrics, color=sys.stderr.isatty()),
          file=sys.stderr, flush=True)
    if response.stats is not None:
        _profile_report("head", response.stats, response.net_stats)
    return metrics


def run_head(model_name, n_total, device, tokenizer, max_new, interactive,
             worker_addr, token_port, control_port, worker_count, ready_timeout,
             head_layer_start, head_layer_end, profiling, prompt_text,
             metrics_file, context_sizes=(), benchmark_output="benchmark-results/context-sweep"):
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

    if context_sizes:
        results = []
        csv_path = svg_path = None
        for context_size in context_sizes:
            ids = build_context_ids(tokenizer, context_size, device)
            print(f"\nBENCHMARK context={context_size} generated_max={max_new}", flush=True)
            response = _generate_response(
                tokenizer, ids, head_model, device, max_new, conn,
                srv_tok, tok, errors, True, False, True)
            tok = response.token_socket
            print(tokenizer.decode(response.tokens, skip_special_tokens=True))
            results.append(_record_response(
                model_name, context_size, worker_count, metrics_file, response))
            csv_path, svg_path = write_context_outputs(results, benchmark_output)
        assert csv_path is not None and svg_path is not None
        print(f"BENCHMARK outputs: {csv_path} {svg_path}", flush=True)
    elif interactive:
        print("\nChat. /exit pra sair.\n")
        history = []
        while True:
            prompt = _read_prompt()
            if prompt is None or prompt.strip() == "/exit": break
            history.append({"role": "user", "content": prompt})
            ids = _prompt_ids(tokenizer, prompt, device, history)
            log.info("HEAD: %d tokens", ids.shape[1])
            response = _generate_response(
                tokenizer, ids, head_model, device, max_new, conn,
                srv_tok, tok, errors, profiling, not profiling)
            tok = response.token_socket
            if profiling:
                print(tokenizer.decode(response.tokens))
            answer = tokenizer.decode(response.tokens, skip_special_tokens=True)
            history.append({"role": "assistant", "content": answer})
            sys.stdout.write("\n")
            sys.stdout.flush()
            _record_response(
                model_name, int(ids.shape[1]), worker_count, metrics_file, response)
    else:
        prompt = prompt_text
        ids = _prompt_ids(tokenizer, prompt, device)
        log.info("HEAD: %r -> %d tokens", prompt, ids.shape[1])
        response = _generate_response(
            tokenizer, ids, head_model, device, max_new, conn,
            srv_tok, tok, errors, profiling, not profiling)
        tok = response.token_socket
        if profiling:
            print(tokenizer.decode(response.tokens))
        _record_response(
            model_name, int(ids.shape[1]), worker_count, metrics_file, response)
        sys.stdout.write("\n")
        sys.stdout.flush()

    _send_stop(controls)
    conn.close(); srv_tok.close()
