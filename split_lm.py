import argparse, io, json, logging, socket, struct, sys, threading, time, warnings
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("split_lm")


def _stat_add(stats, key, seconds):
    if stats is None:
        return
    stats[key] = stats.get(key, 0.0) + seconds


def _stat_inc(stats, key, count=1):
    if stats is None:
        return
    stats[key] = stats.get(key, 0) + count


def _profile_report(node, stats):
    if not stats:
        return
    total = stats.get("total", 0.0)
    parts = [f"PROFILE {node}: tokens={int(stats.get('tokens', 0))} total={total:.3f}s"]
    for key in sorted(k for k in stats if k not in {"tokens", "total"}):
        value = stats[key]
        pct = (value / total * 100.0) if total else 0.0
        parts.append(f"{key}={value:.3f}s ({pct:.1f}%)")
    print(" | ".join(parts), file=sys.stderr, flush=True)


def _stream_token(tokenizer, token_id):
    sys.stdout.write(tokenizer.decode(token_id))
    sys.stdout.flush()


def _load_weights(model_name, layer_start, layer_end, n_total, config):
    shard_paths = _weight_paths(model_name, layer_start, layer_end, n_total)
    is_head = layer_start == 0
    is_last = layer_end == n_total
    w = {}
    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as sd:
            for key in sd.keys():
                if key == "model.embed_tokens.weight" and is_head:
                    w[key] = sd.get_tensor(key)
                if key == "model.norm.weight" and is_last:
                    w[key] = sd.get_tensor(key)
                if key == "model.embed_tokens.weight" and is_last:
                    w["lm_head.weight"] = sd.get_tensor(key)
                if key == "model.rotary_emb.inv_freq" and is_head:
                    w[key] = sd.get_tensor(key)
                if key.startswith("model.layers."):
                    lidx = int(key.split(".")[2])
                    if layer_start <= lidx < layer_end:
                        nk = key.replace(f".{lidx}.", f".{lidx - layer_start}.")
                        w[nk] = sd.get_tensor(key)
    return w


def _wanted_weight(key, layer_start, layer_end, n_total):
    if key == "model.embed_tokens.weight": return layer_start == 0 or layer_end == n_total
    if key == "model.norm.weight": return layer_end == n_total
    if key == "model.rotary_emb.inv_freq": return layer_start == 0
    if not key.startswith("model.layers."): return False
    lidx = int(key.split(".")[2])
    return layer_start <= lidx < layer_end


def _weight_paths(model_name, layer_start, layer_end, n_total):
    try:
        return [hf_hub_download(repo_id=model_name, filename="model.safetensors")]
    except Exception:
        index_path = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    names = {name for key, name in index["weight_map"].items()
             if _wanted_weight(key, layer_start, layer_end, n_total)}
    return [hf_hub_download(repo_id=model_name, filename=name) for name in sorted(names)]


def _build(model_name, layer_start, layer_end, n_total, device):
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_hidden_layers = layer_end - layer_start
    model = AutoModelForCausalLM.from_config(cfg)
    w = _load_weights(model_name, layer_start, layer_end, n_total, cfg)
    model.load_state_dict(w, strict=False)
    model = model.to(device).eval()
    return model


def _causal_mask(seq_len, dev, dtype):
    m = torch.full((seq_len, seq_len), float("-inf"), device=dev, dtype=dtype)
    return torch.triu(m, diagonal=1).unsqueeze(0).unsqueeze(0)


def _recvall(s, n):
    d = b""
    while len(d) < n:
        c = s.recv(n - len(d), socket.MSG_WAITALL)
        if not c: raise ConnectionError("fechou")
        d += c
    return d


def _send(s, t):
    b = io.BytesIO(); torch.save(t.cpu(), b); d = b.getvalue()
    s.sendall(struct.pack("!I", len(d))); s.sendall(d)


def _recv(s, dev):
    n, = struct.unpack("!I", _recvall(s, 4))
    return torch.load(io.BytesIO(_recvall(s, n)), map_location=dev, weights_only=True)


def _send_msg(s, msg):
    d = json.dumps(msg).encode("utf-8")
    s.sendall(struct.pack("!I", len(d))); s.sendall(d)


def _recv_msg(s):
    n, = struct.unpack("!I", _recvall(s, 4))
    return json.loads(_recvall(s, n).decode("utf-8"))


def _control_server(port, ready, errors, controls, lock):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen()
    log.info("HEAD control escutando em :%d...", port)
    while True:
        conn = srv.accept()[0]
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


def _connect(addr, label, attempts=120):
    last = None
    for _ in range(attempts):
        try:
            s = socket.socket(); s.connect(addr); return s
        except OSError as exc:
            last = exc; time.sleep(1)
    raise ConnectionError(f"falha conectando {label} em {addr}: {last}")


def _probe(addr, label, timeout=3):
    try:
        s = socket.create_connection(addr, timeout=timeout)
        s.close()
    except OSError as exc:
        raise ConnectionError(f"probe falhou {label} em {addr}: {exc}") from exc


def _send_stop(controls):
    for control in controls.values():
        try: _send_msg(control, {"type": "STOP", "node": "head"})
        except OSError: pass


def _recv_token(srv_tok, tok, device):
    while True:
        try:
            return _recv(tok, device), tok
        except Exception:
            tok.close()
            tok = srv_tok.accept()[0]


def _send_head_state(conn, ids, head_model, device, stats):
    if head_model is None:
        start = time.perf_counter()
        _send(conn, ids)
        _stat_add(stats, "send", time.perf_counter() - start)
        return
    start = time.perf_counter()
    hidden = _forward(head_model, None, device, True, False, ids)
    _stat_add(stats, "forward", time.perf_counter() - start)
    start = time.perf_counter()
    _send(conn, hidden)
    _stat_add(stats, "send", time.perf_counter() - start)


def _forward(model, hidden, dev, is_head, is_last, input_ids=None):
    with torch.no_grad():
        if is_head:
            hidden = model.model.embed_tokens(input_ids)
            seq_len = input_ids.shape[1]
        else:
            seq_len = hidden.shape[1]
        pos = torch.arange(seq_len, device=dev).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, pos)
        pe = (cos, sin)
        mask = _causal_mask(seq_len, dev, hidden.dtype)
        for layer in model.model.layers:
            hidden = layer(hidden, position_embeddings=pe, attention_mask=mask)
        if is_last:
            hidden = model.model.norm(hidden)
            return model.lm_head(hidden)
    return hidden


# ── head ──────────────────────────────────────────────────────────────────

def run_head(model_name, n_total, device, tokenizer, max_new, interactive,
             worker_addr, token_port, control_port, worker_count, ready_timeout,
             head_layer_start, head_layer_end, profiling, prompt_text):
    """Head conecta no primeiro worker, escuta token de volta."""
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
    tok = srv_tok.accept()[0]; log.info("HEAD: token-back conectou")

    if interactive:
        print("\nChat. /exit pra sair.\n")
        history = []
        while True:
            prompt = input(">>> ")
            if prompt.strip() == "/exit": break
            history.append(f"User: {prompt}")
            full = "\n".join(history) + "\nAssistant:"
            ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
            log.info("HEAD: %d tokens", ids.shape[1])
            stats = {"total": 0.0} if profiling else None
            out_tokens = []
            total_start = time.perf_counter()
            for step in range(max_new):
                _send_head_state(conn, ids, head_model, device, stats)
                if errors: raise RuntimeError(errors[-1])
                start = time.perf_counter()
                token, tok = _recv_token(srv_tok, tok, device)
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
                _profile_report("head", stats)
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        prompt = prompt_text
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        log.info("HEAD: %r -> %d tokens", prompt, ids.shape[1])
        stats = {"total": 0.0} if profiling else None
        out_tokens = []
        total_start = time.perf_counter()
        for step in range(max_new):
            _send_head_state(conn, ids, head_model, device, stats)
            if errors: raise RuntimeError(errors[-1])
            start = time.perf_counter()
            token, tok = _recv_token(srv_tok, tok, device)
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
            _profile_report("head", stats)
        sys.stdout.write("\n")
        sys.stdout.flush()

    _send_stop(controls)
    conn.close(); srv_tok.close()


# ── worker ────────────────────────────────────────────────────────────────

def _send_error(control, node_id, stage, exc):
    msg = {"type": "ERROR", "node": node_id, "stage": stage, "message": str(exc)}
    try: _send_msg(control, msg)
    except Exception: log.exception("WORKER falhou mandando erro: %s", msg)


def run_worker(model_name, layer_start, layer_end, n_total, device,
                tokenizer, max_new, listen_port, send_to, token_back_to,
                control_to, node_id, fake_error, profiling):
    """Worker: escuta upstream, conecta no downstream, manda token pro head."""
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
        model = _build(model_name, layer_start, layer_end, n_total, device)
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
        upstream = srv.accept()[0]
        log.info("WORKER %s: upstream conectou", node_id)
    except Exception as exc:
        _send_error(control, node_id, "startup", exc)
        raise

    try:
        step = 0
        stats = {"total": 0.0} if profiling else None
        total_start = time.perf_counter()
        while True:
            step += 1
            if fake_error == "run":
                raise RuntimeError(f"erro fake de run em {node_id}")
            start = time.perf_counter()
            if is_first:
                ids = _recv(upstream, device)
                _stat_add(stats, "recv", time.perf_counter() - start)
                start = time.perf_counter()
                h = _forward(model, None, device, True, is_last, ids)
            else:
                h = _recv(upstream, device)
                _stat_add(stats, "recv", time.perf_counter() - start)
                start = time.perf_counter()
                h = _forward(model, h, device, False, is_last)
            _stat_add(stats, "forward", time.perf_counter() - start)

            if is_last:
                nid = h[0, -1].argmax()
                start = time.perf_counter()
                _send(token_back, nid.unsqueeze(0))
                _stat_add(stats, "send_token", time.perf_counter() - start)
                _stat_inc(stats, "tokens")
                if not profiling:
                    log.info("WORKER %s: step %d token %d", node_id, step, nid.item())
            else:
                start = time.perf_counter()
                _send(downstream, h)
                _stat_add(stats, "send_downstream", time.perf_counter() - start)
                _stat_inc(stats, "tokens")
                if not profiling:
                    log.info("WORKER %s: step %d encaminhado", node_id, step)
    except ConnectionError as exc:
        if str(exc) != "fechou":
            _send_error(control, node_id, "run", exc)
            raise
        if stats is not None:
            stats["total"] = time.perf_counter() - total_start
            _profile_report(node_id, stats)
        log.info("WORKER %s: upstream fechou", node_id)
    except Exception as exc:
        _send_error(control, node_id, "run", exc)
        raise

    srv.close()
    if downstream: downstream.close()
    if token_back: token_back.close()
    control.close()


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="split LLM")
    ap.add_argument("role", choices=["head", "worker", "probe"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layer-start", type=int, default=None)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--total-layers", type=int, default=None)
    ap.add_argument("--port", type=int, default=9999, help="porta que este no escuta (worker) ou token-back (head)")
    ap.add_argument("--control-port", type=int, default=10001, help="head: porta de controle READY/ERROR")
    ap.add_argument("--control-to", default=None, help="worker: addr:port do head control")
    ap.add_argument("--node-id", default=None, help="worker: nome do no no canal de controle")
    ap.add_argument("--workers", type=int, default=1, help="head: quantos workers esperar READY")
    ap.add_argument("--ready-timeout", type=float, default=120, help="head: segundos esperando READY")
    ap.add_argument("--fake-error", choices=["startup", "run"], default=None, help="worker: injeta erro para testar controle")
    ap.add_argument("--connect", default=None, help="head: addr:port do primeiro worker | worker: addr:port do proximo")
    ap.add_argument("--probe-to", action="append", default=[], help="probe: addr:port para testar")
    ap.add_argument("--token-back-to", default=None, help="worker: addr:port do head pra token (ultimo worker)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--prompt", default="Write a short poem about distributed computers.")
    ap.add_argument("--profiling", action="store_true", help="mede tempos e imprime resumo no fim da request")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="mostra logs INFO tambem no head")
    args = ap.parse_args()

    def _parse(s):
        if not s: return None
        p = s.split(":"); return (p[0], int(p[1]))

    if args.role == "probe":
        if not args.probe_to:
            print("probe precisa de --probe-to addr:port")
            return
        for target in args.probe_to:
            _probe(_parse(target), target)
            print(f"OK {target}")
        return

    if args.role == "head" and not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
        log.setLevel(logging.WARNING)

    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(args.model)
    if t.pad_token is None: t.pad_token = t.eos_token

    cfg = AutoConfig.from_pretrained(args.model)
    n_total = args.total_layers or cfg.num_hidden_layers

    if args.role == "head" and (args.layer_start is not None or args.layer_end is not None):
        if args.layer_start != 0 or args.layer_end is None or args.layer_end >= n_total:
            raise ValueError("head slice precisa de --layer-start 0 e --layer-end entre 1 e total_layers-1")

    if args.role == "head":
        if not args.connect:
            print("head precisa de --connect addr:port do primeiro worker")
            return
        run_head(args.model, n_total, args.device, t, args.max_new_tokens,
                 args.interactive, _parse(args.connect), args.port,
                 args.control_port, args.workers, args.ready_timeout,
                 args.layer_start, args.layer_end, args.profiling, args.prompt)
    else:
        if args.layer_start is None or args.layer_end is None:
            print("worker precisa de --layer-start e --layer-end")
            return
        if not args.control_to:
            print("worker precisa de --control-to addr:port do head")
            return
        if args.layer_end < n_total and not args.connect:
            print("worker intermediario precisa de --connect addr:port do proximo")
            return
        if args.layer_end == n_total and not args.token_back_to:
            print("ultimo worker precisa de --token-back-to addr:port do head")
            return
        node_id = args.node_id or f"worker-{args.layer_start}-{args.layer_end}"
        run_worker(args.model, args.layer_start, args.layer_end, n_total, args.device,
                   t, args.max_new_tokens, args.port,
                   _parse(args.connect), _parse(args.token_back_to),
                   _parse(args.control_to), node_id, args.fake_error, args.profiling)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("FATAL: %s", exc)
        sys.exit(1)
