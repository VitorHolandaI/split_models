import argparse, logging, sys, warnings

from transformers import AutoConfig, AutoTokenizer

from .context_benchmark import parse_context_sizes
from .constants import SUPPORTED_MODEL_TYPES
from .model.loader import _check_supported_config
from .node.head import run_head
from .node.worker import run_worker
from .io.network import _probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("split_lm")


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
    ap.add_argument("--context-sizes", default=None,
                    help="head benchmark: contextos crescentes, ex 128,256,512")
    ap.add_argument("--metrics-file", default="benchmark-results/generation.jsonl")
    ap.add_argument("--benchmark-output", default="benchmark-results/context-sweep")
    args = ap.parse_args()

    if args.profiling and args.interactive:
        raise ValueError("--profiling mede uma request; remova --interactive")

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

    t = AutoTokenizer.from_pretrained(args.model)
    if t.pad_token is None: t.pad_token = t.eos_token

    cfg = AutoConfig.from_pretrained(args.model)
    _check_supported_config(cfg)
    n_total = args.total_layers or cfg.num_hidden_layers
    context_sizes = ()
    if args.context_sizes:
        max_context = int(cfg.max_position_embeddings) - args.max_new_tokens
        context_sizes = parse_context_sizes(args.context_sizes, max_context)

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
                 args.layer_start, args.layer_end, args.profiling, args.prompt,
                 args.metrics_file, context_sizes, args.benchmark_output)
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
