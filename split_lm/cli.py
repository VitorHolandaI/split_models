import argparse, logging, socket, sys, warnings
from collections.abc import Sequence

from transformers import AutoConfig, AutoTokenizer

from .context_benchmark import parse_context_sizes
from .benchmark_profiles import RUNTIME_PRECISIONS
from .constants import DEFAULT_CHAT_MODEL
from .errors import ClusterError
from .model.loader import _check_supported_config
from .node.head import run_head
from .node.worker import run_worker
from .io.network import _probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("split_lm")


def _run_cli(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="split-lm",
        description="Executa um processo interno head, worker ou probe.",
        epilog="Operadores devem preferir `split-cluster exemplos`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "role", choices=["head", "worker", "probe"],
        help="papel interno iniciado pelo orquestrador")
    ap.add_argument(
        "--model", default=DEFAULT_CHAT_MODEL,
        help="repositorio Hugging Face usado igualmente em todos os nos")
    ap.add_argument(
        "--layer-start", type=int, default=None,
        help="inicio inclusivo do intervalo global de camadas")
    ap.add_argument(
        "--layer-end", type=int, default=None,
        help="fim exclusivo do intervalo global de camadas")
    ap.add_argument(
        "--total-layers", type=int, default=None,
        help="numero de camadas do modelo completo")
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
    ap.add_argument("--device", default="cpu", help="dispositivo PyTorch local")
    ap.add_argument(
        "--max-new-tokens", type=int, default=20,
        help="limite de tokens novos por resposta")
    ap.add_argument(
        "--prompt", default="Write a short poem about distributed computers.",
        help="prompt unico para execucao nao interativa")
    ap.add_argument("--profiling", action="store_true", help="mede tempos e imprime resumo no fim da request")
    ap.add_argument("--interactive", action="store_true", help="abrir chat no head")
    ap.add_argument("--verbose", action="store_true", help="mostra logs INFO tambem no head")
    ap.add_argument("--context-sizes", default=None,
                    help="head benchmark: contextos crescentes, ex 128,256,512")
    ap.add_argument(
        "--metrics-file", default="benchmark-results/generation.jsonl",
        help="arquivo JSONL de metricas")
    ap.add_argument(
        "--benchmark-output", default="benchmark-results/context-sweep",
        help="prefixo para CSV e SVG")
    ap.add_argument(
        "--precision", choices=RUNTIME_PRECISIONS, default="auto",
        help="precisao dos pesos locais")
    ap.add_argument(
        "--warmup-runs", type=int, default=0,
        help="geracoes descartadas antes da medicao")
    args = ap.parse_args(argv)

    if args.profiling and args.interactive:
        raise ValueError("--profiling mede uma request; remova --interactive")

    def _parse(s):
        if not s: return None
        p = s.split(":"); return (p[0], int(p[1]))

    if args.role == "probe":
        if not args.probe_to:
            raise ValueError("probe precisa de --probe-to addr:port")
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
            raise ValueError("head precisa de --connect addr:port do primeiro worker")
        run_head(args.model, n_total, args.device, t, args.max_new_tokens,
                 args.interactive, _parse(args.connect), args.port,
                 args.control_port, args.workers, args.ready_timeout,
                 args.layer_start, args.layer_end, args.profiling, args.prompt,
                 args.metrics_file, context_sizes, args.benchmark_output,
                 args.precision, args.warmup_runs)
    else:
        if args.layer_start is None or args.layer_end is None:
            raise ValueError("worker precisa de --layer-start e --layer-end")
        if not args.control_to:
            raise ValueError("worker precisa de --control-to addr:port do head")
        if args.layer_end < n_total and not args.connect:
            raise ValueError("worker intermediario precisa de --connect addr:port do proximo")
        if args.layer_end == n_total and not args.token_back_to:
            raise ValueError("ultimo worker precisa de --token-back-to addr:port do head")
        node_id = args.node_id or f"worker-{args.layer_start}-{args.layer_end}"
        run_worker(args.model, args.layer_start, args.layer_end, n_total, args.device,
                   t, args.max_new_tokens, args.port,
                   _parse(args.connect), _parse(args.token_back_to),
                   _parse(args.control_to), node_id, args.fake_error, args.profiling,
                   args.precision)

def _cli_origin(argv: Sequence[str] | None) -> str:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    return next((role for role in ("head", "worker", "probe") if role in arguments),
                "split-lm")


def main(argv: Sequence[str] | None = None) -> None:
    """Run a head, worker, or probe and classify local failures."""
    try:
        _run_cli(argv)
    except ClusterError:
        raise
    except Exception as exc:
        origin = _cli_origin(argv)
        raise ClusterError.from_exception(
            origin, "local", socket.gethostname(), "local",
            f"{origin}.cli", exc) from exc


if __name__ == "__main__":
    main()
