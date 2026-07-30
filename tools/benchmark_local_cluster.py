"""Run a local multi-process split-LLM benchmark.

Usage example:
    uv run python tools/benchmark_local_cluster.py --nodes 3 --max-new-tokens 10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence

from transformers import AutoConfig, PretrainedConfig

from split_lm.constants import DEFAULT_CHAT_MODEL
from split_lm.model.loader import _check_supported_config


MIB = 1024 * 1024


def split_layers(total_layers: int, nodes: int) -> list[tuple[int, int]]:
    """Return contiguous balanced layer ranges.

    Usage example:
        ranges = split_layers(24, 3)
    """
    if nodes < 2 or nodes > total_layers:
        raise ValueError(f"nodes={nodes}; esperado entre 2 e {total_layers}")
    base, remainder = divmod(total_layers, nodes)
    ranges: list[tuple[int, int]] = []
    start = 0
    for node in range(nodes):
        end = start + base + (node < remainder)
        ranges.append((start, end))
        start = end
    return ranges


def _layer_parameter_count(config: PretrainedConfig) -> int:
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    intermediate = int(config.intermediate_size)
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    attention = hidden * (query_width + 2 * kv_width) + query_width * hidden
    if bool(getattr(config, "attention_bias", False)):
        attention += query_width + 2 * kv_width
    return attention + 3 * hidden * intermediate + 2 * hidden


def estimate_node_bytes(config: PretrainedConfig, nodes: int,
                        bytes_per_parameter: int) -> list[int]:
    """Estimate model-weight bytes held by each local node.

    Usage example:
        sizes = estimate_node_bytes(config, nodes=2, bytes_per_parameter=4)
    """
    ranges = split_layers(int(config.num_hidden_layers), nodes)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    layer_parameters = _layer_parameter_count(config)
    estimates = [(end - start) * layer_parameters for start, end in ranges]
    estimates[0] += vocab * hidden
    estimates[-1] += hidden + vocab * hidden
    return [parameters * bytes_per_parameter for parameters in estimates]


def available_memory_bytes() -> int:
    """Return Linux MemAvailable, with a sysconf fallback.

    Usage example:
        available = available_memory_bytes()
    """
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(pages * os.sysconf("SC_PAGE_SIZE"))


def _format_bytes(value: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--nodes", type=int, default=2, help="head + workers")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--prompt", default="Write a poem about distributed computers.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-port", type=int, default=19000)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--ram-fraction", type=float, default=0.8)
    parser.add_argument("--process-overhead-mib", type=int, default=256)
    parser.add_argument("--bytes-per-parameter", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _common_command(args: argparse.Namespace, total_layers: int) -> list[str]:
    return [
        sys.executable, "-m", "split_lm", "--model", args.model,
        "--total-layers", str(total_layers), "--device", args.device,
        "--max-new-tokens", str(args.max_new_tokens), "--profiling",
    ]


def _head_command(args: argparse.Namespace, ranges: list[tuple[int, int]]) -> list[str]:
    total_layers = ranges[-1][1]
    token_port = args.base_port + args.nodes
    control_port = token_port + 1
    start, end = ranges[0]
    return _common_command(args, total_layers) + [
        "head", "--layer-start", str(start), "--layer-end", str(end),
        "--connect", f"127.0.0.1:{args.base_port + 1}",
        "--port", str(token_port), "--control-port", str(control_port),
        "--workers", str(args.nodes - 1), "--ready-timeout", str(args.timeout),
        "--prompt", args.prompt,
    ]


def _worker_command(args: argparse.Namespace, ranges: list[tuple[int, int]],
                    node: int) -> list[str]:
    total_layers = ranges[-1][1]
    token_port = args.base_port + args.nodes
    control_port = token_port + 1
    start, end = ranges[node]
    command = _common_command(args, total_layers) + [
        "worker", "--layer-start", str(start), "--layer-end", str(end),
        "--port", str(args.base_port + node),
        "--control-to", f"127.0.0.1:{control_port}",
        "--node-id", f"local-{node}",
    ]
    target = "--token-back-to" if node == args.nodes - 1 else "--connect"
    port = token_port if node == args.nodes - 1 else args.base_port + node + 1
    return command + [target, f"127.0.0.1:{port}"]


def _print_plan(args: argparse.Namespace, ranges: list[tuple[int, int]],
                node_bytes: list[int], available: int) -> None:
    overhead = args.process_overhead_mib * MIB
    estimated = sum(node_bytes) + args.nodes * overhead
    print(f"MODEL {args.model} nodes={args.nodes} layers={ranges[-1][1]}")
    for node, ((start, end), model_bytes) in enumerate(zip(ranges, node_bytes)):
        role = "head" if node == 0 else "tail" if node == args.nodes - 1 else "worker"
        print(f"NODE {node} {role}: layers {start}..{end - 1}, model≈{_format_bytes(model_bytes)}")
    print(f"RAM available={_format_bytes(available)} estimated={_format_bytes(estimated)}")
    print("NOTE local processes share CPU and memory bandwidth; this is not a network speedup benchmark",
          flush=True)


def _check_memory(args: argparse.Namespace, node_bytes: list[int], available: int) -> None:
    overhead = args.nodes * args.process_overhead_mib * MIB
    estimated = sum(node_bytes) + overhead
    budget = available * args.ram_fraction
    if estimated <= budget or args.force:
        return
    raise MemoryError(
        f"estimativa {_format_bytes(estimated)} excede orçamento {_format_bytes(budget)}; "
        "reduza --nodes/--bytes-per-parameter ou use --force"
    )


def _stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _run_cluster(args: argparse.Namespace, ranges: list[tuple[int, int]]) -> int:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    head = subprocess.Popen(_head_command(args, ranges), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    labeled = [("head", head)]
    try:
        time.sleep(2)
        for node in range(args.nodes - 1, 0, -1):
            process = subprocess.Popen(_worker_command(args, ranges, node),
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, env=env)
            labeled.append((f"worker-{node}", process))
        deadline = time.monotonic() + args.timeout
        outputs = []
        for label, process in labeled:
            stdout, stderr = process.communicate(timeout=max(1, deadline - time.monotonic()))
            outputs.append((label, process.returncode, stdout, stderr))
    except (Exception, KeyboardInterrupt):
        _stop_processes([process for _, process in labeled])
        raise
    for label, returncode, stdout, stderr in outputs:
        print(f"\n=== {label} exit={returncode} ===")
        if stdout.strip(): print(stdout.strip())
        if stderr.strip(): print(stderr.strip())
    return int(any(returncode != 0 for _, returncode, _, _ in outputs))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = AutoConfig.from_pretrained(args.model)
    _check_supported_config(config)
    ranges = split_layers(int(config.num_hidden_layers), args.nodes)
    node_bytes = estimate_node_bytes(config, args.nodes, args.bytes_per_parameter)
    available = available_memory_bytes()
    _print_plan(args, ranges, node_bytes, available)
    _check_memory(args, node_bytes, available)
    return _run_cluster(args, ranges)


if __name__ == "__main__":
    raise SystemExit(main())
