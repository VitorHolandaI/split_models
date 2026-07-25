"""Benchmark tensor parallelism with torch.distributed.
Example: `uv run python -m experiments.tensor_parallel.benchmark --nodes 2`.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as functional

from split_lm.io.network import _configure_connected_socket, _connect, _tcp_socket
from split_lm.io.protocol import NetworkStats, _recv, _send
from .results import BenchmarkResult, build_result, print_result, write_sweep_outputs
from .sweep import available_memory_bytes, estimate_rank_bytes, parse_token_sweep


@dataclass(frozen=True)
class BenchmarkConfig:
    nodes: int
    tokens: int
    input_features: int
    output_features: int
    iterations: int
    warmup: int
    baseline_threads: int
    worker_threads: int
    dtype: torch.dtype
    wire_dtype: torch.dtype | None
    modes: tuple[str, ...]
    collective: str
    token_sweep: tuple[int, ...]
    max_memory_fraction: float
    output_prefix: str


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--input-features", type=int, default=896)
    parser.add_argument("--output-features", type=int, default=4864)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--worker-threads", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--wire-dtype", choices=["same", "bfloat16", "float16"],
                        default="same")
    parser.add_argument("--mode", choices=["column", "row", "mlp", "both", "all"],
                        default="mlp")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--collective", choices=["gloo", "tcp"], default="gloo")
    parser.add_argument("--collective-port", type=int, default=29501)
    parser.add_argument("--interface", default=None)
    parser.add_argument("--token-sweep", default=None,
                        help="tokens crescentes, exemplo: 1,2,4,8,16,32,64,128")
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--output-prefix", default="benchmark-results/tensor_parallel_sweep")
    return parser.parse_args(argv)


def _config(args: argparse.Namespace, local_spawn: bool) -> BenchmarkConfig:
    modes = ("column", "row") if args.mode == "both" else (args.mode,)
    if args.mode == "all":
        modes = ("column", "row", "mlp")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    wire_dtype = {
        "same": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.wire_dtype]
    token_sweep = parse_token_sweep(args.token_sweep, args.tokens)
    if args.nodes < 2:
        raise ValueError(f"nodes={args.nodes}; esperado pelo menos 2")
    if args.collective == "tcp" and args.nodes != 2:
        raise ValueError(f"nodes={args.nodes}; collective tcp espera exatamente 2")
    if wire_dtype is not None and args.collective != "tcp":
        raise ValueError(f"wire_dtype={args.wire_dtype!r}; esperado collective tcp")
    if not 0 < args.max_memory_fraction <= 1:
        raise ValueError(
            f"max_memory_fraction={args.max_memory_fraction}; esperado entre 0 e 1"
        )
    if len(token_sweep) > 1 and modes != ("mlp",):
        raise ValueError(f"modes={modes}; token_sweep espera apenas mode mlp")
    default_threads = max(1, args.threads // args.nodes) if local_spawn else args.threads
    worker_threads = args.worker_threads or default_threads
    if worker_threads < 1:
        raise ValueError(f"worker_threads={worker_threads}; esperado pelo menos 1")
    return BenchmarkConfig(args.nodes, args.tokens, args.input_features,
                           args.output_features, args.iterations, args.warmup,
                           args.threads, worker_threads, dtype, wire_dtype, modes,
                           args.collective, token_sweep,
                           args.max_memory_fraction, args.output_prefix)


def _free_port() -> int:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = int(server.getsockname()[1])
    server.close()
    return port


def _padded(value: int, nodes: int) -> int:
    return ((value + nodes - 1) // nodes) * nodes


def _baseline(operation: Callable[[], torch.Tensor],
              config: BenchmarkConfig) -> tuple[torch.Tensor, float]:
    torch.set_num_threads(config.baseline_threads)
    for _ in range(config.warmup):
        operation()
    start = time.perf_counter()
    result = operation()
    for _ in range(config.iterations - 1):
        result = operation()
    return result, (time.perf_counter() - start) / config.iterations


def _max_compute_seconds(seconds: float, rank: int) -> float:
    measured = torch.tensor([seconds], dtype=torch.float64)
    dist.reduce(measured, dst=0, op=dist.ReduceOp.MAX)
    return float(measured.item()) if rank == 0 else 0.0


def _report(mode: str, tokens: int, baseline: float, distributed: float,
            compute: float,
            error: float, relative_error: float, minimum_bytes: int,
            matmuls_per_rank: int, iterations: int, sent_frames: int,
            sent_bytes: int) -> BenchmarkResult:
    result = build_result(
        mode, tokens, baseline, distributed, compute, minimum_bytes, error,
        relative_error, matmuls_per_rank, iterations, sent_frames, sent_bytes,
    )
    print_result(result, color=sys.stdout.isatty())
    return result


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    absolute = float((actual - expected).abs().max())
    scale = max(float(expected.abs().max()), torch.finfo(torch.float32).tiny)
    return absolute, absolute / scale


def _column_tensors(config: BenchmarkConfig, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    padded_output = _padded(config.output_features, config.nodes)
    inputs = torch.empty((config.tokens, config.input_features), dtype=config.dtype)
    weight = torch.empty((config.input_features, padded_output), dtype=config.dtype) \
        if rank == 0 else torch.empty(0, dtype=config.dtype)
    if rank == 0:
        torch.manual_seed(7)
        inputs.normal_()
        weight.zero_()
        weight[:, :config.output_features].normal_()
    return inputs, weight


def _column_step(inputs: torch.Tensor, local_weight: torch.Tensor,
                 gathered: list[torch.Tensor] | None) -> tuple[torch.Tensor, float]:
    dist.broadcast(inputs, src=0)
    start = time.perf_counter()
    partial = inputs @ local_weight
    compute = time.perf_counter() - start
    dist.gather(partial, gather_list=gathered, dst=0)
    return partial, compute


def _column_benchmark(config: BenchmarkConfig,
                      rank: int) -> BenchmarkResult | None:
    inputs, weight = _column_tensors(config, rank)
    chunk = _padded(config.output_features, config.nodes) // config.nodes
    local_weight = torch.empty((config.input_features, chunk), dtype=config.dtype)
    shards = [shard.contiguous() for shard in weight.split(chunk, dim=1)] \
        if rank == 0 else None
    dist.scatter(local_weight, scatter_list=shards, src=0)
    gathered = [torch.empty((config.tokens, chunk), dtype=config.dtype)
                for _ in range(config.nodes)] if rank == 0 else None
    baseline, baseline_seconds = _baseline(lambda: inputs @ weight, config) \
        if rank == 0 else (None, 0.0)
    torch.set_num_threads(config.worker_threads)
    for _ in range(config.warmup):
        _column_step(inputs, local_weight, gathered)
    dist.barrier(); start = time.perf_counter(); compute = 0.0
    for _ in range(config.iterations):
        _, seconds = _column_step(inputs, local_weight, gathered); compute += seconds
    dist.barrier(); distributed = (time.perf_counter() - start) / config.iterations
    max_compute = _max_compute_seconds(compute / config.iterations, rank)
    if rank == 0 and gathered is not None and baseline is not None:
        actual = torch.cat(gathered, dim=-1)
        error, relative_error = _errors(actual, baseline)
        network = inputs.numel() * inputs.element_size() * (config.nodes - 1)
        network += actual.numel() * actual.element_size() * (config.nodes - 1) // config.nodes
        return _report("column", config.tokens, baseline_seconds, distributed,
                       max_compute, error, relative_error, network, 1,
                       config.iterations, 0, 0)
    return None


def _row_tensors(config: BenchmarkConfig, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    padded_input = _padded(config.input_features, config.nodes)
    inputs = torch.empty((config.tokens, padded_input), dtype=config.dtype) \
        if rank == 0 else torch.empty(0, dtype=config.dtype)
    weight = torch.empty((padded_input, config.output_features), dtype=config.dtype) \
        if rank == 0 else torch.empty(0, dtype=config.dtype)
    if rank == 0:
        torch.manual_seed(11)
        inputs.zero_(); weight.zero_()
        inputs[:, :config.input_features].normal_()
        weight[:config.input_features].normal_()
    return inputs, weight


def _row_step(local_input: torch.Tensor, local_weight: torch.Tensor,
              input_shards: list[torch.Tensor] | None) -> tuple[torch.Tensor, float]:
    dist.scatter(local_input, scatter_list=input_shards, src=0)
    start = time.perf_counter()
    partial = local_input @ local_weight
    compute = time.perf_counter() - start
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    return partial, compute


def _row_benchmark(config: BenchmarkConfig,
                   rank: int) -> BenchmarkResult | None:
    inputs, weight = _row_tensors(config, rank)
    chunk = _padded(config.input_features, config.nodes) // config.nodes
    local_input = torch.empty((config.tokens, chunk), dtype=config.dtype)
    local_weight = torch.empty((chunk, config.output_features), dtype=config.dtype)
    input_shards = [shard.contiguous() for shard in inputs.split(chunk, dim=1)] \
        if rank == 0 else None
    weight_shards = [shard.contiguous() for shard in weight.split(chunk, dim=0)] \
        if rank == 0 else None
    dist.scatter(local_weight, scatter_list=weight_shards, src=0)
    baseline, baseline_seconds = _baseline(lambda: inputs @ weight, config) \
        if rank == 0 else (None, 0.0)
    torch.set_num_threads(config.worker_threads)
    for _ in range(config.warmup):
        _row_step(local_input, local_weight, input_shards)
    dist.barrier(); start = time.perf_counter(); compute = 0.0; actual = local_input
    for _ in range(config.iterations):
        actual, seconds = _row_step(local_input, local_weight, input_shards); compute += seconds
    dist.barrier(); distributed = (time.perf_counter() - start) / config.iterations
    max_compute = _max_compute_seconds(compute / config.iterations, rank)
    if rank == 0 and baseline is not None:
        error, relative_error = _errors(actual, baseline)
        network = inputs.numel() * inputs.element_size() * (config.nodes - 1) // config.nodes
        network += 2 * (config.nodes - 1) * actual.numel() * actual.element_size()
        return _report("row", config.tokens, baseline_seconds, distributed,
                       max_compute, error, relative_error, network, 1,
                       config.iterations, 0, 0)
    return None


def _full_mlp(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
              down: torch.Tensor) -> torch.Tensor:
    expanded = functional.silu(inputs @ gate) * (inputs @ up)
    return expanded @ down


def _mlp_tensors(config: BenchmarkConfig, rank: int) -> tuple[torch.Tensor, ...]:
    intermediate = _padded(config.output_features, config.nodes)
    inputs = torch.empty((config.tokens, config.input_features), dtype=config.dtype)
    if rank != 0:
        empty = torch.empty(0, dtype=config.dtype)
        return inputs, empty, empty, empty
    torch.manual_seed(17)
    inputs.normal_()
    gate = torch.zeros((config.input_features, intermediate), dtype=config.dtype)
    up = torch.zeros_like(gate)
    down = torch.zeros((intermediate, config.input_features), dtype=config.dtype)
    gate[:, :config.output_features].normal_()
    up[:, :config.output_features].normal_()
    down[:config.output_features].normal_()
    return inputs, gate, up, down


def _scatter_mlp_weight(weight: torch.Tensor, local_weight: torch.Tensor,
                        chunk: int, dim: int, rank: int) -> None:
    shards = [shard.contiguous() for shard in weight.split(chunk, dim=dim)] \
        if rank == 0 else None
    dist.scatter(local_weight, scatter_list=shards, src=0)


def _tcp_all_reduce(partial: torch.Tensor, connection: socket.socket,
                    traffic: NetworkStats | None,
                    wire_dtype: torch.dtype | None = None) -> torch.Tensor:
    wire_partial = partial if wire_dtype is None else partial.to(wire_dtype)
    _send(connection, wire_partial, traffic)
    remote, _ = _recv(connection, "cpu", traffic)
    if wire_dtype is None:
        partial.add_(remote)
        return partial
    result = wire_partial.to(partial.dtype)
    result.add_(remote.to(partial.dtype))
    return result


def _reduce_partial(partial: torch.Tensor, connection: socket.socket | None,
                    traffic: NetworkStats | None,
                    wire_dtype: torch.dtype | None) -> torch.Tensor:
    if connection is not None:
        return _tcp_all_reduce(partial, connection, traffic, wire_dtype)
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    return partial


def _mlp_step(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
              down: torch.Tensor, connection: socket.socket | None,
              traffic: NetworkStats | None = None,
              wire_dtype: torch.dtype | None = None) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    partial = _full_mlp(inputs, gate, up, down)
    compute = time.perf_counter() - start
    return _reduce_partial(partial, connection, traffic, wire_dtype), compute


def _aggregate_sent(traffic: NetworkStats | None, rank: int) -> tuple[int, int]:
    values = torch.tensor([
        traffic.sent_messages if traffic else 0,
        traffic.sent_bytes if traffic else 0,
    ], dtype=torch.int64)
    dist.reduce(values, dst=0, op=dist.ReduceOp.SUM)
    return (int(values[0]), int(values[1])) if rank == 0 else (0, 0)


def _mlp_benchmark(config: BenchmarkConfig, rank: int,
                   connection: socket.socket | None) -> BenchmarkResult | None:
    inputs, gate, up, down = _mlp_tensors(config, rank)
    chunk = _padded(config.output_features, config.nodes) // config.nodes
    local_gate = torch.empty((config.input_features, chunk), dtype=config.dtype)
    local_up = torch.empty_like(local_gate)
    local_down = torch.empty((chunk, config.input_features), dtype=config.dtype)
    _scatter_mlp_weight(gate, local_gate, chunk, 1, rank)
    _scatter_mlp_weight(up, local_up, chunk, 1, rank)
    _scatter_mlp_weight(down, local_down, chunk, 0, rank)
    dist.broadcast(inputs, src=0)
    baseline, baseline_seconds = _baseline(
        lambda: _full_mlp(inputs, gate, up, down), config
    ) if rank == 0 else (None, 0.0)
    torch.set_num_threads(config.worker_threads)
    for _ in range(config.warmup):
        _mlp_step(inputs, local_gate, local_up, local_down, connection,
                  wire_dtype=config.wire_dtype)
    traffic = NetworkStats() if connection is not None else None
    dist.barrier(); start = time.perf_counter(); compute = 0.0; actual = inputs
    for _ in range(config.iterations):
        actual, seconds = _mlp_step(
            inputs, local_gate, local_up, local_down, connection, traffic,
            config.wire_dtype
        )
        compute += seconds
    dist.barrier(); distributed = (time.perf_counter() - start) / config.iterations
    max_compute = _max_compute_seconds(compute / config.iterations, rank)
    sent_frames, sent_bytes = _aggregate_sent(traffic, rank)
    if rank == 0 and baseline is not None:
        error, relative_error = _errors(actual, baseline)
        wire_type = config.wire_dtype or actual.dtype
        wire_size = torch.empty((), dtype=wire_type).element_size()
        network = _mlp_minimum_network_bytes(
            config.tokens, config.input_features, wire_size, config.nodes
        )
        return _report(
            f"mlp-{config.collective}", config.tokens, baseline_seconds,
            distributed, max_compute, error, relative_error, network, 3,
            config.iterations, sent_frames, sent_bytes,
        )
    return None


def _mlp_minimum_network_bytes(tokens: int, hidden_size: int,
                               element_size: int, nodes: int) -> int:
    return 2 * (nodes - 1) * tokens * hidden_size * element_size


def _run_mode(mode: str, config: BenchmarkConfig, rank: int,
              connection: socket.socket | None) -> BenchmarkResult | None:
    if mode == "column":
        return _column_benchmark(config, rank)
    if mode == "row":
        return _row_benchmark(config, rank)
    return _mlp_benchmark(config, rank, connection)


def _open_tcp_collective(rank: int, master_addr: str,
                         collective_port: int) -> socket.socket:
    if rank == 1:
        return _connect((master_addr, collective_port), "tcp collective")
    server = _tcp_socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", collective_port))
    server.listen(1)
    connection = _configure_connected_socket(server.accept()[0])
    server.close()
    return connection


def _sweep_point_safe(config: BenchmarkConfig, rank: int) -> bool:
    element_size = torch.empty((), dtype=config.dtype).element_size()
    estimated = estimate_rank_bytes(
        config.tokens, config.input_features, config.output_features,
        config.nodes, element_size, rank,
    )
    available = available_memory_bytes()
    safe = int(estimated <= available * config.max_memory_fraction)
    safe_tensor = torch.tensor([safe], dtype=torch.int64)
    estimate_tensor = torch.tensor([estimated], dtype=torch.int64)
    available_tensor = torch.tensor([available], dtype=torch.int64)
    dist.all_reduce(safe_tensor, op=dist.ReduceOp.MIN)
    dist.all_reduce(estimate_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(available_tensor, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"SWEEP_RAM tokens={config.tokens} max_estimated_mib={int(estimate_tensor[0]) / 2**20:.1f} "
            f"min_available_mib={int(available_tensor[0]) / 2**20:.1f} "
            f"limit={config.max_memory_fraction:.0%} safe={bool(safe_tensor[0])}"
        )
    return bool(safe_tensor[0])


def _distributed_worker(rank: int, config: BenchmarkConfig, init_method: str,
                        interface: str | None, master_addr: str,
                        collective_port: int) -> None:
    torch.set_num_threads(config.worker_threads)
    if interface:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
    dist.init_process_group("gloo", init_method=init_method,
                            rank=rank, world_size=config.nodes,
                            timeout=datetime.timedelta(seconds=120))
    connection = _open_tcp_collective(rank, master_addr, collective_port) \
        if config.collective == "tcp" else None
    results: list[BenchmarkResult] = []
    try:
        for tokens in config.token_sweep:
            point_config = replace(config, tokens=tokens)
            if not _sweep_point_safe(point_config, rank):
                break
            for mode in point_config.modes:
                result = _run_mode(mode, point_config, rank, connection)
                if result is not None:
                    results.append(result)
                dist.barrier()
            gc.collect()
        if rank == 0 and len(config.token_sweep) > 1 and results:
            csv_path, png_path, speedup_png = write_sweep_outputs(
                results, config.output_prefix
            )
            print("\nSWEEP CONCLUÍDO\n")
            print(f"csv={csv_path}\n")
            print(f"tempos_png={png_path}\n")
            print(f"speedup_png={speedup_png}")
    finally:
        if connection is not None:
            connection.close()
        dist.destroy_process_group()


def _print_config(config: BenchmarkConfig, rank: int | None) -> None:
    if rank not in (None, 0):
        return
    print(
        f"CONFIG nodes={config.nodes} tokens={config.tokens} "
        f"matrix={config.input_features}x{config.output_features} "
        f"baseline_threads={config.baseline_threads} "
        f"worker_threads={config.worker_threads} dtype={config.dtype} "
        f"wire_dtype={config.wire_dtype or config.dtype} "
        f"collective={config.collective} token_sweep={config.token_sweep}"
    )


def main() -> None:
    args = _parse_args()
    local_spawn = args.rank is None
    config = _config(args, local_spawn)
    _print_config(config, args.rank)
    if local_spawn:
        init_method = f"tcp://127.0.0.1:{_free_port()}"
        collective_port = _free_port()
        mp.spawn(_distributed_worker,
                 args=(config, init_method, args.interface, "127.0.0.1",
                       collective_port),
                 nprocs=config.nodes, join=True)
        return
    if not 0 <= args.rank < args.nodes:
        raise ValueError(f"rank={args.rank}; esperado entre 0 e {args.nodes - 1}")
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    _distributed_worker(args.rank, config, init_method, args.interface,
                        args.master_addr, args.collective_port)


if __name__ == "__main__":
    main()
