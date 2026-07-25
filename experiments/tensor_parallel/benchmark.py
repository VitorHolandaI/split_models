"""Benchmark local column/row tensor parallelism with torch.distributed.

Usage example:
    uv run python -m experiments.tensor_parallel.benchmark --nodes 2 --tokens 1
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as functional


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
    modes: tuple[str, ...]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--input-features", type=int, default=896)
    parser.add_argument("--output-features", type=int, default=4864)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--mode", choices=["column", "row", "mlp", "both", "all"],
                        default="mlp")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--interface", default=None)
    return parser.parse_args(argv)


def _config(args: argparse.Namespace, local_spawn: bool) -> BenchmarkConfig:
    modes = ("column", "row") if args.mode == "both" else (args.mode,)
    if args.mode == "all":
        modes = ("column", "row", "mlp")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.nodes < 2:
        raise ValueError(f"nodes={args.nodes}; esperado pelo menos 2")
    worker_threads = max(1, args.threads // args.nodes) if local_spawn else args.threads
    return BenchmarkConfig(args.nodes, args.tokens, args.input_features,
                           args.output_features, args.iterations, args.warmup,
                           args.threads, worker_threads, dtype, modes)


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


def _report(mode: str, baseline: float, distributed: float, compute: float,
            error: float, relative_error: float, minimum_bytes: int) -> None:
    speedup = baseline / distributed if distributed else 0.0
    communication = max(0.0, distributed - compute)
    print(
        f"TENSOR_PARALLEL mode={mode} baseline_ms={baseline * 1000:.3f} "
        f"distributed_ms={distributed * 1000:.3f} speedup={speedup:.3f}x"
    )
    print(
        f"TENSOR_PARALLEL mode={mode} compute_ms={compute * 1000:.3f} "
        f"communication_ms={communication * 1000:.3f} min_network={minimum_bytes}B "
        f"max_error={error:.6g} relative_error={relative_error:.6g}"
    )


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


def _column_benchmark(config: BenchmarkConfig, rank: int) -> None:
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
        _report("column", baseline_seconds, distributed, max_compute, error,
                relative_error, network)


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


def _row_benchmark(config: BenchmarkConfig, rank: int) -> None:
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
        _report("row", baseline_seconds, distributed, max_compute, error,
                relative_error, network)


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


def _mlp_step(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
              down: torch.Tensor) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    partial = _full_mlp(inputs, gate, up, down)
    compute = time.perf_counter() - start
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    return partial, compute


def _mlp_benchmark(config: BenchmarkConfig, rank: int) -> None:
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
        _mlp_step(inputs, local_gate, local_up, local_down)
    dist.barrier(); start = time.perf_counter(); compute = 0.0; actual = inputs
    for _ in range(config.iterations):
        actual, seconds = _mlp_step(inputs, local_gate, local_up, local_down)
        compute += seconds
    dist.barrier(); distributed = (time.perf_counter() - start) / config.iterations
    max_compute = _max_compute_seconds(compute / config.iterations, rank)
    if rank == 0 and baseline is not None:
        error, relative_error = _errors(actual, baseline)
        network = _mlp_minimum_network_bytes(
            config.tokens, config.input_features, actual.element_size(), config.nodes
        )
        _report("mlp", baseline_seconds, distributed, max_compute, error,
                relative_error, network)


def _mlp_minimum_network_bytes(tokens: int, hidden_size: int,
                               element_size: int, nodes: int) -> int:
    return 2 * (nodes - 1) * tokens * hidden_size * element_size


def _run_mode(mode: str, config: BenchmarkConfig, rank: int) -> None:
    if mode == "column":
        _column_benchmark(config, rank)
    elif mode == "row":
        _row_benchmark(config, rank)
    else:
        _mlp_benchmark(config, rank)


def _distributed_worker(rank: int, config: BenchmarkConfig, init_method: str,
                        interface: str | None) -> None:
    torch.set_num_threads(config.worker_threads)
    if interface:
        os.environ["GLOO_SOCKET_IFNAME"] = interface
    dist.init_process_group("gloo", init_method=init_method,
                            rank=rank, world_size=config.nodes,
                            timeout=datetime.timedelta(seconds=120))
    try:
        for mode in config.modes:
            _run_mode(mode, config, rank)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _print_config(config: BenchmarkConfig, rank: int | None) -> None:
    if rank not in (None, 0):
        return
    print(
        f"CONFIG nodes={config.nodes} tokens={config.tokens} "
        f"matrix={config.input_features}x{config.output_features} "
        f"baseline_threads={config.baseline_threads} "
        f"worker_threads={config.worker_threads} dtype={config.dtype}"
    )


def main() -> None:
    args = _parse_args()
    local_spawn = args.rank is None
    config = _config(args, local_spawn)
    _print_config(config, args.rank)
    if local_spawn:
        init_method = f"tcp://127.0.0.1:{_free_port()}"
        mp.spawn(_distributed_worker, args=(config, init_method, args.interface),
                 nprocs=config.nodes, join=True)
        return
    if not 0 <= args.rank < args.nodes:
        raise ValueError(f"rank={args.rank}; esperado entre 0 e {args.nodes - 1}")
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    _distributed_worker(args.rank, config, init_method, args.interface)


if __name__ == "__main__":
    main()
