"""Safe token sweeps for tensor-parallel benchmarks."""

from __future__ import annotations

import os


def parse_token_sweep(specification: str | None,
                      default_tokens: int) -> tuple[int, ...]:
    if not specification:
        return (default_tokens,)
    try:
        tokens = tuple(int(value.strip()) for value in specification.split(","))
    except ValueError as exc:
        raise ValueError(
            f"token_sweep={specification!r}; esperado inteiros separados por vírgula"
        ) from exc
    if not tokens or any(value < 1 for value in tokens):
        raise ValueError(f"token_sweep={tokens}; esperado valores positivos")
    if any(current >= following for current, following in zip(tokens, tokens[1:])):
        raise ValueError(f"token_sweep={tokens}; esperado ordem estritamente crescente")
    return tokens


def estimate_rank_bytes(tokens: int, hidden_size: int, intermediate_size: int,
                        nodes: int, element_size: int, rank: int) -> int:
    padded_intermediate = ((intermediate_size + nodes - 1) // nodes) * nodes
    shard_intermediate = padded_intermediate // nodes
    local_weights = 3 * hidden_size * shard_intermediate
    baseline_weights = 3 * hidden_size * padded_intermediate if rank == 0 else 0
    activation_width = 4 * hidden_size + 6 * padded_intermediate
    if rank != 0:
        activation_width = 4 * hidden_size + 3 * shard_intermediate
    estimated = (local_weights + baseline_weights + tokens * activation_width) * element_size
    return int(estimated * 1.5)


def available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(pages * os.sysconf("SC_PAGE_SIZE"))
