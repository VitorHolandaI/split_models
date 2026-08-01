"""Measure how much context each configuration can hold, and at what cost.

Weights are fixed; the KV cache is not. It grows linearly with context and is
allocated per device, following the layers, so a fleet holds more context the
same way it holds more weight. Context is the axis where capacity keeps
growing, which makes it the honest case for splitting -- unlike raw speed,
where `docs/heterogeneous-split.md` shows the fast machine alone wins.

The number that matters here is not tokens per second. It is the largest
context that loads at all, and whether splitting moves it.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from .master import LLAMA_CLI

# Sizes double so the failure point is bracketed in few runs; each run costs a
# full model load, which dominates the wall clock.
#
# Para em 131072 porque o teto util e o do modelo, nao o da frota: os Qwen
# atuais treinam ate 200k-260k, e 100k ja cobre praticamente todo uso real.
# Pedir acima de n_ctx_train degrada a qualidade sem medir nada novo.
DEFAULT_CONTEXT_SIZES = (4096, 8192, 16384, 32768, 65536, 131072)
_RATES = re.compile(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s")
_OUT_OF_MEMORY = ("failed to allocate", "ErrorOutOfDeviceMemory",
                  "failed to create context", "out of memory")


@dataclass(frozen=True)
class ContextResult:
    """One context size on one configuration."""

    context: int
    endpoints: str
    loaded: bool
    prompt_rate: float
    generation_rate: float
    detail: str

    @property
    def configuration(self) -> str:
        return self.endpoints or "local"


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, context: int,
                   element_bytes: int = 2) -> int:
    """Bytes the KV cache needs for one sequence at `context` tokens.

    Two tensors per layer, K and V, hence the factor of two.

    Usage example: `kv_cache_bytes(36, 2, 128, 131072)`.
    """
    if min(layers, kv_heads, head_dim, context, element_bytes) <= 0:
        raise ValueError(
            f"layers={layers} kv_heads={kv_heads} head_dim={head_dim} "
            f"context={context} element_bytes={element_bytes}; esperado positivos")
    return layers * kv_heads * head_dim * 2 * element_bytes * context


def sweep_command(model_path: str, context: int, endpoints: str = "",
                  prompt: str = "resuma") -> list[str]:
    """Build one measurement run at a fixed context size.

    `--fit on` is deliberately absent: it searches by trial-loading the model,
    which reopens an RPC session per attempt and can fail to converge. A sweep
    needs each point to either load or fail quickly, not to negotiate.

    Usage example: `sweep_command("m.gguf", 8192, "10.0.0.2:50052")`.
    """
    remote = ["--rpc", endpoints, "--split-mode", "layer"] if endpoints else []
    return [LLAMA_CLI, "-m", model_path, *remote, "-ngl", "99",
            "-c", str(context), "-st", "-p", prompt, "-n", "16", "--seed", "42"]


def parse_rates(output: str) -> tuple[float, float]:
    """Read the prompt and generation rates llama-cli prints when it finishes.

    Usage example: `parse_rates("[ Prompt: 60.8 t/s | Generation: 42.6 t/s ]")`.
    """
    found = _RATES.search(output)
    if found is None:
        return 0.0, 0.0
    return float(found.group(1)), float(found.group(2))


def _failure_detail(output: str, returncode: int) -> str:
    for marker in _OUT_OF_MEMORY:
        if marker in output:
            return f"sem memoria: {marker}"
    return f"saiu com codigo {returncode}"


def measure_context(model_path: str, context: int, endpoints: str = "",
                    timeout: float = 900.0) -> ContextResult:
    """Run one context size and report whether it loaded and how fast.

    Usage example: `measure_context("m.gguf", 8192, "10.0.0.2:50052")`.
    """
    command = sweep_command(model_path, context, endpoints)
    try:
        finished = subprocess.run(command, check=False, capture_output=True,
                                  text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return ContextResult(context, endpoints, False, 0.0, 0.0,
                             f"nao terminou em {timeout:.0f}s")
    output = finished.stdout + finished.stderr
    prompt_rate, generation_rate = parse_rates(output)
    if generation_rate <= 0.0:
        return ContextResult(context, endpoints, False, 0.0, 0.0,
                             _failure_detail(output, finished.returncode))
    return ContextResult(context, endpoints, True, prompt_rate, generation_rate, "ok")


# Stops at the first failure instead of running the rest: once a context size
# does not fit, no larger one will, and each attempt costs a full model load.
def run_context_sweep(model_path: str, endpoints: str = "",
                      sizes: Sequence[int] = DEFAULT_CONTEXT_SIZES,
                      timeout: float = 900.0,
                      max_context: int = 0) -> tuple[ContextResult, ...]:
    """Grow the context until it stops loading.

    `max_context` corta a varredura no limite do modelo. Acima de n_ctx_train o
    modelo responde pior sem que a medicao diga nada novo, entao medir ali gasta
    uma carga inteira para produzir um numero enganoso.

    Usage example: `results = run_context_sweep("m.gguf", "10.0.0.2:50052")`.
    """
    collected: list[ContextResult] = []
    for context in sizes:
        if max_context and context > max_context:
            break
        result = measure_context(model_path, context, endpoints, timeout)
        collected.append(result)
        if not result.loaded:
            break
    return tuple(collected)


def largest_loaded(results: Sequence[ContextResult]) -> int:
    """Largest context that loaded, which is the number the sweep exists for.

    Usage example: `largest_loaded(run_context_sweep("m.gguf"))`.
    """
    loaded = [result.context for result in results if result.loaded]
    return max(loaded) if loaded else 0


def format_sweep(results: Sequence[ContextResult]) -> str:
    """Render the sweep as a table for the operator.

    Usage example: `print(format_sweep(results))`.
    """
    if not results:
        return "nenhuma medicao"
    header = f"{'CONTEXTO':>9}  {'CARREGOU':<9} {'PROMPT t/s':>11} {'GERACAO t/s':>12}  DETALHE"
    rows = [
        f"{result.context:>9}  {'sim' if result.loaded else 'nao':<9} "
        f"{result.prompt_rate:>11.1f} {result.generation_rate:>12.1f}  {result.detail}"
        for result in results
    ]
    return "\n".join([header, *rows,
                      f"\nmaior contexto que carregou: {largest_loaded(results)}"])
