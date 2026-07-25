"""Result formatting and exports for tensor-parallel benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    tokens: int
    baseline_ms: float
    distributed_ms: float
    compute_ms: float
    communication_ms: float
    speedup: float
    bytes_per_iteration: float
    effective_network_mbps: float
    minimum_network_bytes: int
    max_error: float
    relative_error: float
    matmuls_per_rank: int
    collective_calls: int
    sent_frames: int
    sent_bytes: int


def effective_network_mbps(bytes_per_iteration: float,
                           communication_seconds: float) -> float:
    if communication_seconds <= 0:
        return 0.0
    return bytes_per_iteration * 8.0 / communication_seconds / 1_000_000.0


def build_result(mode: str, tokens: int, baseline_seconds: float,
                 distributed_seconds: float, compute_seconds: float,
                 minimum_bytes: int, max_error: float, relative_error: float,
                 matmuls_per_rank: int, collective_calls: int,
                 sent_frames: int, sent_bytes: int) -> BenchmarkResult:
    communication = max(0.0, distributed_seconds - compute_seconds)
    speedup = baseline_seconds / distributed_seconds if distributed_seconds else 0.0
    measured_bytes = sent_bytes / collective_calls if sent_frames else minimum_bytes
    network_mbps = effective_network_mbps(measured_bytes, communication)
    return BenchmarkResult(
        mode, tokens, baseline_seconds * 1000, distributed_seconds * 1000,
        compute_seconds * 1000, communication * 1000, speedup, measured_bytes,
        network_mbps, minimum_bytes, max_error, relative_error,
        matmuls_per_rank, collective_calls, sent_frames, sent_bytes,
    )


def _color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def result_lines(result: BenchmarkResult, color: bool = False) -> list[str]:
    speed_color = "32" if result.speedup >= 1 else "31"
    frames = str(result.sent_frames) if result.sent_frames else "gerenciado pelo Gloo"
    sent_bytes = str(result.sent_bytes) if result.sent_frames else "gerenciado pelo Gloo"
    return [
        _color(f"RESULTADO mode={result.mode} tokens={result.tokens}", "1;36", color),
        f"baseline_ms={result.baseline_ms:.3f} | tempo local do MLP completo; menor é melhor",
        f"distributed_ms={result.distributed_ms:.3f} | compute dividido + comunicação; menor é melhor",
        _color(f"speedup={result.speedup:.3f}x | baseline_ms ÷ distributed_ms; acima de 1.0x ganhou", speed_color, color),
        f"compute_ms={result.compute_ms:.3f} | três matmuls no rank mais lento: gate, up e down",
        f"communication_ms={result.communication_ms:.3f} | troca dos parciais + sincronização do collective",
        f"effective_network_mbps={result.effective_network_mbps:.1f} | throughput agregado da aplicação",
        f"network_formula=({result.bytes_per_iteration:.0f} bytes × 8) ÷ ({result.communication_ms / 1000:.6f}s × 1.000.000)",
        f"minimum_network_bytes={result.minimum_network_bytes} | mínimo teórico por collective, sem headers",
        f"max_error={result.max_error:.6g} | maior diferença absoluta contra o resultado local",
        f"relative_error={result.relative_error:.6g} | max_error normalizado pelo maior valor local",
        f"matmuls_per_rank={result.matmuls_per_rank} | multiplicações executadas por rank e iteração",
        f"collective_calls={result.collective_calls} | combinações de resultados medidas, sem warmup",
        f"sent_frames={frames} | frames enviados por todos os ranks, sem warmup",
        f"sent_bytes={sent_bytes} | bytes reais enviados por todos os ranks, sem warmup",
    ]


def print_result(result: BenchmarkResult, color: bool = False) -> None:
    print("\n".join(result_lines(result, color)))


def write_csv(results: Sequence[BenchmarkResult], output_prefix: str) -> Path:
    path = Path(output_prefix).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=BenchmarkResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)
    return path


def write_png(results: Sequence[BenchmarkResult], output_prefix: str) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as exc:
        raise RuntimeError("matplotlib ausente; execute uv sync --extra plots") from exc
    path = Path(output_prefix).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = [result.tokens for result in results]
    figure, axis = pyplot.subplots(figsize=(9, 5))
    axis.plot(tokens, [result.baseline_ms for result in results], "o-", label="Local")
    axis.plot(tokens, [result.distributed_ms for result in results], "o-", label="Distribuído")
    winners = [result.tokens for result in results if result.speedup >= 1]
    if winners:
        axis.axvline(winners[0], color="green", linestyle="--", label=f"Threshold: {winners[0]} tokens")
    axis.set(xlabel="Tokens processados juntos", ylabel="Tempo por iteração (ms)",
             title="MLP local vs tensor parallel")
    axis.set_xscale("log", base=2); axis.grid(True, alpha=0.3); axis.legend()
    figure.tight_layout(); figure.savefig(path, dpi=160); pyplot.close(figure)
    return path


def write_sweep_outputs(results: Sequence[BenchmarkResult],
                        output_prefix: str) -> tuple[Path, Path]:
    if not results:
        raise ValueError("results vazio; esperado pelo menos uma medição")
    return write_csv(results, output_prefix), write_png(results, output_prefix)
