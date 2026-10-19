"""Result formatting and exports for tensor-parallel benchmarks."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_Metric = tuple[str, str, str, str]


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


def _metric_lines(metric: _Metric, color: bool) -> list[str]:
    name, value, description, code = metric
    field = _color(name, code, color)
    displayed_value = _color(value, f"1;{code}", color)
    explanation = _color(f"  {description}", "2", color)
    return [f"{field}={displayed_value}", explanation, ""]


def _section_lines(title: str, metrics: Sequence[_Metric],
                   color: bool) -> list[str]:
    lines = [_color(f"=== {title} ===", "1;37", color), ""]
    for metric in metrics:
        lines.extend(_metric_lines(metric, color))
    return lines


def _performance_metrics(result: BenchmarkResult) -> tuple[_Metric, ...]:
    speed_color = "32" if result.speedup >= 1 else "31"
    return (
        ("baseline_ms", f"{result.baseline_ms:.3f} ms",
         "Tempo local do MLP completo; menor é melhor.", "36"),
        ("distributed_ms", f"{result.distributed_ms:.3f} ms",
         "Compute dividido mais comunicação; menor é melhor.", "35"),
        ("speedup", f"{result.speedup:.3f}x",
         "baseline_ms ÷ distributed_ms; acima de 1.0x ganhou.", speed_color),
    )


def _network_metrics(result: BenchmarkResult) -> tuple[_Metric, ...]:
    formula = (f"({result.bytes_per_iteration:.0f} bytes × 8) ÷ "
               f"({result.communication_ms / 1000:.6f}s × 1.000.000)")
    return (
        ("compute_ms", f"{result.compute_ms:.3f} ms",
         "Três matmuls no rank mais lento: gate, up e down.", "34"),
        ("communication_ms", f"{result.communication_ms:.3f} ms",
         "Troca dos parciais mais sincronização do collective.", "33"),
        ("effective_network_mbps", f"{result.effective_network_mbps:.1f} Mbit/s",
         "Throughput agregado medido pela aplicação.", "36"),
        ("network_formula", formula,
         "Fórmula usada para calcular o throughput efetivo.", "34"),
        ("minimum_network_bytes", str(result.minimum_network_bytes),
         "Mínimo teórico por collective, sem headers.", "35"),
    )


def _accuracy_metrics(result: BenchmarkResult) -> tuple[_Metric, ...]:
    return (
        ("max_error", f"{result.max_error:.6g}",
         "Maior diferença absoluta contra o resultado local.", "31"),
        ("relative_error", f"{result.relative_error:.6g}",
         "max_error normalizado pelo maior valor local.", "33"),
    )


def _counter_metrics(result: BenchmarkResult) -> tuple[_Metric, ...]:
    frames = str(result.sent_frames) if result.sent_frames else "gerenciado pelo Gloo"
    sent_bytes = str(result.sent_bytes) if result.sent_frames else "gerenciado pelo Gloo"
    return (
        ("matmuls_per_rank", str(result.matmuls_per_rank),
         "Multiplicações executadas por rank e iteração.", "34"),
        ("collective_calls", str(result.collective_calls),
         "Combinações de resultados medidas, sem warmup.", "35"),
        ("sent_frames", frames,
         "Frames enviados por todos os ranks, sem warmup.", "36"),
        ("sent_bytes", sent_bytes,
         "Bytes reais enviados por todos os ranks, sem warmup.", "33"),
    )


def result_lines(result: BenchmarkResult, color: bool = False) -> list[str]:
    sections = (
        ("DESEMPENHO", _performance_metrics(result)),
        ("COMPUTE E REDE", _network_metrics(result)),
        ("PRECISÃO", _accuracy_metrics(result)),
        ("CONTADORES", _counter_metrics(result)),
    )
    lines = [_color(f"RESULTADO | mode={result.mode} | tokens={result.tokens}", "1;36", color), ""]
    for title, metrics in sections:
        lines.extend(_section_lines(title, metrics, color))
    return lines


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


def _load_pyplot() -> ModuleType:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as exc:
        raise RuntimeError("matplotlib ausente; execute uv sync --extra plots") from exc
    return pyplot


def write_png(results: Sequence[BenchmarkResult], output_prefix: str) -> Path:
    pyplot = _load_pyplot()
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
    axis.set_xscale("log", base=2)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    pyplot.close(figure)
    return path


def _write_speedup_png(results: Sequence[BenchmarkResult], output_prefix: str) -> Path:
    pyplot = _load_pyplot()
    path = Path(f"{output_prefix}_speedup.png")
    speedups = [result.speedup for result in results]
    positions = list(range(len(results)))
    colors = ["#22c55e" if speedup >= 1 else "#ef4444" for speedup in speedups]
    figure, axis = pyplot.subplots(figsize=(9, 6))
    bars = axis.barh(positions, speedups, color=colors)
    axis.axvline(1, color="#f59e0b", linestyle="--", label="Equilíbrio: 1.0x")
    axis.set_yticks(positions, labels=[str(result.tokens) for result in results])
    axis.bar_label(bars, labels=[f"{speedup:.2f}x" for speedup in speedups], padding=4)
    axis.set(xlabel="Speedup atingido", ylabel="Tokens processados juntos",
             title="Speedup tensor parallel por quantidade de tokens")
    axis.set_xlim(0, max(1.1, max(speedups) * 1.2))
    axis.grid(True, axis="x", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    pyplot.close(figure)
    return path


def write_sweep_outputs(results: Sequence[BenchmarkResult],
                        output_prefix: str) -> tuple[Path, Path, Path]:
    if not results:
        raise ValueError("results vazio; esperado pelo menos uma medição")
    return (write_csv(results, output_prefix), write_png(results, output_prefix),
            _write_speedup_png(results, output_prefix))
