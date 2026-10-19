"""Generation latency metrics shared by local and split inference."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GenerationMetrics:
    recorded_at: str
    mode: str
    model: str
    precision: str
    workers: int
    prompt_tokens: int
    generated_tokens: int
    total_seconds: float
    time_to_first_token_seconds: float
    average_inter_token_seconds: float
    tokens_per_second: float


def _validate_times(started_at: float, finished_at: float,
                    token_times: Sequence[float]) -> None:
    if finished_at < started_at:
        raise ValueError(
            f"finished_at={finished_at} anterior a started_at={started_at}; esperado ordem temporal"
        )
    if any(current > following for current, following in zip(token_times, token_times[1:])):
        raise ValueError(f"token_times={tuple(token_times)}; esperado ordem crescente")
    if token_times and (token_times[0] < started_at or token_times[-1] > finished_at):
        raise ValueError(
            f"token_times={tuple(token_times)} fora de {started_at}..{finished_at}; esperado intervalo medido"
        )


def build_generation_metrics(mode: str, model: str, prompt_tokens: int,
                             started_at: float, finished_at: float,
                             token_times: Sequence[float], workers: int,
                             recorded_at: str | None = None,
                             precision: str = "unknown") -> GenerationMetrics:
    """Calculate end-to-end, first-token, and inter-token measurements."""
    _validate_times(started_at, finished_at, token_times)
    total = finished_at - started_at
    generated = len(token_times)
    first = token_times[0] - started_at if token_times else 0.0
    inter_token = ((token_times[-1] - token_times[0]) / (generated - 1)
                   if generated > 1 else 0.0)
    rate = generated / total if total else 0.0
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    return GenerationMetrics(
        timestamp, mode, model, precision, workers, prompt_tokens, generated,
        total, first, inter_token, rate)


def append_metrics(path: str | Path, metrics: GenerationMetrics) -> Path:
    """Append one durable JSONL generation record."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
    return output


def _color(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _metric_line(label: str, value: str, code: str, color: bool) -> str:
    return f"{label:<24}{_color(value, f'1;{code}', color)}"


def format_metrics(metrics: GenerationMetrics, color: bool = False) -> str:
    """Render aligned, optionally colored, millisecond generation metrics."""
    title = _color(
        f"=== GENERATION METRICS | {metrics.mode.upper()} | {metrics.precision.upper()} ===",
        "1;36", color)
    lines = [
        title, "",
        _metric_line("prompt_tokens", str(metrics.prompt_tokens), "34", color),
        _metric_line("generated_tokens", str(metrics.generated_tokens), "34", color),
        _metric_line("total_latency", f"{metrics.total_seconds * 1000:.1f} ms", "35", color),
        _metric_line("time_to_first_token", f"{metrics.time_to_first_token_seconds * 1000:.1f} ms", "33", color),
        _metric_line("average_inter_token", f"{metrics.average_inter_token_seconds * 1000:.1f} ms", "33", color),
        _metric_line("tokens_per_second", f"{metrics.tokens_per_second:.2f} tok/s", "32", color),
    ]
    return "\n".join(lines)
