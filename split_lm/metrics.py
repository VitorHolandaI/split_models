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
                             recorded_at: str | None = None) -> GenerationMetrics:
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
        timestamp, mode, model, workers, prompt_tokens, generated,
        total, first, inter_token, rate)


def append_metrics(path: str | Path, metrics: GenerationMetrics) -> Path:
    """Append one durable JSONL generation record."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), sort_keys=True) + "\n")
    return output


def format_metrics(metrics: GenerationMetrics) -> str:
    """Render the concise report shown after each generated message."""
    return (
        f"METRICS {metrics.mode} | prompt={metrics.prompt_tokens} tokens | "
        f"generated={metrics.generated_tokens} | total={metrics.total_seconds:.3f}s | "
        f"tok/s={metrics.tokens_per_second:.2f} | "
        f"TTFT={metrics.time_to_first_token_seconds * 1000:.1f}ms | "
        f"avg_inter_token={metrics.average_inter_token_seconds * 1000:.1f}ms"
    )
