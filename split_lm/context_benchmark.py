"""Context-length sweep parsing and dependency-free result exports."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .metrics import GenerationMetrics


def parse_context_sizes(specification: str, max_context: int) -> tuple[int, ...]:
    """Parse strictly increasing context sizes within the model limit."""
    try:
        sizes = tuple(int(value.strip()) for value in specification.split(","))
    except ValueError as exc:
        raise ValueError(f"context_sizes={specification!r}; esperado inteiros separados por virgula") from exc
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"context_sizes={sizes}; esperado valores positivos")
    if any(current >= following for current, following in zip(sizes, sizes[1:])):
        raise ValueError(f"context_sizes={sizes}; esperado ordem estritamente crescente")
    if sizes[-1] > max_context:
        raise ValueError(f"context_sizes={sizes}; esperado ultimo <= max_context={max_context}")
    return sizes


def _points(values: Sequence[float], left: int, top: int,
            width: int, height: int) -> str:
    maximum = max(values) if values else 1.0
    scale = maximum or 1.0
    step = width / max(1, len(values) - 1)
    return " ".join(
        f"{left + index * step:.1f},{top + height - value / scale * height:.1f}"
        for index, value in enumerate(values)
    )


def _panel(title: str, labels: Sequence[int], values: Sequence[float],
           top: int, color: str) -> list[str]:
    left, width, height = 90, 800, 120
    points = _points(values, left, top, width, height)
    lines = [
        f'<text x="20" y="{top - 12}" class="title">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis"/>',
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>',
    ]
    step = width / max(1, len(labels) - 1)
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * step
        lines.append(f'<circle cx="{x:.1f}" cy="{_point_y(values, value, top, height):.1f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + height + 18}" text-anchor="middle">{label}</text>')
        lines.append(f'<text x="{x:.1f}" y="{_point_y(values, value, top, height) - 8:.1f}" text-anchor="middle">{value:.2f}</text>')
    return lines


def _point_y(values: Sequence[float], value: float, top: int, height: int) -> float:
    maximum = max(values) if values else 1.0
    return top + height - value / (maximum or 1.0) * height


def _svg(results: Sequence[GenerationMetrics]) -> str:
    contexts = [result.prompt_tokens for result in results]
    panels = (
        ("Latencia total (s)", [result.total_seconds for result in results], 70, "#2563eb"),
        ("Tokens por segundo", [result.tokens_per_second for result in results], 270, "#16a34a"),
        ("Media entre tokens (ms)", [result.average_inter_token_seconds * 1000 for result in results], 470, "#ea580c"),
    )
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="660">',
             '<style>text{font:12px sans-serif}.title{font:bold 15px sans-serif}.axis{stroke:#475569;stroke-width:1}</style>']
    for title, values, top, color in panels:
        lines.extend(_panel(title, contexts, values, top, color))
    lines.append('</svg>')
    return "\n".join(lines)


def write_context_outputs(results: Sequence[GenerationMetrics],
                          output_prefix: str | Path) -> tuple[Path, Path]:
    """Write context-sweep CSV and SVG files without optional dependencies."""
    if not results:
        raise ValueError("results vazio; esperado pelo menos uma medicao")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path, svg_path = prefix.with_suffix(".csv"), prefix.with_suffix(".svg")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=GenerationMetrics.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    svg_path.write_text(_svg(results), encoding="utf-8")
    return csv_path, svg_path
