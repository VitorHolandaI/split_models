"""Aggregation, saturation detection, and the JSONL/CSV artifacts.

Every request record lands in one JSONL line; every load point lands in one CSV
row with percentiles. Saturation is a property of a sweep: the first point
where raising the load no longer raises throughput (or latency blows through a
bound) marks the envelope end.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .capacity_load import PointResult, RequestRecord

_THROUGHPUT_GROWTH_EPSILON = 0.05
_LATENCY_P95_BOUND_S = 60.0


def percentile(values: Sequence[float], percent: float) -> float:
    """Nearest-rank percentile of `values` (1..100), or 0.0 when empty.

    Usage example: `lat = percentile([0.5, 0.9], 95)`.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percent / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True)
class PointSummary:
    """One CSV row: aggregates of a point with a saturation verdict."""

    profile: str
    scenario: str
    users: int
    context: int
    nodes: int
    requests_ok: int
    requests_failed: int
    tokens_generated: int
    p50_ttft_s: float
    p95_ttft_s: float
    p99_ttft_s: float
    p50_e2e_s: float
    p95_e2e_s: float
    p99_e2e_s: float
    total_tok_s: float
    busy_slots_peak: int
    saturated: bool


def summarize_point(result: PointResult, saturated: bool = False) -> PointSummary:
    """Aggregate one point's request records into a summary row.

    Usage example: `row = summarize_point(point_result)`.
    """
    ok = [record for record in result.records if record.http_status == 200
          and not record.error]
    ttfts = [record.ttft_s for record in ok]
    e2es = [record.e2e_s for record in ok]
    total_tokens = sum(record.tokens for record in ok)
    total_time = max(sum(record.e2e_s for record in ok), 1e-6)
    return PointSummary(
        result.profile, result.scenario, result.users, result.context,
        result.nodes, len(ok), len(result.records) - len(ok), total_tokens,
        percentile(ttfts, 50), percentile(ttfts, 95), percentile(ttfts, 99),
        percentile(e2es, 50), percentile(e2es, 95), percentile(e2es, 99),
        total_tokens / total_time, result.busy_slots_peak, saturated)


def detect_saturation(summaries: Sequence[PointSummary]) -> list[bool]:
    """Mark each point saturated once throughput stops scaling with users.

    Compares each point against the previous one in sweep order; a point is
    saturated when its throughput gained less than the epsilon over the
    previous load (or p95 crossed the latency bound), which marks the envelope
    end.

    Usage example: `flags = detect_saturation(rows)`.
    """
    flags: list[bool] = []
    previous: PointSummary | None = None
    for row in summaries:
        if row.p95_e2e_s > _LATENCY_P95_BOUND_S:
            flags.append(True)
        elif previous is None:
            flags.append(False)
        elif row.users == previous.users and row.context == previous.context:
            flags.append(False)
        else:
            grew = row.total_tok_s >= previous.total_tok_s * (1.0 + _THROUGHPUT_GROWTH_EPSILON)
            flags.append(not grew)
        previous = row
    return flags


def write_request_jsonl(records: Iterable[RequestRecord], path: Path) -> None:
    """Append every request record as one JSON line at `path`."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


_SUMMARY_FIELDS = ("profile", "scenario", "users", "context", "nodes",
                   "requests_ok", "requests_failed", "tokens_generated",
                   "p50_ttft_s", "p95_ttft_s", "p99_ttft_s",
                   "p50_e2e_s", "p95_e2e_s", "p99_e2e_s",
                   "total_tok_s", "busy_slots_peak", "saturated")


def write_summary_csv(summaries: Sequence[PointSummary], path: Path) -> None:
    """Write one row per point summary, in sweep order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(_SUMMARY_FIELDS))
        for summary in summaries:
            writer.writerow([
                summary.profile, summary.scenario, summary.users,
                summary.context, summary.nodes, summary.requests_ok,
                summary.requests_failed, summary.tokens_generated,
                summary.p50_ttft_s, summary.p95_ttft_s, summary.p99_ttft_s,
                summary.p50_e2e_s, summary.p95_e2e_s, summary.p99_e2e_s,
                summary.total_tok_s, summary.busy_slots_peak,
                summary.saturated])
