"""Concurrent streaming load against llama-server, per-request records.

One client does one completion: it POSTs to `/completion` with `stream: true`,
counts SSE `data:` events as generated tokens, and records first-token latency
(TTFT) and wall time. The engine fans out `users` clients per wave and keeps a
per-request JSON record so failures are data, not skips.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import IO, Protocol, cast

from .capacity_profiles import CapacityPoint, CapacityProfile


class CompletionClient(Protocol):
    """One streaming completion against a llama-server endpoint."""

    def generate(self, prompt: str, n_predict: int) -> GenerationResult: ...


@dataclass(frozen=True)
class GenerationResult:
    """What one completion produced, timing and all."""

    tokens: int
    ttft_s: float
    e2e_s: float
    http_status: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.http_status == 200


class TokenCounter(Protocol):
    """Counts tokens for a text, so prompts reach an exact context size."""

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class RequestRecord:
    """One measured request, ready for the JSONL line."""

    profile: str
    scenario: str
    users: int
    context: int
    nodes: int
    wave: int
    request_id: int
    ttft_s: float
    e2e_s: float
    tokens: int
    tok_s: float
    http_status: int
    error: str

    @property
    def ok(self) -> bool:
        return self.http_status == 200 and not self.error

    def to_json(self) -> dict[str, object]:
        return {"profile": self.profile, "scenario": self.scenario,
                "users": self.users, "context": self.context,
                "nodes": self.nodes, "wave": self.wave,
                "request_id": self.request_id, "ttft_s": self.ttft_s,
                "e2e_s": self.e2e_s, "tokens": self.tokens,
                "tok_s": self.tok_s, "http_status": self.http_status,
                "error": self.error}


@dataclass(frozen=True)
class PointResult:
    """Aggregate of one load point across its measured waves."""

    profile: str
    scenario: str
    users: int
    context: int
    nodes: int
    records: tuple[RequestRecord, ...]
    busy_slots_peak: int


def exact_prompt(target_tokens: int, counter: TokenCounter,
                 max_steps: int = 50) -> str:
    """Grow a text until the tokenizer counts exactly `target_tokens`.

    The greedy search appends the same unit and backs off on overshoot; content
    is irrelevant for capacity, the token count is the point.

    Usage example: `prompt = exact_prompt(2048, counter)`.
    """
    unit = "capacidade de contexto "
    text = unit
    low = 0
    high = target_tokens * len(unit) + len(unit) * 4
    best_text = unit
    best_delta = target_tokens
    for _ in range(max_steps):
        text = unit * (high // len(unit))
        count = counter.count(text)
        delta = abs(count - target_tokens)
        if delta < best_delta:
            best_text, best_delta = text, delta
        if count == target_tokens:
            return text
        if count < target_tokens:
            low = high
            high *= 2
        else:
            high = low + (high - low) // 2
    tol = min(max(1, target_tokens // 20), 6)
    if best_delta <= tol:
        return best_text
    raise RuntimeError(
        f"nao alcancou {target_tokens} tokens em {max_steps} passos; "
        f"ultimo count={counter.count(text)}")


class HttpCompletionClient:
    """Real `/completion` client over urllib, streaming SSE events."""

    def __init__(self, endpoint: str,
                 opener: Callable[..., object] = urllib.request.urlopen,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._opener = opener
        self._clock = clock

    def generate(self, prompt: str, n_predict: int) -> GenerationResult:
        body = json.dumps({"prompt": prompt, "n_predict": n_predict,
                           "stream": True}).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint + "/completion", data=body,
            headers={"Content-Type": "application/json"})
        start = self._clock()
        first_token_at: float | None = None
        tokens = 0
        try:
            with cast(IO[bytes], self._opener(request)) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    if payload.get("stop"):
                        break
                    if first_token_at is None:
                        first_token_at = self._clock()
                    tokens += 1
            status = 200
            error = ""
        except (urllib.error.URLError, OSError, ValueError) as exc:
            status = getattr(exc, "code", 0)
            error = f"{type(exc).__name__}: {exc}"
            tokens = 0
            first_token_at = None
        end = self._clock()
        ttft = (first_token_at - start) if first_token_at is not None else end - start
        return GenerationResult(tokens, ttft, end - start, status, error)


class HttpTokenCounter:
    """Counts tokens through `/tokenize` on the same endpoint."""

    def __init__(self, endpoint: str,
                 opener: Callable[..., object] = urllib.request.urlopen) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._opener = opener

    def count(self, text: str) -> int:
        body = json.dumps({"content": text}).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint + "/tokenize", data=body,
            headers={"Content-Type": "application/json"})
        with cast(IO[bytes], self._opener(request)) as response:
            payload = json.loads(response.read().decode("utf-8"))
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(
                f"resposta de /tokenize={payload!r}; esperado campo tokens lista")
        return len(tokens)


class SlotsSampler(Protocol):
    """Samples busy slots on the serving endpoint."""

    def sample(self) -> int: ...


class HttpSlotsSampler:
    """Busy slots from `GET /slots`; `state` is 0 when idle, 1 when processing."""

    def __init__(self, endpoint: str,
                 opener: Callable[..., object] = urllib.request.urlopen) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._opener = opener

    def sample(self) -> int:
        try:
            with cast(IO[bytes], self._opener(self._endpoint + "/slots")) as response:
                slots = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return 0
        if not isinstance(slots, list):
            return 0
        return sum(1 for slot in slots
                   if isinstance(slot, dict) and slot.get("state") == 1)


def run_point(profile: CapacityProfile, scenario: str, point: CapacityPoint,
              clients: Sequence[CompletionClient],
              sampler: SlotsSampler | None = None,
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> PointResult:
    """Run every measured wave of one point, `users` concurrent each.

    The client for a request is chosen round-robin over `clients`, which makes
    the replicated scenario (several endpoints) and the distributed one (one
    master) use the same engine.

    Usage example: `result = run_point(profile, "distributed", point, clients)`.
    """
    if not clients:
        raise ValueError("clients vazio; esperado ao menos um endpoint")
    records: list[RequestRecord] = []
    request_id = 0
    peak_busy = 0
    for wave in range(1, profile.warmup_waves + profile.measured_waves + 1):
        measured = wave > profile.warmup_waves
        def one(client_index: int) -> None:
            nonlocal request_id
            client = clients[client_index % len(clients)]
            result = client.generate("x", profile.generated)
            now_id = request_id
            request_id += 1
            if not measured:
                return
            elapsed = max(result.e2e_s, 1e-6)
            records.append(RequestRecord(
                profile.name, scenario, point.users, point.context, point.nodes,
                wave, now_id, result.ttft_s, result.e2e_s, result.tokens,
                result.tokens / elapsed, result.http_status, result.error))
        with ThreadPoolExecutor(max_workers=point.users) as pool:
            for _ in range(point.users):
                pool.submit(one, request_id % len(clients))
        if sampler is not None:
            peak_busy = max(peak_busy, sampler.sample())
        if wave < profile.warmup_waves + profile.measured_waves:
            sleep(0.2)
    return PointResult(profile.name, scenario, point.users, point.context,
                       point.nodes, tuple(records), peak_busy)
