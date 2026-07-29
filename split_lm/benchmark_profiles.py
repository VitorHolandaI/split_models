"""Strict TOML benchmark profiles shared by local and split runners."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


PRECISIONS = ("bfloat16", "float32", "int8", "int4")
RUNTIME_PRECISIONS = ("auto", *PRECISIONS)
_FIELDS = {
    "description", "model", "context_sizes", "max_new_tokens", "warmup_runs",
    "precision", "metrics_file", "output_prefix",
}


class BenchmarkProfileError(ValueError):
    """A benchmark profile is incomplete or unsafe to execute."""


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    description: str
    model: str
    context_sizes: tuple[int, ...]
    max_new_tokens: int
    warmup_runs: int
    precision: str
    metrics_file: str
    output_prefix: str


def _text(profile: str, fields: dict[str, object], name: str) -> str:
    value = fields.get(name)
    if not isinstance(value, str) or not value:
        raise BenchmarkProfileError(f"profile={profile} {name}={value!r}; esperado texto nao vazio")
    return value


def _integer(profile: str, fields: dict[str, object], name: str) -> int:
    value = fields.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkProfileError(f"profile={profile} {name}={value!r}; esperado inteiro")
    return value


def _contexts(profile: str, fields: dict[str, object]) -> tuple[int, ...]:
    value = fields.get("context_sizes")
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise BenchmarkProfileError(
            f"profile={profile} context_sizes={value!r}; esperado lista de inteiros")
    contexts = tuple(value)
    if not contexts or any(item < 1 for item in contexts):
        raise BenchmarkProfileError(f"profile={profile} context_sizes={contexts}; esperado positivos")
    if any(left >= right for left, right in zip(contexts, contexts[1:])):
        raise BenchmarkProfileError(f"profile={profile} context_sizes={contexts}; esperado crescente")
    return contexts


def _parse_profile(name: str, value: object) -> BenchmarkProfile:
    if not isinstance(value, dict):
        raise BenchmarkProfileError(f"profile={name} value={value!r}; esperado tabela TOML")
    unknown, missing = set(value) - _FIELDS, _FIELDS - set(value)
    if unknown or missing:
        raise BenchmarkProfileError(
            f"profile={name} unknown={sorted(unknown)} missing={sorted(missing)}; esperado {_FIELDS}")
    precision = _text(name, value, "precision")
    if precision not in PRECISIONS:
        raise BenchmarkProfileError(
            f"profile={name} precision={precision!r}; esperado um de {PRECISIONS}")
    max_new, warmups = _integer(name, value, "max_new_tokens"), _integer(name, value, "warmup_runs")
    if max_new < 1 or warmups < 0:
        raise BenchmarkProfileError(
            f"profile={name} max_new_tokens={max_new} warmup_runs={warmups}; esperado tokens > 0 e warmup >= 0")
    return BenchmarkProfile(
        name, _text(name, value, "description"), _text(name, value, "model"),
        _contexts(name, value), max_new, warmups, precision,
        _text(name, value, "metrics_file"), _text(name, value, "output_prefix"))


def load_profiles(path: str | Path = "benchmarks.toml") -> dict[str, BenchmarkProfile]:
    """Load and validate every named profile from a TOML file."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchmarkProfileError(f"benchmark_config={config_path}: {exc}") from exc
    raw_profiles = document.get("profiles")
    if set(document) != {"profiles"} or not isinstance(raw_profiles, dict):
        raise BenchmarkProfileError(
            f"benchmark_config={config_path} root={sorted(document)}; esperado somente [profiles.*]")
    return {name: _parse_profile(name, value) for name, value in raw_profiles.items()}


def load_profile(path: str | Path, name: str) -> BenchmarkProfile:
    """Load one named profile and report available alternatives on a miss."""
    profiles = load_profiles(path)
    try:
        return profiles[name]
    except KeyError as exc:
        raise BenchmarkProfileError(
            f"benchmark_profile={name!r} ausente; disponiveis={sorted(profiles)}") from exc


def profile_overrides(profile: BenchmarkProfile,
                      local: bool = False) -> dict[str, object]:
    """Return CLI namespace values represented by one profile."""
    output = profile.output_prefix + "-local" if local else profile.output_prefix
    return {
        "model": profile.model,
        "context_sizes": ",".join(map(str, profile.context_sizes)),
        "max_new_tokens": profile.max_new_tokens,
        "warmup_runs": profile.warmup_runs,
        "precision": profile.precision,
        "metrics_file": profile.metrics_file,
        "benchmark_output": output,
    }


def format_profiles(profiles: dict[str, BenchmarkProfile]) -> str:
    """Render profiles in a compact operator-facing table."""
    header = (f"{'PROFILE':<22} {'PRECISION':<10} {'WARMUP':>6} "
              f"{'GENERATE':>8} {'MODEL':<32} CONTEXTS")
    rows = [
        f"{profile.name:<22} {profile.precision:<10} {profile.warmup_runs:>6} "
        f"{profile.max_new_tokens:>8} {profile.model:<32} "
        f"{','.join(map(str, profile.context_sizes))}"
        for profile in profiles.values()
    ]
    return "\n".join([header, *rows])
