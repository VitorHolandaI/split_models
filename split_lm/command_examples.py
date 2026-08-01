"""Operator-facing command examples generated from configured models."""

from __future__ import annotations

from .benchmark_profiles import BenchmarkProfile


def _unique_models(default_model: str,
                   profiles: dict[str, BenchmarkProfile]) -> tuple[str, ...]:
    models = dict.fromkeys([default_model, *(profile.model for profile in profiles.values())])
    return tuple(models)


def _model_examples(default_model: str,
                    profiles: dict[str, BenchmarkProfile]) -> list[str]:
    lines = ["MODELOS CONFIGURADOS"]
    for model in _unique_models(default_model, profiles):
        label = "padrao da appliance" if model == default_model else "benchmark; valide a RAM"
        lines.extend([
            f"\n{model} ({label})",
            f"  SPLIT_MODEL={model} /root/split-master.sh",
            f"  split-cluster master --discover --model {model}",
        ])
    return lines


def _profile_examples(profiles: dict[str, BenchmarkProfile]) -> list[str]:
    lines = ["\nPERFIS DE TESTE"]
    for profile in profiles.values():
        lines.extend([
            f"\n{profile.name}: {profile.description}",
            f"  modelo: {profile.model}",
            f"  split-cluster master --discover --benchmark-profile {profile.name}",
            f"  split-cluster benchmark-local --benchmark-profile {profile.name}",
        ])
    return lines


def format_command_examples(default_model: str,
                            profiles: dict[str, BenchmarkProfile]) -> str:
    """Render runnable discovery, model, and benchmark commands."""
    general = [
        "COMANDOS GERAIS",
        "  split-cluster worker",
        "  split-cluster nodes",
        "  /root/split-master.sh",
        "  split-cluster benchmarks",
    ]
    return "\n".join([*general, "", *_model_examples(default_model, profiles),
                      *_profile_examples(profiles)])
