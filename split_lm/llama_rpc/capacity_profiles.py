"""Strict capacity profiles: the four grids a benchmark run may execute.

A profile is a named grid of points (users, context, nodes) plus the generation
shape. The parser refuses unknown names and malformed grids so a typo in a run
never silently produces a different measurement than the plan promised.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mode_state import MODEL_PROFILES


class CapacityProfileError(ValueError):
    """A profile was requested with a name or grid this version does not have."""


_VALID_PROFILE_NAMES = frozenset(
    {"users-2k", "context-2users", "nodes-8users-2k", "capacity-envelope"})


@dataclass(frozen=True)
class CapacityPoint:
    """One load point: concurrent users, context tokens, and fleet nodes."""

    users: int
    context: int
    nodes: int


@dataclass(frozen=True)
class CapacityProfile:
    """The fixed parameters and point grid of one benchmark profile."""

    name: str
    model: str
    generated: int
    warmup_waves: int
    measured_waves: int
    points: tuple[CapacityPoint, ...]

    def __post_init__(self) -> None:
        if self.name not in _VALID_PROFILE_NAMES:
            raise CapacityProfileError(
                f"name={self.name!r}; esperado um de "
                f"{sorted(_VALID_PROFILE_NAMES)}")
        if self.generated < 1:
            raise CapacityProfileError(
                f"generated={self.generated}; esperado inteiro >= 1")
        if self.warmup_waves < 0 or self.measured_waves < 1:
            raise CapacityProfileError(
                f"warmup={self.warmup_waves}, measured={self.measured_waves}; "
                "esperado warmup >= 0 e measured >= 1")
        for point in self.points:
            if point.users < 1 or point.context < 1 or point.nodes < 1:
                raise CapacityProfileError(
                    f"ponto={point}; esperado users, context e nodes >= 1")


def _grid(users: tuple[int, ...], contexts: tuple[int, ...],
          nodes: tuple[int, ...]) -> tuple[CapacityPoint, ...]:
    return tuple(CapacityPoint(u, c, n)
                 for u in users for c in contexts for n in nodes)


# Quatro perfis fixos do plan.md. O eixo de cada um existe para responder uma
# pergunta: quantos usuarios cabem, o que o contexto custa, se nos excedentes
# ajudam, e onde o envelope termina.
PROFILES: dict[str, CapacityProfile] = {
    "users-2k": CapacityProfile(
        "users-2k", MODEL_PROFILES["small"], 128, 1, 3,
        _grid((2, 4, 8, 16, 32), (2048,), (1,))),
    "context-2users": CapacityProfile(
        "context-2users", MODEL_PROFILES["small"], 128, 1, 3,
        _grid((2,), (512, 1024, 2048, 4096, 8192), (1,))),
    "nodes-8users-2k": CapacityProfile(
        "nodes-8users-2k", MODEL_PROFILES["small"], 128, 1, 3,
        _grid((8,), (2048,), (1, 2, 4, 8))),
    "capacity-envelope": CapacityProfile(
        "capacity-envelope", MODEL_PROFILES["small"], 128, 1, 3,
        _grid((2, 4, 8, 16), (1024, 2048, 4096), (1,))),
}


def get_profile(name: str) -> CapacityProfile:
    """Resolve a profile name, refusing anything outside the fixed set.

    Usage example: `profile = get_profile("users-2k")`.
    """
    profile = PROFILES.get(name)
    if profile is None:
        raise CapacityProfileError(
            f"perfil={name!r}; esperado um de {sorted(PROFILES)}")
    return profile
