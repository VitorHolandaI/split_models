"""Select the smallest useful llama.cpp fleet from measured node capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .examples import KNOWN_MODELS
from .fleet_inventory import NodeInventory
from .hardware_benchmark import HardwareScore

_APPLICATION_RAM_FRACTION = 0.5
_MAX_NODES = 24


class CapacityPlanError(ValueError):
    """A model or machine count does not fit the measured fleet."""


@dataclass(frozen=True)
class RankedNode:
    """One inventoried node plus its measured hardware and llama.cpp speed."""

    inventory: NodeInventory
    hardware: HardwareScore

    @property
    def tokens_per_second(self) -> float:
        """Effective small-model speed: Vulkan when present, CPU otherwise."""
        if self.hardware.vulkan_tokens_per_second > 0:
            return self.hardware.vulkan_tokens_per_second
        return self.hardware.cpu_tokens_per_second

    @property
    def safe_model_gib(self) -> float:
        return (self.inventory.mem_total_kb / (1024 * 1024)
                * _APPLICATION_RAM_FRACTION)


@dataclass(frozen=True)
class CapacityPlan:
    """Master-first nodes ordered by measured speed, with a safe lower bound."""

    model_gigabytes: float
    ordered_nodes: tuple[RankedNode, ...]
    minimum_total: int

    def select(self, total: int) -> tuple[RankedNode, ...]:
        """Return the requested master-inclusive prefix after range validation."""
        maximum = len(self.ordered_nodes)
        if not self.minimum_total <= total <= maximum:
            raise CapacityPlanError(
                f"maquinas={total}; esperado entre {self.minimum_total} e {maximum}")
        return self.ordered_nodes[:total]


@dataclass(frozen=True)
class ModelAllocation:
    """Estimated contiguous model slice assigned to one selected node."""

    node: RankedNode
    model_gigabytes: float
    layer_start: int | None
    layer_end: int | None


def allocate_model(plan: CapacityPlan,
                   selected: tuple[RankedNode, ...],
                   model_layers: int | None = None
                   ) -> tuple[ModelAllocation, ...]:
    """Estimate model bytes and contiguous layers for each selected node.

    The estimate follows the same safe RAM budget used by the planner. The
    runtime still lets llama.cpp refine the split from actual tensor sizes.
    """
    if model_layers is not None and model_layers < 1:
        raise CapacityPlanError(
            f"model_layers={model_layers}; esperado inteiro >= 1")
    remaining = plan.model_gigabytes
    consumed = 0.0
    allocations: list[ModelAllocation] = []
    for index, node in enumerate(selected):
        amount = min(node.safe_model_gib, max(remaining, 0.0))
        start, end = _layer_range(
            consumed, amount, plan.model_gigabytes, model_layers,
            index == len(selected) - 1)
        allocations.append(ModelAllocation(node, amount, start, end))
        consumed += amount
        remaining -= amount
    return tuple(allocations)


def _layer_range(consumed: float, amount: float, total: float,
                 layers: int | None, last: bool
                 ) -> tuple[int | None, int | None]:
    if layers is None or amount <= 0:
        return None, None
    start = min(layers - 1, math.ceil(consumed / total * layers))
    end = layers - 1 if last else max(
        start, math.ceil((consumed + amount) / total * layers) - 1)
    return start, min(end, layers - 1)


def known_model_gigabytes(model_specification: str) -> float:
    """Return measured GGUF size for one exact catalog repository and quant."""
    for model in KNOWN_MODELS:
        candidates = {model.repo, f"{model.repo}:{model.quant}"}
        if model_specification in candidates:
            return model.gigabytes
    raise CapacityPlanError(
        f"modelo={model_specification!r} sem tamanho medido; esperado item de "
        "`split-llama exemplos` ou --model-size-gb")


def known_model_layers(model_specification: str) -> int:
    """Return the known transformer layer count for a catalog model."""
    repository = model_specification.split(":", 1)[0]
    layers = {"ggml-org/gpt-oss-20b-GGUF": 24,
              "unsloth/gpt-oss-20b-GGUF": 24}.get(repository)
    if layers is None:
        raise CapacityPlanError(
            f"modelo={model_specification!r} sem numero de camadas conhecido; "
            "esperado --model-layers N")
    return layers


def build_capacity_plan(model_gigabytes: float,
                        candidates: tuple[RankedNode, ...],
                        master_host: str) -> CapacityPlan:
    """Build a master-first plan using 50% of each node's measured RAM."""
    if model_gigabytes <= 0:
        raise CapacityPlanError(
            f"modelo={model_gigabytes}; esperado tamanho em GB > 0")
    if len(candidates) > _MAX_NODES:
        raise CapacityPlanError(
            f"maquinas={len(candidates)}; esperado no maximo {_MAX_NODES}")
    master = next((node for node in candidates
                   if node.inventory.host == master_host), None)
    if master is None:
        raise CapacityPlanError(
            f"master={master_host!r}; esperado um dos hosts inventariados")
    workers = tuple(sorted((node for node in candidates if node is not master),
                           key=lambda node: node.tokens_per_second,
                           reverse=True))
    ordered = (master, *workers)
    minimum = _minimum_total(model_gigabytes, ordered)
    return CapacityPlan(model_gigabytes, ordered, minimum)


def _minimum_total(model_gigabytes: float,
                   nodes: tuple[RankedNode, ...]) -> int:
    accumulated = 0.0
    for index, node in enumerate(nodes, start=1):
        accumulated += node.safe_model_gib
        if accumulated >= model_gigabytes:
            return index
    raise CapacityPlanError(
        f"modelo={model_gigabytes:.2f} GB; frota oferece somente "
        f"{accumulated:.2f} GiB seguros")
