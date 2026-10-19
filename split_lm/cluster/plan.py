"""Build a contiguous vertical-split plan from discovered nodes."""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1024 ** 3
APP_MEMORY_PER_NODE = 4 * GIB
MAX_CLUSTER_NODES = 24


class ModelCapacityError(ValueError):
    """The model exceeds the appliance's safe aggregate memory budget."""


@dataclass(frozen=True)
class ClusterNode:
    node_id: str
    hostname: str
    address: str
    control_port: int
    cpu_count: int
    available_memory: int


@dataclass(frozen=True)
class WorkerAssignment:
    node: ClusterNode
    layers: tuple[int, int]
    control_to: str
    connect_to: str | None
    token_back_to: str | None


@dataclass(frozen=True)
class VerticalPlan:
    head_layers: tuple[int, int]
    first_worker: str
    workers: tuple[WorkerAssignment, ...]


def required_cluster_nodes(model_bytes: int) -> int:
    """Return total nodes needed under the 4 GiB per-node safety budget.

    Usage example: `nodes = required_cluster_nodes(40 * 1024 ** 3)`.
    """
    if model_bytes <= 0:
        raise ValueError(f"model_bytes={model_bytes}; esperado inteiro positivo")
    required = max(2, (model_bytes + APP_MEMORY_PER_NODE - 1) // APP_MEMORY_PER_NODE)
    if required > MAX_CLUSTER_NODES:
        capacity = APP_MEMORY_PER_NODE * MAX_CLUSTER_NODES / GIB
        raise ModelCapacityError(
            f"modelo={model_bytes / GIB:.1f} GiB excede capacidade maxima={capacity:.1f} GiB"
        )
    return required


def _split_layers(total_layers: int, nodes: int) -> tuple[tuple[int, int], ...]:
    if nodes < 2 or nodes > total_layers or nodes > MAX_CLUSTER_NODES:
        raise ValueError(
            f"nodes={nodes}, total_layers={total_layers}; esperado 2 <= nodes <= "
            f"min(total_layers, maximo={MAX_CLUSTER_NODES})"
        )
    base, remainder = divmod(total_layers, nodes)
    sizes = (base + (index < remainder) for index in range(nodes))
    ranges: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


def build_vertical_plan(total_layers: int, head_address: str,
                        workers: tuple[ClusterNode, ...]) -> VerticalPlan:
    """Assign contiguous layers to a head and its ordered workers.

    Usage example: `plan = build_vertical_plan(24, "10.0.0.1", workers)`.
    """
    ranges = _split_layers(total_layers, len(workers) + 1)
    control_to = f"{head_address}:10001"
    assignments = []
    for index, node in enumerate(workers):
        next_node = workers[index + 1] if index + 1 < len(workers) else None
        assignments.append(WorkerAssignment(
            node, ranges[index + 1], control_to,
            f"{next_node.address}:9999" if next_node else None,
            None if next_node else f"{head_address}:9999",
        ))
    return VerticalPlan(ranges[0], f"{workers[0].address}:9999", tuple(assignments))
