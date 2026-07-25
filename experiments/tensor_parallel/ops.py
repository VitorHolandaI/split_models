"""Reference tensor-parallel matrix multiplication operations."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def column_parallel(inputs: torch.Tensor, weight: torch.Tensor,
                    shards: int) -> torch.Tensor:
    """Split output columns and concatenate partial products.

    Usage example:
        output = column_parallel(inputs, weight, shards=2)
    """
    weight_shards = torch.tensor_split(weight, shards, dim=1)
    return torch.cat([inputs @ shard for shard in weight_shards], dim=-1)


def row_parallel(inputs: torch.Tensor, weight: torch.Tensor,
                 shards: int) -> torch.Tensor:
    """Split input rows and sum partial products.

    Usage example:
        output = row_parallel(inputs, weight, shards=2)
    """
    input_shards = torch.tensor_split(inputs, shards, dim=-1)
    weight_shards = torch.tensor_split(weight, shards, dim=0)
    partials = [input_shard @ weight_shard
                for input_shard, weight_shard in zip(input_shards, weight_shards)]
    return torch.stack(partials).sum(dim=0)


def mlp_parallel(inputs: torch.Tensor, gate: torch.Tensor, up: torch.Tensor,
                 down: torch.Tensor, shards: int) -> torch.Tensor:
    """Keep the expanded MLP representation sharded until down projection.

    Usage example:
        output = mlp_parallel(inputs, gate, up, down, shards=2)
    """
    gate_shards = torch.tensor_split(gate, shards, dim=1)
    up_shards = torch.tensor_split(up, shards, dim=1)
    down_shards = torch.tensor_split(down, shards, dim=0)
    partials = []
    for gate_shard, up_shard, down_shard in zip(gate_shards, up_shards, down_shards):
        expanded = functional.silu(inputs @ gate_shard) * (inputs @ up_shard)
        partials.append(expanded @ down_shard)
    return torch.stack(partials).sum(dim=0)
