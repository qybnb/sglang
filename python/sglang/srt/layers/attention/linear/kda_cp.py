"""Context-parallel communication helpers for KDA prefill.

KDA is recurrent along the token dimension, so a rank-local token shard cannot
be scanned independently.  The prefill path transposes the CP layout with an
all-to-all:

    local tokens x all local-TP heads
        -> all tokens x a CP shard of the heads

After the full-sequence KDA scan, the inverse all-to-all restores the original
rank-local token layout.  Decode never enters these helpers.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import get_attention_cp_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args


def kda_use_prefill_cp(forward_batch: Any) -> bool:
    """Whether this forward should run the KDA sequence/head all-to-all."""
    parallel = get_parallel()
    server_args = get_global_server_args()
    mode = forward_batch.forward_mode
    return bool(
        server_args.enable_prefill_context_parallel
        and parallel.attn_cp_size > 1
        and forward_batch.attn_cp_metadata is not None
        and mode.is_context_parallel_extend()
        and not mode.is_mixed()
        and not mode.is_target_verify()
        and not mode.is_draft_extend_v2()
    )


def _validate_zigzag_metadata(metadata: Any, cp_size: int) -> None:
    required = (
        "split_list",
        "cp_reverse_index",
        "reverse_split_len",
        "per_rank_actual_token",
        "max_rank_len",
    )
    missing = [name for name in required if getattr(metadata, name, None) is None]
    if missing:
        raise NotImplementedError(
            "KDA prefill context parallel currently requires the zigzag CP "
            f"strategy; metadata is missing {missing}."
        )
    if len(metadata.per_rank_actual_token) != cp_size:
        raise ValueError(
            "KDA CP metadata/group size mismatch: "
            f"metadata={len(metadata.per_rank_actual_token)}, cp_size={cp_size}"
        )


def _rank_order_to_natural(x: torch.Tensor, metadata: Any) -> torch.Tensor:
    """Convert ``[rank0-local, rank1-local, ...]`` into natural token order."""
    chunks = torch.split(x, metadata.reverse_split_len, dim=0)
    return torch.cat([chunks[i] for i in metadata.cp_reverse_index], dim=0)


def _natural_to_rank_order(x: torch.Tensor, metadata: Any) -> torch.Tensor:
    """Inverse of :func:`_rank_order_to_natural`."""
    natural_chunks = torch.split(x, metadata.split_list, dim=0)
    inverse = [0] * len(metadata.cp_reverse_index)
    for natural_index, rank_order_index in enumerate(metadata.cp_reverse_index):
        inverse[rank_order_index] = natural_index
    return torch.cat([natural_chunks[inverse[i]] for i in range(len(inverse))], dim=0)


def sequence_to_head_a2a(
    x: torch.Tensor,
    forward_batch: Any,
    *,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """Transpose a CP token shard into a full-sequence head shard.

    Args:
        x: ``[local_tokens, local_tp_heads, ...]`` in the rank-local zigzag
            token order.
    Returns:
        ``[global_tokens, local_tp_heads / cp_size, ...]`` in natural
        per-request token order.
    """
    if x.ndim < 2:
        raise ValueError(f"KDA CP expects [tokens, heads, ...], got {tuple(x.shape)}")

    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    metadata = forward_batch.attn_cp_metadata
    _validate_zigzag_metadata(metadata, cp_size)

    num_heads = x.shape[1]
    if num_heads % cp_size != 0:
        raise ValueError(
            "KDA CP requires the local attention-TP head count to be divisible "
            f"by cp_size, got heads={num_heads}, cp_size={cp_size}."
        )
    expected_local_tokens = metadata.per_rank_actual_token[parallel.attn_cp_rank]
    if x.shape[0] != expected_local_tokens:
        raise ValueError(
            "KDA CP local token count does not match CP metadata: "
            f"tensor={x.shape[0]}, metadata={expected_local_tokens}, "
            f"cp_rank={parallel.attn_cp_rank}."
        )

    max_tokens = metadata.max_rank_len[0]
    if x.shape[0] < max_tokens:
        padding = [0, 0] * (x.ndim - 1) + [0, max_tokens - x.shape[0]]
        x = F.pad(x, padding)

    heads_per_cp_rank = num_heads // cp_size
    send = (
        x.view(max_tokens, cp_size, heads_per_cp_rank, *x.shape[2:])
        .transpose(0, 1)
        .contiguous()
    )
    recv = torch.empty_like(send)
    (group or get_attention_cp_group()).all_to_all_single(recv, send)

    rank_order = torch.cat(
        [
            recv[rank, : metadata.per_rank_actual_token[rank]]
            for rank in range(cp_size)
        ],
        dim=0,
    )
    return _rank_order_to_natural(rank_order, metadata)


def head_to_sequence_a2a(
    x: torch.Tensor,
    forward_batch: Any,
    *,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """Inverse KDA CP transpose.

    Args:
        x: ``[global_tokens, heads_per_cp_rank, ...]`` in natural token order.
    Returns:
        ``[local_tokens, local_tp_heads, ...]`` in this rank's zigzag order.
    """
    if x.ndim < 2:
        raise ValueError(f"KDA CP expects [tokens, heads, ...], got {tuple(x.shape)}")

    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    metadata = forward_batch.attn_cp_metadata
    _validate_zigzag_metadata(metadata, cp_size)
    if x.shape[0] != sum(metadata.split_list):
        raise ValueError(
            "KDA CP global token count does not match CP metadata: "
            f"tensor={x.shape[0]}, metadata={sum(metadata.split_list)}."
        )

    rank_order = _natural_to_rank_order(x, metadata)
    rank_chunks = torch.split(rank_order, metadata.per_rank_actual_token, dim=0)
    max_tokens = metadata.max_rank_len[0]
    send = x.new_zeros((cp_size, max_tokens, *x.shape[1:]))
    for rank, chunk in enumerate(rank_chunks):
        send[rank, : chunk.shape[0]].copy_(chunk)

    recv = torch.empty_like(send)
    (group or get_attention_cp_group()).all_to_all_single(recv, send.contiguous())

    local_tokens = metadata.per_rank_actual_token[parallel.attn_cp_rank]
    return torch.cat(
        [recv[head_rank, :local_tokens] for head_rank in range(cp_size)], dim=1
    )


def all_gather_cp_heads(
    x: torch.Tensor,
    *,
    head_dim: int = 1,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """Replicate a CP-head-sharded tensor on every CP rank."""
    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    if cp_size == 1:
        return x

    head_dim = head_dim % x.ndim
    local = x.movedim(head_dim, 0).contiguous()
    gathered = local.new_empty((local.shape[0] * cp_size, *local.shape[1:]))
    (group or get_attention_cp_group()).all_gather_into_tensor(gathered, local)
    return gathered.movedim(0, head_dim).contiguous()


__all__ = [
    "all_gather_cp_heads",
    "head_to_sequence_a2a",
    "kda_use_prefill_cp",
    "sequence_to_head_a2a",
]
