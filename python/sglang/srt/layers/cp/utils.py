# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Public import facade and runtime helpers for context parallel strategies."""

from typing import Any, Optional, Tuple

import torch.nn.functional as F

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    get_cp_strategy,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)

CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
    {
        "KimiK3ForConditionalGeneration",
        "Qwen3MoeForCausalLM",
    }
)


def enable_cp_v2() -> bool:
    """Return whether the CP-v2 path is enabled for this process."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_CP_V2.get())


def can_cp_v2_apply(forward_batch, num_tokens: Optional[int] = None) -> bool:
    """Return whether the real local batch is structurally eligible for CP-v2.

    ``num_tokens`` lets callers planning padding evaluate the final token count
    before ``forward_batch.input_ids`` is resized. The eager path omits it and
    evaluates the already-padded input.

    This helper deliberately ignores ``local_prefill_cp_active``. The Kimi
    scheduler uses it to evaluate the real local batch once, while
    :func:`is_cp_v2_active` consumes the latched value during both buffer
    planning and execution.
    """
    if not enable_cp_v2():
        return False
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False
    if getattr(forward_mode, "is_mixed", lambda: False)():
        return False

    strategy = get_cp_strategy()
    if strategy is None:
        return False

    if num_tokens is None:
        input_ids = getattr(forward_batch, "input_ids", None)
        if input_ids is None:
            return False
        num_tokens = len(input_ids)

    return strategy.can_apply(int(num_tokens), forward_batch)


def is_cp_v2_active(forward_batch, num_tokens: Optional[int] = None) -> bool:
    """Return whether this forward must execute through CP-v2.

    Kimi-K3 with DP-attention latches this decision before idle batches are
    fabricated. Reuse the local value in every later stage so an idle rank
    cannot plan non-CP buffers and then switch to CP after its ``IDLE`` batch
    is converted to a shadow ``EXTEND`` batch. Other attention-DP replicas do
    not participate in this decision.
    """
    latched = getattr(forward_batch, "local_prefill_cp_active", None)
    if latched is not None:
        if not latched or not enable_cp_v2() or get_cp_strategy() is None:
            return False

        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None:
            return False
        if forward_mode.is_context_parallel_extend():
            return not getattr(forward_mode, "is_mixed", lambda: False)()
        return False

    return can_cp_v2_apply(forward_batch, num_tokens=num_tokens)


def prepare_cp_forward(forward_batch) -> None:
    """Build CP-v2 metadata for an active context-parallel prefill batch."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    num_tokens = len(forward_batch.input_ids)

    seq_lens_cpu = _to_int_list(getattr(forward_batch, "seq_lens_cpu", None))
    extend_lens_cpu = _to_int_list(getattr(forward_batch, "extend_seq_lens_cpu", None))
    forward_batch.attn_cp_metadata = strategy.build_metadata(
        num_tokens=num_tokens,
        seqs_len=seq_lens_cpu,
        extend_seqs_len=extend_lens_cpu,
    )


def get_cp_local_token_capacity(forward_batch, num_tokens: int) -> int:
    """Return the CP-local physical row count used by layer collectives."""
    strategy = get_cp_strategy()
    if strategy is None:
        raise RuntimeError("CP local token capacity requested without a CP strategy")
    return strategy.get_local_token_capacity(int(num_tokens), forward_batch)


def _pad_cp_tensor_to_capacity(x: Any, capacity: int, dim: int) -> Any:
    dim = dim % x.ndim
    pad_rows = int(capacity) - int(x.shape[dim])
    if pad_rows < 0:
        raise ValueError(
            "CP-local tensor exceeds its planned capacity: "
            f"shape={tuple(x.shape)}, dim={dim}, capacity={capacity}."
        )
    if pad_rows == 0:
        return x
    padding = [0, 0] * x.ndim
    reverse_dim = x.ndim - 1 - dim
    padding[2 * reverse_dim + 1] = pad_rows
    return F.pad(x, padding, mode="constant", value=0)


def get_cp_rank_actual_tokens(forward_batch) -> int:
    """Return this CP rank's real model rows before rank-local alignment."""
    from sglang.srt.runtime_context import get_parallel

    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    if metadata is None:
        raise RuntimeError("CP rank token count requested without CP metadata")
    return int(metadata.per_rank_actual_token[get_parallel().attn_cp_rank])


def trim_cp_attention_inputs(hidden_states: Any, positions: Any, forward_batch):
    """Remove rank-local alignment rows before attention/KV-state updates."""
    actual_tokens = get_cp_rank_actual_tokens(forward_batch)
    if hidden_states.shape[0] < actual_tokens or positions.shape[-1] < actual_tokens:
        raise ValueError(
            "CP attention input is smaller than its real token count: "
            f"hidden={hidden_states.shape[0]}, positions={positions.shape[-1]}, "
            f"actual={actual_tokens}."
        )
    return (
        hidden_states[:actual_tokens],
        positions[..., :actual_tokens],
        int(hidden_states.shape[0]),
    )


def pad_cp_attention_output(hidden_states: Any, capacity: int) -> Any:
    """Restore physical rows after attention for attention-TP collectives."""
    return _pad_cp_tensor_to_capacity(hidden_states, capacity, dim=0)


def pad_cp_model_inputs(hidden_states: Any, positions: Any, forward_batch):
    """Pad one CP rank for Kimi's layer-internal attention-TP token scatter."""
    strategy = get_cp_strategy()
    if strategy is None:
        raise RuntimeError("CP model input padding requested without a CP strategy")
    capacity = strategy.get_local_token_capacity(
        len(forward_batch.input_ids), forward_batch
    )
    return (
        _pad_cp_tensor_to_capacity(hidden_states, capacity, dim=0),
        _pad_cp_tensor_to_capacity(positions, capacity, dim=-1),
    )


def cp_split_before_forward(
    complete_hidden_states: Any,
    complete_position_ids: Any,
    forward_batch,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Shard embeddings and positions for CP-v2 model-runner forwarding."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    assert complete_hidden_states is not None
    assert getattr(forward_batch, "attn_cp_metadata", None) is not None
    return (
        strategy.shard_hidden_states(complete_hidden_states, forward_batch),
        strategy.shard_position_ids(complete_position_ids, forward_batch),
    )


def cp_gather_after_forward(x: Any, forward_batch, stream: Optional[Any] = None):
    """Gather CP-v2 hidden states at the model boundary when this batch is active."""
    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None

    if isinstance(x, tuple):
        hidden_states, *rest = x
        hidden_states = strategy.gather_hidden_states(
            hidden_states, forward_batch, stream
        )
        return (hidden_states, *rest)

    return strategy.gather_hidden_states(x, forward_batch, stream)


def cp_gather_aux_hidden_states_after_forward(
    aux_hidden_states: Optional[list[Any]],
    forward_batch,
    stream: Optional[Any] = None,
) -> Optional[list[Any]]:
    """Restore captured per-layer hidden states to the full CP token order.

    The CP model boundary gathers the final hidden state before logits.  Draft
    models such as DSpark additionally consume intermediate target hidden
    states, which leave the model in the same CP-local zigzag layout and must
    undergo the identical inverse gather before they are written to draft KV.
    """
    if aux_hidden_states is None:
        return None

    assert is_cp_v2_active(forward_batch)
    strategy = get_cp_strategy()
    assert strategy is not None
    return [
        strategy.gather_hidden_states(hidden_states, forward_batch, stream)
        for hidden_states in aux_hidden_states
    ]


def _to_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(x) for x in values]


__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "ContextParallelMetadata",
    "ContextParallelStrategy",
    "ContextParallelStrategyKind",
    "InterleaveCPStrategy",
    "InterleaveContextParallelMetadata",
    "ZigzagCPStrategy",
    "ZigzagContextParallelMetadata",
    "CP_V2_DEFAULT_MODEL_CLASSES",
    "can_cp_v2_apply",
    "enable_cp_v2",
    "get_cp_local_token_capacity",
    "get_cp_rank_actual_tokens",
    "get_cp_strategy",
    "is_cp_v2_active",
    "pad_cp_attention_output",
    "pad_cp_model_inputs",
    "cp_gather_after_forward",
    "cp_gather_aux_hidden_states_after_forward",
    "cp_split_before_forward",
    "prepare_cp_forward",
    "trim_cp_attention_inputs",
]
