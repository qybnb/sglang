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

    This helper deliberately ignores ``global_prefill_cp_active``. The DP
    scheduler uses it to compute the local candidate before reaching global
    consensus, while :func:`is_cp_v2_active` consumes that consensus during
    buffer planning and execution.
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

    Kimi-K3 with DP-attention synchronizes this decision before idle batches
    are fabricated. Reuse the synchronized value in every later stage so an
    idle rank cannot plan non-CP buffers and then switch to CP after its
    ``IDLE`` batch is converted to a shadow ``EXTEND`` batch.
    """
    forced = getattr(forward_batch, "global_prefill_cp_active", None)
    if forced is not None:
        if not forced or not enable_cp_v2() or get_cp_strategy() is None:
            return False

        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None:
            return False
        if forward_mode.is_context_parallel_extend():
            return not getattr(forward_mode, "is_mixed", lambda: False)()

        # During DP buffer planning an idle rank has not yet been converted to
        # the fabricated EXTEND batch that shadows active ranks. It must still
        # reserve the CP-local buffer selected by the synchronized decision.
        return bool(
            getattr(forward_mode, "is_idle", lambda: False)()
            and getattr(forward_batch, "is_extend_in_batch", False)
        )

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
    "get_cp_strategy",
    "is_cp_v2_active",
    "cp_gather_after_forward",
    "cp_split_before_forward",
    "prepare_cp_forward",
]
