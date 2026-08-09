import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_cp import (
    all_gather_cp_heads,
    head_to_sequence_a2a,
    sequence_to_head_a2a,
)
from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
)
from sglang.srt.layers.cp.utils import (
    cp_split_before_forward,
    enable_cp_v2,
    is_cp_v2_active,
)
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.utils.cp_utils import (
    get_npu_mla_cp_ring_fallback_reason,
    get_zigzag_cp_rank_block_lengths,
    get_zigzag_cp_rank_chunk_indices,
    get_zigzag_mla_cp_ring_visibility,
    pack_paged_prefix_cache,
    use_npu_mla_cp_ring,
)
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    handle_attention_ascend,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ExtendMode:
    def is_context_parallel_extend(self):
        return True

    def is_mixed(self):
        return False


class _MixedMode(_ExtendMode):
    def is_mixed(self):
        return True


class _FakeCPGroup:
    def __init__(self, all_rank_tensors):
        self.all_rank_tensors = all_rank_tensors

    def cp_all_gather_into_tensor_async(self, output, input_tensor, stream):
        del input_tensor, stream
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class _FakeA2AGroup:
    def __init__(self, all_rank_sends, rank):
        self.all_rank_sends = all_rank_sends
        self.rank = rank

    def all_to_all_single(self, output, input_tensor):
        torch.testing.assert_close(input_tensor, self.all_rank_sends[self.rank])
        for source_rank, source_send in enumerate(self.all_rank_sends):
            output[source_rank].copy_(source_send[self.rank])


class _FakeHeadGatherGroup:
    def __init__(self, all_rank_tensors):
        self.all_rank_tensors = all_rank_tensors

    def all_gather_into_tensor(self, output, input_tensor):
        del input_tensor
        torch.cat(self.all_rank_tensors, dim=0, out=output)


class TestCPStrategyUnit(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def test_ascend_prefill_cp_dispatches_to_mla(self):
        forward_mode = SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        forward_batch = SimpleNamespace(forward_mode=forward_mode)
        attn = SimpleNamespace(use_dsa=False, mla_enable_prefill_cp=True)

        with patch(
            "sglang.srt.models.deepseek_common.attention_backend_handler."
            "mla_use_prefill_cp",
            return_value=True,
        ) as use_prefill_cp, patch(
            "sglang.srt.models.deepseek_common.attention_backend_handler."
            "use_npu_mla_cp_ring",
            return_value=False,
        ) as use_ring:
            method = handle_attention_ascend(attn, forward_batch)

        self.assertEqual(method, AttnForwardMethod.MLA_NPU)
        use_prefill_cp.assert_called_once_with(forward_batch, True)
        use_ring.assert_called_once_with(forward_batch, attn)

    def test_ascend_prefill_cp_ring_dispatches_to_mha(self):
        forward_mode = SimpleNamespace(
            is_extend=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        forward_batch = SimpleNamespace(forward_mode=forward_mode)
        attn = SimpleNamespace(
            use_dsa=False,
            mla_enable_prefill_cp=True,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

        with patch(
            "sglang.srt.models.deepseek_common.attention_backend_handler."
            "mla_use_prefill_cp",
            return_value=True,
        ), patch(
            "sglang.srt.models.deepseek_common.attention_backend_handler."
            "use_npu_mla_cp_ring",
            return_value=True,
        ):
            method = handle_attention_ascend(attn, forward_batch)

        self.assertEqual(method, AttnForwardMethod.MHA_NPU)

    def test_mla_cp_ring_batch_guard(self):
        metadata = SimpleNamespace(bs=1, split_list=[64] * 8)
        forward_batch = SimpleNamespace(
            attn_cp_metadata=metadata,
            extend_prefix_lens_cpu=[0],
        )

        def update_payload_metadata():
            rank_tokens = []
            for rank in range(4):
                early_lens, late_lens = get_zigzag_cp_rank_block_lengths(
                    metadata.split_list, metadata.bs, 4, rank
                )
                rank_tokens.append(sum(early_lens) + sum(late_lens))
            metadata.per_rank_actual_token = rank_tokens
            metadata.max_rank_len = [max(rank_tokens)] * 4

        update_payload_metadata()
        with patch(
            "sglang.srt.layers.utils.cp_utils.get_global_server_args",
            return_value=SimpleNamespace(mla_cp_backend="ring"),
        ), patch(
            "sglang.srt.layers.utils.cp_utils.get_parallel",
            return_value=SimpleNamespace(attn_cp_size=4),
        ):
            self.assertIsNone(get_npu_mla_cp_ring_fallback_reason(forward_batch))
            self.assertTrue(use_npu_mla_cp_ring(forward_batch))

            metadata.bs = 2
            metadata.split_list = [65] + [64] * 7 + [33] + [32] * 7
            forward_batch.extend_prefix_lens_cpu = [0, 0]
            update_payload_metadata()
            self.assertIsNone(get_npu_mla_cp_ring_fallback_reason(forward_batch))
            self.assertTrue(use_npu_mla_cp_ring(forward_batch))

            forward_batch.extend_prefix_lens_cpu = [0, 65]
            self.assertIsNone(get_npu_mla_cp_ring_fallback_reason(forward_batch))

            forward_batch.extend_prefix_lens_cpu = [0, 1]
            self.assertEqual(
                get_npu_mla_cp_ring_fallback_reason(forward_batch),
                "non-zero prefix length must be at least the largest zigzag "
                "block for npu_ring_mla",
            )

            metadata.split_list[-1] = 0
            self.assertEqual(
                get_npu_mla_cp_ring_fallback_reason(forward_batch),
                "each request's zigzag blocks must have non-zero lengths",
            )
            forward_batch.extend_prefix_lens_cpu = [0, 0]
            metadata.split_list = [513] * 8 + [32] * 8
            self.assertIn(
                "512-token",
                get_npu_mla_cp_ring_fallback_reason(forward_batch),
            )

            metadata.split_list = [65] + [64] * 7 + [33] + [32] * 7
            update_payload_metadata()
            metadata.max_rank_len = [1] * 4
            self.assertEqual(
                get_npu_mla_cp_ring_fallback_reason(forward_batch),
                "context-parallel ring payload metadata is invalid",
            )

    def test_mla_cp_ring_maps_rank_shards_to_natural_cache_locations(self):
        cp_size = 4
        blocks_per_request = 2 * cp_size
        split_list = [2] * blocks_per_request + [1] * blocks_per_request
        full_kv = torch.arange(sum(split_list)).unsqueeze(1)
        natural_chunks = torch.split(full_kv, split_list, dim=0)
        restored_cache = torch.full_like(full_kv, -1)

        for rank in range(cp_size):
            shard_indices = get_zigzag_cp_rank_chunk_indices(
                bs=2, cp_size=cp_size, cp_rank=rank
            )
            shard = torch.cat(
                [natural_chunks[index] for index in shard_indices], dim=0
            )
            shard_cache_locs = torch.cat(
                [natural_chunks[index].flatten() for index in shard_indices], dim=0
            )
            restored_cache[shard_cache_locs] = shard

        self.assertEqual(
            get_zigzag_cp_rank_chunk_indices(bs=2, cp_size=4, cp_rank=0),
            [0, 8, 7, 15],
        )
        torch.testing.assert_close(restored_cache, full_kv)

        with self.assertRaisesRegex(ValueError, "Invalid zigzag CP topology"):
            get_zigzag_cp_rank_chunk_indices(bs=2, cp_size=4, cp_rank=4)

    def test_mla_cp_ring_derives_uneven_source_block_lengths(self):
        split_list = list(range(1, 17))
        self.assertEqual(
            get_zigzag_cp_rank_block_lengths(split_list, 2, 4, 0),
            ([1, 9], [8, 16]),
        )
        self.assertEqual(
            get_zigzag_cp_rank_block_lengths(split_list, 2, 4, 3),
            ([4, 12], [5, 13]),
        )

        for total_tokens in range(8, 80):
            base, remainder = divmod(total_tokens, 8)
            blocks = [
                base + (1 if block_id < remainder else 0)
                for block_id in range(8)
            ]
            for cp_rank in range(4):
                local_early, local_late = get_zigzag_cp_rank_block_lengths(
                    blocks, 1, 4, cp_rank
                )
                for source_rank in range(4):
                    source_early, source_late = (
                        get_zigzag_cp_rank_block_lengths(
                            blocks, 1, 4, source_rank
                        )
                    )
                    early_to_prev, _, late_to_next = (
                        get_zigzag_mla_cp_ring_visibility(cp_rank, source_rank)
                    )
                    if early_to_prev:
                        self.assertGreaterEqual(source_early[0], local_early[0])
                    self.assertGreaterEqual(source_early[0], local_late[0])
                    if late_to_next:
                        self.assertGreaterEqual(source_late[0], local_late[0])

    def test_mla_cp_ring_packs_prefix_pages_per_request(self):
        selected_pages = torch.arange(12).view(3, 4, 1)
        packed_prefix = pack_paged_prefix_cache(
            selected_pages, prefix_lens=[3, 5], page_size=4
        )
        torch.testing.assert_close(
            packed_prefix.flatten(), torch.tensor([0, 1, 2, 4, 5, 6, 7, 8])
        )

    def test_mla_cp_ring_zigzag_visibility_is_causal(self):
        cp_size = 4
        for cp_rank in range(cp_size):
            prev_visible_blocks = []
            next_visible_blocks = []
            for source_rank in range(cp_size):
                early_to_prev, early_to_next, late_to_next = (
                    get_zigzag_mla_cp_ring_visibility(cp_rank, source_rank)
                )
                if early_to_prev:
                    prev_visible_blocks.append(source_rank)
                if early_to_next:
                    next_visible_blocks.append(source_rank)
                if late_to_next:
                    next_visible_blocks.append(2 * cp_size - 1 - source_rank)

            self.assertEqual(prev_visible_blocks, list(range(cp_rank + 1)))
            self.assertEqual(
                sorted(next_visible_blocks),
                list(range(2 * cp_size - cp_rank)),
            )

    def test_strategy_kind_maps_cli_values(self):
        self.assertEqual(ContextParallelStrategyKind.NONE.value, 0)
        self.assertEqual(
            ContextParallelStrategyKind.from_string("zigzag"),
            ContextParallelStrategyKind.ZIGZAG,
        )
        self.assertEqual(
            ContextParallelStrategyKind.from_string("interleave"),
            ContextParallelStrategyKind.INTERLEAVE,
        )
        self.assertEqual(ContextParallelStrategyKind.ZIGZAG.cli_value, "zigzag")
        self.assertEqual(ContextParallelStrategyKind.INTERLEAVE.cli_value, "interleave")

    def test_init_cp_strategy_binds_zigzag_strategy(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
            )
        )

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_zigzag())
        self.assertFalse(is_interleave())
        self.assertEqual(get_cp_strategy_kind(), ContextParallelStrategyKind.ZIGZAG)

    def test_get_cp_strategy_is_initialized_under_cp_v1_and_cp_v2(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="interleave",
                attn_cp_size=4,
            )
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=False
        ):
            self.assertIsNotNone(get_cp_strategy())
            self.assertTrue(is_cp_enabled())
            self.assertTrue(is_interleave())

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertIsNotNone(get_cp_strategy())


class TestCPZigzagStrategy(CustomTestCase):
    def setUp(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=4,
                attention_backend="fa3",
            )
        )

    def tearDown(self):
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))

    def _metadata_for_rank(self, rank, *, cp_size, seq_lens, extend_seq_lens):
        strategy = ZigzagCPStrategy(cp_size=cp_size)
        with get_parallel().override(attn_cp_rank=rank):
            return strategy.build_metadata(
                num_tokens=sum(extend_seq_lens),
                seqs_len=seq_lens,
                extend_seqs_len=extend_seq_lens,
            )

    def _forward_batch(self, metadata, extend_seq_lens):
        return SimpleNamespace(
            input_ids=torch.arange(sum(extend_seq_lens)),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=extend_seq_lens,
            attn_cp_metadata=metadata,
        )

    def test_enable_cp_v2_and_is_cp_v2_active(self):
        active_batch = SimpleNamespace(
            input_ids=torch.arange(8),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[8],
        )
        inactive_batch = SimpleNamespace(
            input_ids=torch.arange(7),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[7],
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=False
        ):
            self.assertFalse(enable_cp_v2())
            self.assertFalse(is_cp_v2_active(active_batch))

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ), get_parallel().override(attn_tp_size=1):
            self.assertTrue(enable_cp_v2())
            self.assertTrue(is_cp_v2_active(active_batch))
            self.assertFalse(is_cp_v2_active(inactive_batch))

    def test_cp_v2_uses_prospective_padded_token_count(self):
        batch = SimpleNamespace(
            input_ids=torch.arange(9),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[9],
        )

        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get",
                return_value=True,
            ),
            get_parallel().override(attn_tp_size=2),
        ):
            self.assertFalse(is_cp_v2_active(batch))
            self.assertTrue(is_cp_v2_active(batch, num_tokens=16))

    def test_cp_v2_rejects_mixed_and_unbalanced_ragged_batches(self):
        mixed_batch = SimpleNamespace(
            input_ids=torch.arange(16),
            forward_mode=_MixedMode(),
            extend_seq_lens_cpu=[16],
        )
        unsafe_ragged_batch = SimpleNamespace(
            input_ids=torch.arange(40),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[14, 14, 10],
        )
        safe_ragged_batch = SimpleNamespace(
            input_ids=torch.arange(24),
            forward_mode=_ExtendMode(),
            extend_seq_lens_cpu=[9, 15],
        )

        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get",
                return_value=True,
            ),
            get_parallel().override(attn_tp_size=2),
        ):
            self.assertFalse(is_cp_v2_active(mixed_batch))
            self.assertFalse(is_cp_v2_active(unsafe_ragged_batch))
            self.assertTrue(is_cp_v2_active(safe_ragged_batch))

    def _expected_metadata(self, *, rank, cp_size, seq_lens, extend_seq_lens):
        bs = len(extend_seq_lens)
        cp_segment_num = cp_size * 2
        prefix_offsets = [
            max(int(seq_lens[i]) - int(extend_seq_lens[i]), 0) for i in range(bs)
        ]

        per_seq_block_sizes = []
        split_list = []
        for length in extend_seq_lens:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = [
            sum(
                block_sizes[rank_id] + block_sizes[cp_segment_num - 1 - rank_id]
                for block_sizes in per_seq_block_sizes
            )
            for rank_id in range(cp_size)
        ]
        max_rank_len = [max(per_rank_actual_token)] * cp_size

        zigzag_index = list(range(rank, rank + bs * cp_segment_num, cp_segment_num))
        zigzag_index += list(
            range(cp_segment_num - rank - 1, bs * cp_segment_num, cp_segment_num)
        )

        cp_reverse_index = []
        for batch_id in range(bs):
            cp_reverse_index.extend(
                list(range(batch_id, cp_segment_num * bs, 2 * bs))
                + list(range((cp_segment_num - 1) * bs + batch_id, 0, -2 * bs))
            )

        reverse_split_len = []
        for rank_id in range(cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][rank_id])
            for batch_id in range(bs):
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - rank_id]
                )

        kv_len_prev_list = []
        kv_len_next_list = []
        actual_seq_q_prev_list = []
        actual_seq_q_next_list = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - rank])
            )
            actual_seq_q_prev_list.append(block_sizes[rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - rank - 1])

        return {
            "bs": bs,
            "total_seq_lens": sum(extend_seq_lens),
            "split_list": split_list,
            "zigzag_index": zigzag_index,
            "per_rank_actual_token": per_rank_actual_token,
            "max_rank_len": max_rank_len,
            "reverse_split_len": reverse_split_len,
            "cp_reverse_index": cp_reverse_index,
            "kv_len_prev_list": kv_len_prev_list,
            "kv_len_next_list": kv_len_next_list,
            "actual_seq_q_prev_list": actual_seq_q_prev_list,
            "actual_seq_q_next_list": actual_seq_q_next_list,
        }

    def _assert_metadata_matches(self, metadata, expected):
        self.assertEqual(metadata.bs, expected["bs"])
        self.assertEqual(metadata.total_seq_lens, expected["total_seq_lens"])
        self.assertEqual(metadata.split_list, expected["split_list"])
        self.assertEqual(metadata.zigzag_index, expected["zigzag_index"])
        self.assertEqual(
            metadata.per_rank_actual_token, expected["per_rank_actual_token"]
        )
        self.assertEqual(metadata.max_rank_len, expected["max_rank_len"])
        self.assertEqual(metadata.reverse_split_len, expected["reverse_split_len"])
        self.assertEqual(metadata.cp_reverse_index, expected["cp_reverse_index"])
        self.assertEqual(metadata.kv_len_prev_list, expected["kv_len_prev_list"])
        self.assertEqual(metadata.kv_len_next_list, expected["kv_len_next_list"])
        self.assertEqual(
            metadata.actual_seq_q_prev_list, expected["actual_seq_q_prev_list"]
        )
        self.assertEqual(
            metadata.actual_seq_q_next_list, expected["actual_seq_q_next_list"]
        )
        self.assertEqual(
            metadata.cu_seqlens_q_prev_tensor.cpu().tolist(),
            [0]
            + list(
                torch.tensor(expected["actual_seq_q_prev_list"]).cumsum(dim=0).tolist()
            ),
        )
        self.assertEqual(
            metadata.cu_seqlens_q_next_tensor.cpu().tolist(),
            [0]
            + list(
                torch.tensor(expected["actual_seq_q_next_list"]).cumsum(dim=0).tolist()
            ),
        )

    def _padded_rank_tensors(self, x, *, cp_size, seq_lens, extend_seq_lens):
        per_rank = []
        metas = []
        for rank in range(cp_size):
            metadata = self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
            )
            metas.append(metadata)
            fb = self._forward_batch(metadata, extend_seq_lens)
            local = ZigzagCPStrategy(cp_size=cp_size).shard_hidden_states(x, fb)
            pad = metadata.max_rank_len[0] - local.shape[0]
            if pad:
                local = torch.nn.functional.pad(
                    local,
                    [0, 0] * (local.ndim - 1) + [0, pad],
                )
            per_rank.append(local)
        return metas, per_rank

    def test_zigzag_metadata_for_batched_sequences(self):
        cases = [
            (4, [11, 13], [9, 10]),
            (2, [8], [8]),
            (4, [100000, 200000, 80], [100000, 200000, 64]),
            (4, [100005, 200011, 25], [100000, 200000, 16]),
        ]

        for cp_size, seq_lens, extend_seq_lens in cases:
            for rank in range(cp_size):
                with self.subTest(
                    cp_size=cp_size,
                    rank=rank,
                    seq_lens=seq_lens,
                    extend_seq_lens=extend_seq_lens,
                ):
                    metadata = self._metadata_for_rank(
                        rank,
                        cp_size=cp_size,
                        seq_lens=seq_lens,
                        extend_seq_lens=extend_seq_lens,
                    )
                    expected = self._expected_metadata(
                        rank=rank,
                        cp_size=cp_size,
                        seq_lens=seq_lens,
                        extend_seq_lens=extend_seq_lens,
                    )
                    self._assert_metadata_matches(metadata, expected)

    def test_zigzag_shards_hidden_states_and_position_ids(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        positions = torch.arange(sum(extend_seq_lens))

        for rank in range(cp_size):
            metadata = self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
            )
            fb = self._forward_batch(metadata, extend_seq_lens)
            strategy = ZigzagCPStrategy(cp_size=cp_size)
            chunks = torch.split(x, metadata.split_list, dim=0)
            position_chunks = torch.split(positions, metadata.split_list, dim=-1)
            expected_x = torch.cat([chunks[i] for i in metadata.zigzag_index], dim=0)
            expected_positions = torch.cat(
                [position_chunks[i] for i in metadata.zigzag_index], dim=-1
            )

            local_x = strategy.shard_hidden_states(x, fb)
            local_positions = strategy.shard_position_ids(positions, fb)
            with patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
            ):
                helper_x, helper_positions = cp_split_before_forward(
                    x,
                    positions,
                    fb,
                )

            self.assertTrue(torch.equal(local_x, expected_x))
            self.assertTrue(torch.equal(local_positions, expected_positions))
            self.assertTrue(torch.equal(helper_x, expected_x))
            self.assertTrue(torch.equal(helper_positions, expected_positions))

    def test_zigzag_gathers_hidden_states_to_original_order(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        x = torch.arange(sum(extend_seq_lens) * 2).view(sum(extend_seq_lens), 2)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            x,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

        for rank in range(cp_size):
            local_x = padded_rank_tensors[rank][
                : metas[rank].per_rank_actual_token[rank]
            ]
            fb = self._forward_batch(metas[rank], extend_seq_lens)

            with (
                patch(
                    "sglang.srt.layers.cp.zigzag.get_attention_cp_group",
                    return_value=_FakeCPGroup(padded_rank_tensors),
                ),
                patch(
                    "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
            ):
                gathered = ZigzagCPStrategy(cp_size=cp_size).gather_hidden_states(
                    local_x, fb, stream=None
                )

            self.assertTrue(torch.equal(gathered, x))

    def test_zigzag_gathers_kv_cache_to_original_order(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        kv = torch.arange(sum(extend_seq_lens) * 2 * 3).view(sum(extend_seq_lens), 2, 3)
        metas, padded_rank_tensors = self._padded_rank_tensors(
            kv,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )

        for rank in range(cp_size):
            local_kv = padded_rank_tensors[rank][
                : metas[rank].per_rank_actual_token[rank]
            ]
            fb = self._forward_batch(metas[rank], extend_seq_lens)

            with (
                patch(
                    "sglang.srt.layers.cp.zigzag.get_attention_cp_group",
                    return_value=_FakeCPGroup(padded_rank_tensors),
                ),
                patch(
                    "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
            ):
                gathered = ZigzagCPStrategy(cp_size=cp_size).gather_kv_cache(
                    local_kv, fb, stream=None
                )

            self.assertTrue(torch.equal(gathered, kv))

    def test_zigzag_attention_dispatch_runs_prev_then_next(self):
        cp_size = 2
        seq_lens = [8]
        extend_seq_lens = [8]
        metadata = self._metadata_for_rank(
            0,
            cp_size=cp_size,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
        )
        fb = SimpleNamespace(attn_cp_metadata=metadata)
        q = torch.arange(4 * 2).view(4, 2)
        calls = []

        def attn_fn(q_chunk, cu_seqlens_q, cache_seqlens, max_seqlen_q):
            calls.append(
                (
                    q_chunk.clone(),
                    cu_seqlens_q.clone(),
                    cache_seqlens.clone(),
                    max_seqlen_q,
                )
            )
            return q_chunk + 100

        out = ZigzagCPStrategy(cp_size=cp_size).run_attention(
            q, fb, device=torch.device("cpu"), attn_fn=attn_fn
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0][0], q[:2]))
        self.assertTrue(torch.equal(calls[1][0], q[2:]))
        self.assertTrue(torch.equal(out, q + 100))

    def test_kda_a2a_transposes_sequence_and_heads_roundtrip(self):
        cp_size = 4
        seq_lens = [11, 13]
        extend_seq_lens = [9, 10]
        num_heads = 8
        heads_per_rank = num_heads // cp_size
        x = torch.arange(
            sum(extend_seq_lens) * num_heads * 2, dtype=torch.float32
        ).view(sum(extend_seq_lens), num_heads, 2)

        metadata = [
            self._metadata_for_rank(
                rank,
                cp_size=cp_size,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
            )
            for rank in range(cp_size)
        ]
        local_inputs = []
        sequence_sends = []
        for rank in range(cp_size):
            fb = self._forward_batch(metadata[rank], extend_seq_lens)
            local = ZigzagCPStrategy(cp_size=cp_size).shard_hidden_states(x, fb)
            local_inputs.append(local)
            max_tokens = metadata[rank].max_rank_len[0]
            padded = torch.nn.functional.pad(
                local,
                [0, 0, 0, 0, 0, max_tokens - local.shape[0]],
            )
            sequence_sends.append(
                padded.view(max_tokens, cp_size, heads_per_rank, 2)
                .transpose(0, 1)
                .contiguous()
            )

        head_shards = []
        for rank in range(cp_size):
            fb = self._forward_batch(metadata[rank], extend_seq_lens)
            with get_parallel().override(
                attn_cp_size=cp_size, attn_cp_rank=rank
            ):
                head_shard = sequence_to_head_a2a(
                    local_inputs[rank],
                    fb,
                    group=_FakeA2AGroup(sequence_sends, rank),
                )
            expected = x[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
            self.assertTrue(torch.equal(head_shard, expected))
            head_shards.append(head_shard + 1000 * rank)

        inverse_sends = []
        for head_shard in head_shards:
            natural_chunks = torch.split(head_shard, metadata[0].split_list, dim=0)
            rank_order_chunks = [None] * len(natural_chunks)
            for natural_index, rank_order_index in enumerate(
                metadata[0].cp_reverse_index
            ):
                rank_order_chunks[rank_order_index] = natural_chunks[natural_index]
            rank_order = torch.cat(rank_order_chunks, dim=0)
            destination_chunks = torch.split(
                rank_order, metadata[0].per_rank_actual_token, dim=0
            )
            send = head_shard.new_zeros(
                cp_size,
                metadata[0].max_rank_len[0],
                heads_per_rank,
                head_shard.shape[-1],
            )
            for destination_rank, chunk in enumerate(destination_chunks):
                send[destination_rank, : chunk.shape[0]].copy_(chunk)
            inverse_sends.append(send)

        for rank in range(cp_size):
            fb = self._forward_batch(metadata[rank], extend_seq_lens)
            with get_parallel().override(
                attn_cp_size=cp_size, attn_cp_rank=rank
            ):
                restored = head_to_sequence_a2a(
                    head_shards[rank],
                    fb,
                    group=_FakeA2AGroup(inverse_sends, rank),
                )
            expected = local_inputs[rank].clone()
            for head_rank in range(cp_size):
                expected[
                    :,
                    head_rank * heads_per_rank : (head_rank + 1) * heads_per_rank,
                ] += 1000 * head_rank
            self.assertTrue(torch.equal(restored, expected))

    def test_kda_state_head_all_gather(self):
        cp_size = 4
        local_states = [
            torch.full((2, 3, 5), rank, dtype=torch.float32)
            for rank in range(cp_size)
        ]
        group = _FakeHeadGatherGroup(
            [state.movedim(1, 0).contiguous() for state in local_states]
        )
        expected = torch.cat(local_states, dim=1)

        for rank in range(cp_size):
            with get_parallel().override(
                attn_cp_size=cp_size, attn_cp_rank=rank
            ):
                gathered = all_gather_cp_heads(
                    local_states[rank], head_dim=1, group=group
                )
            self.assertTrue(torch.equal(gathered, expected))


if __name__ == "__main__":
    unittest.main()
