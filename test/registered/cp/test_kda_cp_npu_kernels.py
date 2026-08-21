import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    AscendAttnMaskBuilder,
)
from sglang.srt.layers.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_affine,
    merge_kda_cp_affine_states,
)
from sglang.srt.layers.attention.fla.kda import chunk_kda_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.linear.kda_cp import (
    _begin_all_gather_fixed_shape,
)

HAS_NPU = bool(
    hasattr(torch, "npu")
    and hasattr(torch.npu, "is_available")
    and torch.npu.is_available()
)


class TestMLACPRingCausalPlan(unittest.TestCase):
    def _run_plan(self, batched_diagonal: bool, batch_prefix_max_tokens: int = 16384):
        backend = object.__new__(AscendAttnBackend)
        backend._mla_cp_ring_seq_lens_cache = {}
        backend._mla_cp_ring_tiled_logged = True
        backend.mla_cp_ring_batch_causal_tiles = batched_diagonal
        backend.mla_cp_ring_batch_prefix_tiles = True
        backend.mla_cp_ring_batch_prefix_max_tokens = batch_prefix_max_tokens
        calls = []

        def fake_segment(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            value,
            seq_lens,
            layer,
            causal,
            previous_output,
            previous_lse,
        ):
            calls.append(
                (
                    causal,
                    q_nope.shape[0],
                    k_nope.shape[0],
                    seq_lens.tolist(),
                )
            )
            output = (
                torch.zeros_like(q_nope) if previous_output is None else previous_output
            )
            lse = (
                torch.zeros(layer.tp_q_head_num, q_nope.shape[0])
                if previous_lse is None
                else previous_lse
            )
            return output, lse

        backend._run_mla_cp_ring_segment = fake_segment
        token_count = 1300
        q_nope = torch.empty(token_count, 2, 128)
        q_rope = torch.empty(token_count, 2, 64)
        k_nope = torch.empty_like(q_nope)
        k_rope = torch.empty_like(q_rope)
        value = torch.empty_like(q_nope)
        output, lse = backend._run_mla_cp_ring_causal_tiled(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            value,
            [token_count],
            [token_count],
            SimpleNamespace(tp_q_head_num=2),
            previous_output=None,
            previous_lse=None,
        )
        self.assertEqual(tuple(output.shape), tuple(q_nope.shape))
        self.assertEqual(tuple(lse.shape), (2, token_count))
        return calls

    def test_batched_diagonal_and_prefix_collapse_large_causal_launches(self):
        calls = self._run_plan(batched_diagonal=True)
        self.assertEqual(
            calls,
            [
                (
                    True,
                    1300,
                    1300,
                    [[512, 512, 276], [512, 512, 276]],
                ),
                (
                    False,
                    788,
                    1536,
                    [[512, 276], [512, 1024]],
                ),
            ],
        )

    def test_per_tile_rollback_keeps_established_launch_order(self):
        calls = self._run_plan(batched_diagonal=False)
        self.assertEqual(
            calls,
            [
                (True, 512, 512, [[512], [512]]),
                (False, 512, 512, [[512], [512]]),
                (True, 512, 512, [[512], [512]]),
                (False, 276, 1024, [[276], [1024]]),
                (True, 276, 276, [[276], [276]]),
            ],
        )

    def test_batched_prefix_respects_temporary_token_cap(self):
        calls = self._run_plan(batched_diagonal=True, batch_prefix_max_tokens=1000)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0][0])
        self.assertFalse(calls[1][0])
        self.assertFalse(calls[2][0])


@unittest.skipUnless(HAS_NPU, "Ascend NPU is required")
class TestKDACPNPUKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.device = torch.device("npu:0")

    def test_native_transition_matches_recurrent_fallback(self):
        torch.manual_seed(7)
        segment_lens = (67, 131)
        token_count = sum(segment_lens)
        num_heads = 4
        head_dim = 128
        k = (
            torch.randn(
                1,
                token_count,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        ).contiguous()
        w = (torch.randn_like(k) * 0.01).contiguous()
        u = (torch.randn_like(k) * 0.01).contiguous()
        gk = (
            -torch.rand(
                1,
                token_count,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 0.02
        ).contiguous()
        cu_seqlens = torch.tensor(
            [0, segment_lens[0], token_count],
            device=self.device,
            dtype=torch.int32,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_KDA_CP_NATIVE_TRANSITION": "0",
                "SGLANG_KDA_CP_PARALLEL_PREPROCESS": "0",
            },
        ):
            expected = chunk_gated_delta_rule_fwd_affine(k, w, u, gk, cu_seqlens)
        with patch.dict(
            os.environ,
            {
                "SGLANG_KDA_CP_NATIVE_TRANSITION": "1",
                "SGLANG_KDA_CP_PARALLEL_PREPROCESS": "0",
            },
        ):
            actual = chunk_gated_delta_rule_fwd_affine(k, w, u, gk, cu_seqlens)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_shape_gather_is_deferred_until_wait(self):
        local = torch.arange(12, device=self.device).view(3, 4)
        context = SimpleNamespace(
            cp_size=2,
            group=SimpleNamespace(device_group=object()),
            scratch_buffers={},
        )

        class FakeWork:
            waited = False

            def wait(self):
                self.waited = True

        work = FakeWork()

        def fake_all_gather(output, input_, *, group, async_op):
            self.assertIs(group, context.group.device_group)
            self.assertTrue(async_op)
            output[: local.shape[0]].copy_(input_)
            output[local.shape[0] :].copy_(input_ + 100)
            return work

        with (
            patch.dict(os.environ, {"SGLANG_KDA_CP_ASYNC_GATHER": "1"}),
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            pending = _begin_all_gather_fixed_shape(local, context, scratch_key="unit")

        self.assertFalse(work.waited)
        gathered = pending.wait()
        self.assertTrue(work.waited)
        torch.testing.assert_close(gathered[0], local)
        torch.testing.assert_close(gathered[1], local + 100)

    def test_mla_ring_exchange_buffers_are_ping_ponged(self):
        backend = object.__new__(AscendAttnBackend)
        packed = torch.empty(17, 576, device=self.device, dtype=torch.bfloat16)
        first = backend._get_mla_cp_ring_exchange_buffer(packed, 0)
        second = backend._get_mla_cp_ring_exchange_buffer(packed, 1)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(
            first.data_ptr(),
            backend._get_mla_cp_ring_exchange_buffer(packed, 0).data_ptr(),
        )

        resized = torch.empty(19, 576, device=self.device, dtype=torch.bfloat16)
        resized_first = backend._get_mla_cp_ring_exchange_buffer(resized, 0)
        self.assertEqual(tuple(resized_first.shape), tuple(resized.shape))

    def test_fused_merge_matches_natural_order_reference(self):
        torch.manual_seed(11)
        cp_size = 2
        num_heads = 2
        key_dim = value_dim = 128
        additive = (
            torch.randn(
                cp_size,
                2,
                num_heads,
                key_dim,
                value_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-3
        )
        identity = torch.eye(key_dim, device=self.device, dtype=torch.float32).view(
            1, 1, 1, key_dim, key_dim
        )
        transition = identity + (
            torch.randn(
                cp_size,
                2,
                num_heads,
                key_dim,
                key_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-4
        )
        gathered = torch.cat((additive, transition), dim=-1).contiguous()
        initial = (
            torch.randn(
                1,
                num_heads,
                key_dim,
                value_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-2
        )

        for cp_rank in range(cp_size):
            state = initial.clone()
            local_reference = []
            for block_id in range(2 * cp_size):
                owner_rank = (
                    block_id if block_id < cp_size else 2 * cp_size - block_id - 1
                )
                source_segment = int(block_id >= cp_size)
                if owner_rank == cp_rank:
                    local_reference.append(state[0].clone())
                transform = gathered[owner_rank, source_segment]
                state = (
                    torch.matmul(transform[..., value_dim:], state)
                    + transform[..., :value_dim]
                )

            local_actual = torch.empty(
                2,
                num_heads,
                key_dim,
                value_dim,
                device=self.device,
                dtype=torch.float32,
            )
            final_actual = torch.empty_like(initial)
            merge_kda_cp_affine_states(
                gathered,
                initial,
                local_actual,
                final_actual,
                cp_rank=cp_rank,
            )
            torch.npu.synchronize()
            torch.testing.assert_close(
                local_actual,
                torch.stack(local_reference),
                rtol=2e-3,
                atol=2e-3,
            )
            torch.testing.assert_close(
                final_actual,
                state,
                rtol=2e-3,
                atol=2e-3,
            )

    def test_fused_merge_writes_split_radix_checkpoint(self):
        torch.manual_seed(13)
        cp_size = 2
        max_segments = 3
        num_heads = 2
        key_dim = value_dim = 128
        additive = (
            torch.randn(
                cp_size,
                max_segments,
                num_heads,
                key_dim,
                value_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-3
        )
        identity = torch.eye(key_dim, device=self.device, dtype=torch.float32).view(
            1, 1, 1, key_dim, key_dim
        )
        transition = identity + (
            torch.randn(
                cp_size,
                max_segments,
                num_heads,
                key_dim,
                key_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-4
        )
        gathered = torch.cat((additive, transition), dim=-1).contiguous()
        initial = (
            torch.randn(
                1,
                num_heads,
                key_dim,
                value_dim,
                device=self.device,
                dtype=torch.float32,
            )
            * 1e-2
        )

        # Natural blocks are 0, 1, split(2a, 2b), 3.  Rank 1 owns the three
        # middle transforms and the radix checkpoint follows 2a.
        owners = torch.tensor([0, 1, 1, 1, 0], device=self.device, dtype=torch.int32)
        sources = torch.tensor([0, 0, 1, 2, 1], device=self.device, dtype=torch.int32)
        local_ids = torch.tensor(
            [-1, 0, 1, 2, -1], device=self.device, dtype=torch.int32
        )
        track_step = 2

        state = initial.clone()
        local_reference = []
        tracked_reference = None
        for step_id, (owner, source) in enumerate(
            zip(owners.cpu().tolist(), sources.cpu().tolist())
        ):
            if owner == 1:
                local_reference.append(state[0].clone())
            transform = gathered[owner, source]
            state = (
                torch.matmul(transform[..., value_dim:], state)
                + transform[..., :value_dim]
            )
            if step_id == track_step:
                tracked_reference = state.clone()

        local_actual = torch.empty(
            3,
            num_heads,
            key_dim,
            value_dim,
            device=self.device,
            dtype=torch.float32,
        )
        final_actual = torch.empty_like(initial)
        tracked_actual = torch.empty_like(initial)
        merge_kda_cp_affine_states(
            gathered,
            initial,
            local_actual,
            final_actual,
            cp_rank=1,
            owner_ranks=owners,
            source_segments=sources,
            local_indices=local_ids,
            local_steps=(1, 2, 3),
            tracked_state=tracked_actual,
            track_step=track_step,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            local_actual, torch.stack(local_reference), rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(final_actual, state, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(
            tracked_actual, tracked_reference, rtol=2e-3, atol=2e-3
        )

    def test_kda_inter_block_32_matches_established_block_16(self):
        torch.manual_seed(17)
        token_count = 137
        num_heads = 4
        key_dim = 128
        q = torch.randn(
            1,
            token_count,
            num_heads,
            key_dim,
            device=self.device,
            dtype=torch.bfloat16,
        ).contiguous()
        k = torch.randn_like(q).contiguous()
        gk = (-torch.rand_like(q, dtype=torch.float32) * 0.02).contiguous()
        beta = torch.rand(
            1,
            token_count,
            num_heads,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        cu_seqlens = torch.tensor(
            [0, 67, token_count], device=self.device, dtype=torch.int32
        )

        expected = chunk_kda_scaled_dot_kkt_fwd(
            q,
            k,
            gk,
            beta,
            scale=key_dim**-0.5,
            cu_seqlens=cu_seqlens,
            inter_block_size=16,
        )
        actual = chunk_kda_scaled_dot_kkt_fwd(
            q,
            k,
            gk,
            beta,
            scale=key_dim**-0.5,
            cu_seqlens=cu_seqlens,
            inter_block_size=32,
        )
        torch.npu.synchronize()
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=2e-3, atol=2e-3
            )

    def test_kda_fused_full_chunk_matches_split_kernels(self):
        torch.manual_seed(19)
        token_count = 137
        num_heads = 4
        key_dim = 128
        q = torch.randn(
            1,
            token_count,
            num_heads,
            key_dim,
            device=self.device,
            dtype=torch.bfloat16,
        ).contiguous()
        k = torch.randn_like(q).contiguous()
        gk = (-torch.rand_like(q, dtype=torch.float32) * 0.02).contiguous()
        beta = torch.rand(
            1,
            token_count,
            num_heads,
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        cu_seqlens = torch.tensor(
            [0, 67, token_count], device=self.device, dtype=torch.int32
        )

        expected = chunk_kda_scaled_dot_kkt_fwd(
            q,
            k,
            gk,
            beta,
            scale=key_dim**-0.5,
            cu_seqlens=cu_seqlens,
            inter_block_size=32,
            fused_full_chunk=False,
        )
        with patch.dict(
            os.environ,
            {"SGLANG_KDA_CP_FUSED_FULL_CHUNK": "1"},
        ):
            actual = chunk_kda_scaled_dot_kkt_fwd(
                q,
                k,
                gk,
                beta,
                scale=key_dim**-0.5,
                cu_seqlens=cu_seqlens,
                inter_block_size=32,
                fused_full_chunk=True,
            )
        torch.npu.synchronize()
        for expected_tensor, actual_tensor in zip(expected, actual):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=2e-3, atol=2e-3
            )

    def test_large_mla_ring_causal_block_matches_reference(self):
        torch.manual_seed(23)
        token_count = 600
        num_heads = 2
        scale = 192**-0.5
        backend = object.__new__(AscendAttnBackend)
        backend.ringmla_mask = AscendAttnMaskBuilder.generate_attn_mask(
            512, "norm", torch.bfloat16
        ).to(self.device)
        q_nope = (
            torch.randn(
                token_count,
                num_heads,
                128,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        q_rope = (
            torch.randn(
                token_count,
                num_heads,
                64,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        k_nope = torch.randn_like(q_nope) * 0.01
        k_rope = torch.randn_like(q_rope) * 0.01
        value = torch.randn_like(q_nope) * 0.01
        layer = SimpleNamespace(
            tp_q_head_num=num_heads,
            tp_k_head_num=num_heads,
            scaling=scale,
        )

        actual, _ = backend._run_mla_cp_ring_segment(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            value,
            torch.tensor([[token_count], [token_count]], dtype=torch.int32),
            layer,
            causal=True,
            previous_output=None,
            previous_lse=None,
        )
        q = torch.cat((q_nope, q_rope), dim=-1).float()
        k = torch.cat((k_nope, k_rope), dim=-1).float()
        scores = torch.einsum("thd,shd->hts", q, k) * scale
        scores.masked_fill_(
            torch.ones(
                token_count,
                token_count,
                device=self.device,
                dtype=torch.bool,
            ).triu(1),
            float("-inf"),
        )
        expected = torch.einsum("hts,shd->thd", scores.softmax(dim=-1), value.float())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=1e-4)

    def test_large_mla_ring_causal_block_merges_existing_prefix(self):
        torch.manual_seed(29)
        token_count = 600
        prefix_count = 96
        num_heads = 2
        scale = 192**-0.5
        backend = object.__new__(AscendAttnBackend)
        backend.ringmla_mask = AscendAttnMaskBuilder.generate_attn_mask(
            512, "norm", torch.bfloat16
        ).to(self.device)
        backend._mla_cp_ring_seq_lens_cache = {}
        backend.mla_cp_ring_batch_causal_tiles = True
        q_nope = (
            torch.randn(
                token_count,
                num_heads,
                128,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        q_rope = (
            torch.randn(
                token_count,
                num_heads,
                64,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        prefix_k_nope = (
            torch.randn(
                prefix_count,
                num_heads,
                128,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        prefix_k_rope = (
            torch.randn(
                prefix_count,
                num_heads,
                64,
                device=self.device,
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        prefix_value = torch.randn_like(prefix_k_nope) * 0.01
        k_nope = torch.randn_like(q_nope) * 0.01
        k_rope = torch.randn_like(q_rope) * 0.01
        value = torch.randn_like(q_nope) * 0.01
        layer = SimpleNamespace(
            tp_q_head_num=num_heads,
            tp_k_head_num=num_heads,
            scaling=scale,
        )

        prefix_output, prefix_lse = backend._run_mla_cp_ring_segment(
            q_nope,
            q_rope,
            prefix_k_nope,
            prefix_k_rope,
            prefix_value,
            torch.tensor([[token_count], [prefix_count]], dtype=torch.int32),
            layer,
            causal=False,
            previous_output=None,
            previous_lse=None,
        )
        actual, _ = backend._run_mla_cp_ring_segment(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            value,
            torch.tensor([[token_count], [token_count]], dtype=torch.int32),
            layer,
            causal=True,
            previous_output=prefix_output,
            previous_lse=prefix_lse,
        )

        q = torch.cat((q_nope, q_rope), dim=-1).float()
        full_k = torch.cat(
            (
                torch.cat((prefix_k_nope, prefix_k_rope), dim=-1),
                torch.cat((k_nope, k_rope), dim=-1),
            ),
            dim=0,
        ).float()
        full_value = torch.cat((prefix_value, value), dim=0).float()
        scores = torch.einsum("thd,shd->hts", q, full_k) * scale
        scores[:, :, prefix_count:].masked_fill_(
            torch.ones(
                token_count,
                token_count,
                device=self.device,
                dtype=torch.bool,
            ).triu(1),
            float("-inf"),
        )
        expected = torch.einsum("hts,shd->thd", scores.softmax(dim=-1), full_value)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
