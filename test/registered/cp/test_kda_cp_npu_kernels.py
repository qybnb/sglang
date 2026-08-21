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

HAS_NPU = bool(
    hasattr(torch, "npu")
    and hasattr(torch.npu, "is_available")
    and torch.npu.is_available()
)


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
            expected = chunk_gated_delta_rule_fwd_affine(
                k, w, u, gk, cu_seqlens
            )
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
        identity = torch.eye(
            key_dim, device=self.device, dtype=torch.float32
        ).view(1, 1, 1, key_dim, key_dim)
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
                    block_id
                    if block_id < cp_size
                    else 2 * cp_size - block_id - 1
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
        identity = torch.eye(
            key_dim, device=self.device, dtype=torch.float32
        ).view(1, 1, 1, key_dim, key_dim)
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
        local_ids = torch.tensor([-1, 0, 1, 2, -1], device=self.device, dtype=torch.int32)
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
        expected = torch.einsum(
            "hts,shd->thd", scores.softmax(dim=-1), value.float()
        )
        torch.npu.synchronize()
        torch.testing.assert_close(actual.float(), expected, rtol=5e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
