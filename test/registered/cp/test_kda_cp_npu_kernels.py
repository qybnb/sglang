import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_affine,
    merge_kda_cp_affine_states,
)

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


if __name__ == "__main__":
    unittest.main()
