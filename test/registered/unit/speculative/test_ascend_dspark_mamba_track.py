from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendHybridLinearAttnBackend,
)


class TestAscendDSparkMambaTrack(unittest.TestCase):
    def test_ssm_tracking_uses_tracking_slots_and_steps(self):
        main_slots = torch.tensor([5, 6], dtype=torch.int32)
        track_slots = torch.tensor([9, 10], dtype=torch.int64)
        last_steps = torch.tensor([1, 2], dtype=torch.int64)
        track_steps = torch.tensor([3, -1], dtype=torch.int64)

        caches = SimpleNamespace(
            conv=[torch.empty(1)],
            temporal=torch.empty(1),
            intermediate_ssm=torch.empty((1, 2, 8, 1, 1, 1)),
            intermediate_conv_window=[torch.empty(1)],
        )
        linear_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(mamba_cache_indices=main_slots),
            req_to_token_pool=SimpleNamespace(
                get_speculative_mamba2_params_all_layers=lambda: caches
            ),
            _dspark_target_verify=True,
        )
        backend = AscendHybridLinearAttnBackend.__new__(
            AscendHybridLinearAttnBackend
        )
        backend.linear_attn_backend = linear_backend

        module = (
            "sglang.srt.hardware_backend.npu.attention."
            "ascend_hybrid_linear_attn_backend"
        )
        with (
            patch(f"{module}.move_intermediate_cache") as move,
            patch(f"{module}.speculative_state_scatter_npu"),
        ):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_steps,
                mamba_track_indices=track_slots,
                mamba_steps_to_track=track_steps,
                model=None,
            )

        self.assertEqual(move.call_count, 2)
        main_call, track_call = move.call_args_list
        torch.testing.assert_close(main_call.args[2], main_slots)
        torch.testing.assert_close(main_call.args[4], last_steps.to(torch.int32))
        torch.testing.assert_close(track_call.args[2], track_slots)
        torch.testing.assert_close(track_call.args[4], track_steps)


if __name__ == "__main__":
    unittest.main()
