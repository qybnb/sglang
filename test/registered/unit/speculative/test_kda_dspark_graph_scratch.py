import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    build_dspark_verify_scratch_indices,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)


class TestKDADSparkGraphScratch(unittest.TestCase):
    def test_global_graph_rows_share_local_padding_sentinel(self):
        # DP4 with global graph BS4 and one local request has two physical
        # scratch rows: row 0 for the real request and row 1 as the sentinel.
        indices = build_dspark_verify_scratch_indices(
            num_rows=4, scratch_slots=2, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            indices, torch.tensor([0, 1, 1, 1], dtype=torch.int32)
        )

    def test_rows_are_unique_when_scratch_capacity_is_sufficient(self):
        indices = build_dspark_verify_scratch_indices(
            num_rows=4, scratch_slots=5, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            indices, torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        )

    def test_rejects_missing_scratch_storage(self):
        with self.assertRaisesRegex(ValueError, "scratch_slots must be positive"):
            build_dspark_verify_scratch_indices(
                num_rows=1, scratch_slots=0, device=torch.device("cpu")
            )

    def test_cuda_graph_init_expands_indices_to_global_bucket(self):
        backend = KDAAttnBackend.__new__(KDAAttnBackend)
        backend._dspark_target_verify = True
        backend.device = torch.device("cpu")
        backend.req_to_token_pool = SimpleNamespace(
            size=1,
            get_speculative_mamba2_params_all_layers=lambda: SimpleNamespace(
                intermediate_ssm=torch.empty((1, 2, 8, 1, 1, 1))
            ),
        )
        backend.verify_intermediate_state_indices = torch.tensor(
            [0], dtype=torch.int32
        )

        with patch.object(MambaAttnBackendBase, "init_cuda_graph_state") as parent:
            backend.init_cuda_graph_state(max_bs=4, max_num_tokens=32)

        parent.assert_called_once_with(4, 32)
        torch.testing.assert_close(
            backend.verify_intermediate_state_indices,
            torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
