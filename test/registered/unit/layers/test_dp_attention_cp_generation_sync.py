import unittest
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.layers.dp_attention import (
    broadcast_tensor_within_attention_dp_group,
)
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAttentionDPGenerationSync(unittest.TestCase):
    def test_sampler_always_syncs_cp_replicas(self):
        sampler = Sampler.__new__(Sampler)
        token_ids = torch.tensor([31], dtype=torch.int64)
        sampling_info = MagicMock(grammars=None)

        with (
            patch(
                "sglang.srt.layers.sampler.is_dp_attention_enabled",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.sampler.get_attention_cp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.sampler.broadcast_tensor_within_attention_dp_group"
            ) as sync,
            patch("sglang.srt.layers.sampler.torch.distributed.all_reduce") as reduce,
        ):
            sampler._sync_token_ids_across_tp(token_ids, sampling_info)

        sync.assert_called_once_with(token_ids)
        reduce.assert_not_called()

    def test_cp0_seeds_tp_row_then_cp_columns(self):
        tensor = torch.tensor([17], dtype=torch.int64)
        tp_group = MagicMock()
        cp_group = MagicMock()

        with (
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_group",
                return_value=tp_group,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_group",
                return_value=cp_group,
            ),
        ):
            result = broadcast_tensor_within_attention_dp_group(tensor)

        self.assertIs(result, tensor)
        self.assertEqual(tp_group.broadcast.call_args_list, [call(tensor, src=0)])
        self.assertEqual(cp_group.broadcast.call_args_list, [call(tensor, src=0)])

    def test_nonzero_cp_rank_only_receives_its_cp_column(self):
        tensor = torch.tensor([23], dtype=torch.int64)
        tp_group = MagicMock()
        cp_group = MagicMock()

        with (
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_rank",
                return_value=2,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_group",
                return_value=tp_group,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_group",
                return_value=cp_group,
            ),
        ):
            broadcast_tensor_within_attention_dp_group(tensor)

        tp_group.broadcast.assert_not_called()
        self.assertEqual(cp_group.broadcast.call_args_list, [call(tensor, src=0)])

    def test_single_rank_topology_is_a_noop(self):
        tensor = torch.tensor([29], dtype=torch.int64)
        tp_group = MagicMock()
        cp_group = MagicMock()

        with (
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_rank",
                return_value=0,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_tp_group",
                return_value=tp_group,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_attention_cp_group",
                return_value=cp_group,
            ),
        ):
            broadcast_tensor_within_attention_dp_group(tensor)

        tp_group.broadcast.assert_not_called()
        cp_group.broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
