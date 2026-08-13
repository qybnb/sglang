import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.utils.cp_utils import get_cp_padding_align_size
from sglang.srt.managers.scheduler_components.dp_attn import (
    _global_prefill_cp_active,
    _requires_global_prefill_cp_consensus,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDPPrefillCPConsensus(unittest.TestCase):
    @staticmethod
    def _row(num_tokens: int, candidate: bool):
        # Keep this aligned with MLPSyncBatchInfo._get_local_tensor.
        return [num_tokens, 0, 0, 1, 0, 0, 0, int(candidate)]

    def test_idle_dp_is_neutral_when_all_active_batches_are_eligible(self):
        rank_info = torch.tensor(
            [
                self._row(2048, True),
                self._row(0, False),
                self._row(4096, True),
                self._row(0, False),
            ],
            dtype=torch.int64,
        )
        self.assertTrue(_global_prefill_cp_active(rank_info))

    def test_one_short_or_decode_batch_disables_cp_for_the_round(self):
        rank_info = torch.tensor(
            [
                self._row(2048, True),
                self._row(1, False),
                self._row(0, False),
                self._row(2048, True),
            ],
            dtype=torch.int64,
        )
        self.assertFalse(_global_prefill_cp_active(rank_info))

    def test_no_active_batch_does_not_enable_cp(self):
        rank_info = torch.tensor(
            [self._row(0, False), self._row(0, False)], dtype=torch.int64
        )
        self.assertFalse(_global_prefill_cp_active(rank_info))

    def test_policy_is_scoped_to_kimi_k3_cp_v2(self):
        server_args = SimpleNamespace(enable_prefill_cp=True, attn_cp_size=4)
        kimi_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["KimiK3ForConditionalGeneration"]
            ),
            hf_text_config=SimpleNamespace(
                architectures=["KimiLinearForCausalLM"]
            ),
        )
        qwen_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
            hf_text_config=None,
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertTrue(
                _requires_global_prefill_cp_consensus(server_args, kimi_config)
            )
            self.assertFalse(
                _requires_global_prefill_cp_consensus(server_args, qwen_config)
            )

    def test_disabled_round_drops_cp_only_padding(self):
        forced_off = SimpleNamespace(global_prefill_cp_active=False)
        forced_on = SimpleNamespace(global_prefill_cp_active=True)

        with (
            get_parallel().override(attn_cp_size=4),
            patch(
                "sglang.srt.layers.utils.cp_utils.is_prefill_cp_in_seq_split",
                return_value=True,
            ),
        ):
            self.assertEqual(get_cp_padding_align_size(forced_off), 1)
            self.assertEqual(get_cp_padding_align_size(forced_on), 8)


if __name__ == "__main__":
    unittest.main()
