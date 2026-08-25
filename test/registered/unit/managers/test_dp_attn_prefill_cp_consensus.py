import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.utils.cp_utils import get_cp_padding_align_size
from sglang.srt.managers.scheduler_components.dp_attn import (
    MLPSyncBatchInfo,
    _local_prefill_cp_candidate,
    _maybe_log_local_prefill_cp_decision,
    _requires_local_prefill_cp_latch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDPPrefillCPLocalLatch(unittest.TestCase):
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
                _requires_local_prefill_cp_latch(server_args, kimi_config)
            )
            self.assertFalse(
                _requires_local_prefill_cp_latch(server_args, qwen_config)
            )

    def test_real_local_batch_is_latched_without_cross_dp_consensus(self):
        eligible = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[8],
        )
        short = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[3],
        )
        decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            extend_lens=None,
        )
        strategy = ZigzagCPStrategy(cp_size=2)

        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.cp.utils.get_cp_strategy",
                return_value=strategy,
            ),
            get_parallel().override(attn_tp_size=4),
        ):
            self.assertTrue(_local_prefill_cp_candidate(eligible, 8, 4))
            self.assertFalse(_local_prefill_cp_candidate(short, 3, 4))
            self.assertFalse(_local_prefill_cp_candidate(decode, 1, 4))
            self.assertFalse(_local_prefill_cp_candidate(None, 0, 4))

    def test_local_cp_state_is_not_serialized_into_dp_all_gather(self):
        common = dict(
            dp_size=2,
            tp_size=4,
            cp_size=2,
            num_tokens=8,
            num_tokens_for_logprob=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.EXTEND.value,
            can_run_breakable_cuda_graph=True,
        )
        cp_on = MLPSyncBatchInfo(**common, local_prefill_cp_active=True)
        cp_off = MLPSyncBatchInfo(**common, local_prefill_cp_active=False)

        self.assertEqual(cp_on._get_local_tensor("cpu").shape[0], 7)
        self.assertTrue(
            torch.equal(
                cp_on._get_local_tensor("cpu"), cp_off._get_local_tensor("cpu")
            )
        )

    def test_local_latch_removes_cp_alignment_from_global_token_table(self):
        local_off = SimpleNamespace(local_prefill_cp_active=False)
        local_on = SimpleNamespace(local_prefill_cp_active=True)

        with (
            get_parallel().override(attn_cp_size=4, attn_tp_size=4),
            patch(
                "sglang.srt.layers.utils.cp_utils.is_prefill_cp_in_seq_split",
                return_value=True,
            ),
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get",
                return_value=True,
            ),
        ):
            self.assertEqual(get_cp_padding_align_size(local_off), 1)
            self.assertEqual(get_cp_padding_align_size(local_on), 1)
            self.assertEqual(get_cp_padding_align_size(), 32)

    def test_per_batch_log_is_strictly_opt_in(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[8, 5],
            batch_size=lambda: 2,
        )
        log_target = "sglang.srt.managers.scheduler_components.dp_attn.logger.info"

        with patch(log_target) as log_info:
            _maybe_log_local_prefill_cp_decision(
                enabled=False,
                local_batch=batch,
                local_prefill_cp_active=True,
                num_tokens=16,
                attn_cp_size=2,
                attn_dp_rank=1,
            )
            log_info.assert_not_called()

            _maybe_log_local_prefill_cp_decision(
                enabled=True,
                local_batch=batch,
                local_prefill_cp_active=True,
                num_tokens=16,
                attn_cp_size=2,
                attn_dp_rank=1,
            )
            log_info.assert_called_once()
            self.assertEqual(log_info.call_args.args[1:7], (1, "EXTEND", 2, 16, [8, 5], True))

    def test_per_batch_log_reports_short_extend_reason(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[8, 3],
            batch_size=lambda: 2,
        )
        log_target = "sglang.srt.managers.scheduler_components.dp_attn.logger.info"

        with patch(log_target) as log_info:
            _maybe_log_local_prefill_cp_decision(
                enabled=True,
                local_batch=batch,
                local_prefill_cp_active=False,
                num_tokens=12,
                attn_cp_size=2,
                attn_dp_rank=0,
            )

        self.assertEqual(
            log_info.call_args.args[-1], "short_extend_min_3_required_4"
        )


if __name__ == "__main__":
    unittest.main()
