from types import SimpleNamespace
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)


class TestDSparkVerifySeqLens(unittest.TestCase):
    @staticmethod
    def _executor(backend):
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=backend)
        )
        executor._verify_backend_self_adds_seq_lens_cache = None
        return executor

    def test_explicit_backend_capability(self):
        backend = SimpleNamespace(target_verify_self_adds_seq_lens=True)
        executor = self._executor(backend)
        self.assertTrue(executor._verify_backend_self_adds_seq_lens())

    def test_ordinary_backend_keeps_preextended_lengths(self):
        executor = self._executor(SimpleNamespace())
        self.assertFalse(executor._verify_backend_self_adds_seq_lens())

    def test_legacy_raw_verify_backend(self):
        backend = SimpleNamespace(make_forward_metadata_from_raw_verify=lambda: None)
        executor = self._executor(backend)
        self.assertTrue(executor._verify_backend_self_adds_seq_lens())

    def test_simulated_accept_recomputes_bonus_and_trim(self):
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.gamma = 3
        executor.verify_num_draft_tokens = 4
        executor._simulate_acc_len = 3.0
        executor._simulated_correct_drafts_buf = None

        # Row k predicts token 10*k + request_id.  A simulated acceptance
        # length of three means two accepted drafts and therefore the bonus
        # must come from verify row two, independently for each request.
        predictions = torch.tensor([[0, 10, 20, 30], [1, 11, 21, 31]])
        logits = torch.full((8, 40), -100.0)
        logits.scatter_(1, predictions.reshape(-1, 1), 100.0)

        correct_len, bonus, cap_trim_lens = executor._simulated_accept_outcome(
            bs=2,
            dtype=torch.int64,
            device=torch.device("cpu"),
            target_logits=logits,
        )

        torch.testing.assert_close(correct_len, torch.tensor([2, 2]))
        torch.testing.assert_close(bonus, torch.tensor([20, 21]))
        torch.testing.assert_close(cap_trim_lens, torch.tensor([0, 0]))


if __name__ == "__main__":
    unittest.main()
