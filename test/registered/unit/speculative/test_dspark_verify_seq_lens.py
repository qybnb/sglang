from types import SimpleNamespace
import unittest

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


if __name__ == "__main__":
    unittest.main()
