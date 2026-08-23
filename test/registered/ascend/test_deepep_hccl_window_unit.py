import os
import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.token_dispatcher import deepep


class TestDeepEPHcclWindow(unittest.TestCase):
    def test_raises_global_window_to_deepep_window_on_npu(self):
        with (
            patch.object(deepep, "_is_npu", True),
            patch.dict(
                os.environ,
                {"HCCL_BUFFSIZE": "200", "DEEPEP_HCCL_BUFFSIZE": "1800"},
                clear=True,
            ),
        ):
            deepep._ensure_npu_deepep_hccl_window()
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "1800")

    def test_preserves_larger_global_window(self):
        with (
            patch.object(deepep, "_is_npu", True),
            patch.dict(
                os.environ,
                {"HCCL_BUFFSIZE": "2000", "DEEPEP_HCCL_BUFFSIZE": "1800"},
                clear=True,
            ),
        ):
            deepep._ensure_npu_deepep_hccl_window()
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "2000")

    def test_does_not_change_non_npu_environment(self):
        with (
            patch.object(deepep, "_is_npu", False),
            patch.dict(
                os.environ,
                {"HCCL_BUFFSIZE": "200", "DEEPEP_HCCL_BUFFSIZE": "1800"},
                clear=True,
            ),
        ):
            deepep._ensure_npu_deepep_hccl_window()
            self.assertEqual(os.environ["HCCL_BUFFSIZE"], "200")


if __name__ == "__main__":
    unittest.main()
