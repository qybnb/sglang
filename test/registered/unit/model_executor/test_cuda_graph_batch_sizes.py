"""CPU-only tests for decode CUDA graph batch-size alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model_runner(capture_bs, pool_size=1):
    return SimpleNamespace(
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=list(capture_bs))
            ),
            enable_two_batch_overlap=False,
            enable_torch_compile=False,
            torch_compile_max_bs=32,
        ),
        req_to_token_pool=SimpleNamespace(size=pool_size),
    )


class TestCudaGraphBatchSizes(unittest.TestCase):
    def test_dp_local_dspark_draft_skips_target_cp_alignment(self):
        runner = _model_runner([1])
        with (
            get_parallel().override(attn_tp_size=4, attn_cp_size=4),
            patch(
                "sglang.srt.model_executor.runner.base_cuda_graph_runner."
                "is_dp_local_cuda_graph_capture",
                return_value=True,
            ),
            patch(
                "sglang.srt.model_executor.runner.base_cuda_graph_runner."
                "require_gathered_buffer",
                return_value=True,
            ),
        ):
            capture_bs, _ = get_batch_sizes_to_capture(runner)

        self.assertEqual(capture_bs, [1])

    def test_target_capture_still_obeys_tp_cp_alignment(self):
        runner = _model_runner([1, 4])
        with (
            get_parallel().override(attn_tp_size=4, attn_cp_size=4),
            patch(
                "sglang.srt.model_executor.runner.base_cuda_graph_runner."
                "is_dp_local_cuda_graph_capture",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner.base_cuda_graph_runner."
                "require_gathered_buffer",
                return_value=True,
            ),
        ):
            capture_bs, _ = get_batch_sizes_to_capture(runner)

        self.assertEqual(capture_bs, [4])


if __name__ == "__main__":
    unittest.main()
