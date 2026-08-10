import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import is_mla_backend, setup_state_kv_args
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_hybrid_pool(*, use_mla: bool) -> HybridLinearKVPool:
    pool = object.__new__(HybridLinearKVPool)
    pool.use_mla = use_mla
    pool.full_kv_pool = object.__new__(MLATokenToKVPool)
    return pool


class TestHybridMLADisaggregation(unittest.TestCase):
    def test_hybrid_mla_pool_uses_mla_transfer_path(self):
        self.assertTrue(is_mla_backend(_make_hybrid_pool(use_mla=True)))
        self.assertFalse(is_mla_backend(_make_hybrid_pool(use_mla=False)))
        invalid_pool = _make_hybrid_pool(use_mla=True)
        invalid_pool.full_kv_pool = object()
        self.assertFalse(is_mla_backend(invalid_pool))

    def test_npu_hybrid_mla_preserves_kv_buffer_groups(self):
        full_kv_pool = object.__new__(NPUMLATokenToKVPool)
        full_kv_pool.layer_num = 6

        pool = _make_hybrid_pool(use_mla=True)
        pool.full_kv_pool = full_kv_pool
        pool.mamba_pool = SimpleNamespace(
            get_contiguous_buf_infos=lambda: ([100], [200], [20]),
            get_state_dim_per_tensor=lambda: [64],
        )

        kv_args = KVArgs()
        # Six latent-K buffers followed by six unequal-width K-RoPE buffers.
        kv_args.kv_data_ptrs = list(range(12))
        setup_state_kv_args(kv_args, pool, total_kv_layers=24)

        self.assertEqual(kv_args.kv_buf_groups, 2)
        self.assertEqual(kv_args.total_kv_layers, 24)
        self.assertEqual(kv_args.state_types, [StateType.MAMBA])


if __name__ == "__main__":
    unittest.main()
