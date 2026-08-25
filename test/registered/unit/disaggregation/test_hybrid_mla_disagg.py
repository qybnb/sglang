import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
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


def _make_draft_pool(*, page_size: int = 128):
    return SimpleNamespace(
        page_size=page_size,
        get_contiguous_buf_infos=lambda: (
            list(range(1000, 1010)),
            [4096] * 10,
            [512] * 10,
        ),
    )


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

    def test_dspark_draft_kv_is_separate_from_target_mla_groups(self):
        full_kv_pool = object.__new__(NPUMLATokenToKVPool)
        full_kv_pool.layer_num = 6
        full_kv_pool.page_size = 128

        pool = _make_hybrid_pool(use_mla=True)
        pool.page_size = 128
        pool.full_kv_pool = full_kv_pool
        pool.mamba_pool = SimpleNamespace(
            get_contiguous_buf_infos=lambda: ([100], [200], [20]),
            get_state_dim_per_tensor=lambda: [64],
        )

        kv_args = KVArgs()
        # Only target latent-K and K-RoPE buffers participate in MLA grouping.
        kv_args.kv_data_ptrs = list(range(12))
        draft_pool = _make_draft_pool()
        setup_state_kv_args(
            kv_args,
            pool,
            draft_token_to_kv_pool=draft_pool,
            total_kv_layers=24,
            draft_kv_as_state=True,
        )

        self.assertEqual(kv_args.kv_buf_groups, 2)
        self.assertEqual(len(kv_args.kv_data_ptrs), 12)
        self.assertEqual(
            kv_args.state_types,
            [StateType.MAMBA, StateType.DRAFT_KV],
        )
        self.assertEqual(kv_args.state_data_ptrs[1], list(range(1000, 1010)))

    def test_dspark_draft_kv_requires_matching_page_size(self):
        pool = _make_hybrid_pool(use_mla=True)
        pool.page_size = 128
        pool.mamba_pool = SimpleNamespace(
            get_contiguous_buf_infos=lambda: ([100], [200], [20]),
            get_state_dim_per_tensor=lambda: [64],
        )
        kv_args = KVArgs()
        kv_args.kv_data_ptrs = []

        with self.assertRaisesRegex(ValueError, "same page size"):
            setup_state_kv_args(
                kv_args,
                pool,
                draft_token_to_kv_pool=_make_draft_pool(page_size=64),
                draft_kv_as_state=True,
            )

    def test_ascend_mla_state_component_bypasses_target_grouping(self):
        manager = object.__new__(AscendKVManager)
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=0,
            kv_buf_groups=2,
            total_kv_layers=24,
            mla_compression_ratios=None,
        )
        src = list(range(10))
        dst = list(range(100, 110))

        sliced_src, sliced_dst, count = manager.get_mla_kv_ptrs_with_pp(
            src, dst, StateType.DRAFT_KV
        )

        self.assertEqual(sliced_src, src)
        self.assertEqual(sliced_dst, dst)
        self.assertEqual(count, 10)


if __name__ == "__main__":
    unittest.main()
