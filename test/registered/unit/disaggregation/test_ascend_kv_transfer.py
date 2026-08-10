import concurrent.futures
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo


def _registration_frames(include_buffer_metadata: bool = True):
    frames = [
        b"room",
        b"127.0.0.1",
        b"1234",
        b"session",
        struct.pack("2Q", 3000, 4000),
        b"",
        b"",
        b"0",
        b"1",
        b"4",
        b"",
        b"",
        b"",
        b"",
    ]
    if include_buffer_metadata:
        frames.extend(
            [
                struct.pack("2Q", 64, 128),
                struct.pack("2Q", 4, 8),
            ]
        )
    return frames


class TestMooncakeRegistrationMetadata(unittest.TestCase):
    def test_parses_per_buffer_lengths(self):
        info = KVArgsRegisterInfo.from_zmq(_registration_frames())

        self.assertEqual(info.dst_kv_ptrs, [3000, 4000])
        self.assertEqual(info.dst_kv_data_lens, [64, 128])
        self.assertEqual(info.dst_kv_item_lens, [4, 8])

    def test_old_registration_falls_back_to_scalar_item_len(self):
        info = KVArgsRegisterInfo.from_zmq(
            _registration_frames(include_buffer_metadata=False)
        )

        self.assertEqual(info.dst_kv_data_lens, [])
        self.assertEqual(info.dst_kv_item_lens, [4, 4])


class TestAscendKVTransfer(unittest.TestCase):
    def _manager(self, dst_data_lens=(64, 64)):
        manager = object.__new__(AscendKVManager)
        manager.pp_size = 1
        manager.enable_custom_mem_pool = False
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[64, 64],
            kv_item_lens=[4, 4],
        )
        manager.decode_kv_args_table = {
            "session": SimpleNamespace(
                dst_kv_data_lens=list(dst_data_lens),
                dst_kv_item_lens=[4, 4],
                dst_kv_item_len=4,
            )
        }
        manager._transfer_data = Mock(return_value=0)
        return manager

    def test_validates_and_transfers_in_one_batch(self):
        manager = self._manager()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager.send_kvcache(
                "session",
                np.array([1, 2], dtype=np.int32),
                [3000, 4000],
                np.array([3, 4], dtype=np.int32),
                executor,
            )

        self.assertEqual(ret, 0)
        manager._transfer_data.assert_called_once_with(
            "session",
            [(1004, 3012, 8), (2004, 4012, 8)],
        )

    def test_rejects_destination_range_before_memfabric(self):
        manager = self._manager(dst_data_lens=(16, 16))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager.send_kvcache(
                "session",
                np.array([1, 2], dtype=np.int32),
                [3000, 4000],
                np.array([3, 4], dtype=np.int32),
                executor,
            )

        self.assertEqual(ret, -1)
        manager._transfer_data.assert_not_called()

    def test_combined_batch_failure_retries_per_buffer(self):
        manager = self._manager()
        manager._transfer_data.side_effect = [-2000, 0, 0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            ret = manager.send_kvcache(
                "session",
                np.array([1, 2], dtype=np.int32),
                [3000, 4000],
                np.array([3, 4], dtype=np.int32),
                executor,
            )

        self.assertEqual(ret, 0)
        self.assertEqual(manager._transfer_data.call_count, 3)
        self.assertFalse(manager.should_blacklist_session_on_transfer_failure())


if __name__ == "__main__":
    unittest.main()
