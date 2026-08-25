import concurrent.futures
import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.ascend.diagnostics import (
    new_transfer_id,
    write_ascend_kv_diag,
)
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.utils.network import get_local_ip_auto

logger = logging.getLogger(__name__)
_request_diag_context = threading.local()


class AscendKVManager(MooncakeKVManager):
    def _write_diag(self, event: str, **fields) -> None:
        request_fields = getattr(_request_diag_context, "fields", {}).copy()
        request_fields.update(fields)
        write_ascend_kv_diag(
            event,
            role=getattr(getattr(self, "engine", None), "role", None),
            engine_session_id=getattr(
                getattr(self, "engine", None), "session_id", None
            ),
            engine_rank=getattr(self.kv_args, "engine_rank", None),
            npu_id=getattr(self.kv_args, "gpu_id", None),
            attn_tp_rank=getattr(self, "attn_tp_rank", None),
            attn_tp_size=getattr(self, "attn_tp_size", None),
            attn_cp_rank=getattr(self, "attn_cp_rank", None),
            attn_cp_size=getattr(self, "attn_cp_size", None),
            attn_dp_rank=getattr(self, "attn_dp_rank", None),
            attn_dp_size=getattr(self, "attn_dp_size", None),
            pp_rank=getattr(self, "pp_rank", None),
            pp_size=getattr(self, "pp_size", None),
            page_size=getattr(self.kv_args, "page_size", None),
            **request_fields,
        )

    def record_transfer_diagnostic(self, event: str, **fields) -> None:
        if event == "chunk_dispatch_start":
            _request_diag_context.fields = {
                key: fields.get(key)
                for key in (
                    "room",
                    "remote_session_id",
                    "endpoint",
                    "dst_port",
                    "chunk_id",
                    "is_last_chunk",
                )
            }
        self._write_diag(event, **fields)
        if event == "chunk_dispatch_result":
            _request_diag_context.fields = {}

    @staticmethod
    def _index_summary(indices: npt.NDArray[np.int32]) -> dict:
        values = np.asarray(indices).reshape(-1)
        count = int(values.size)
        return {
            "count": count,
            "min": int(values.min()) if count else None,
            "max": int(values.max()) if count else None,
            "head": [int(value) for value in values[:8]],
            "tail": [int(value) for value in values[-8:]] if count > 8 else [],
        }

    def should_blacklist_session_on_transfer_failure(self) -> bool:
        # MemFabric uses the same non-zero return path for transport failures
        # and local/remote address-range validation errors. An address error
        # does not mean that the decode process or its session is dead.
        return False

    def init_engine(self):
        # TransferEngine initialized on ascend.
        local_ip = get_local_ip_auto()
        self.engine = AscendTransferEngine(
            hostname=local_ip,
            npu_id=self.kv_args.gpu_id,
            disaggregation_mode=self.disaggregation_mode,
        )

    def register_buffer_to_engine(self):
        self._write_diag(
            "manager_register_start",
            is_mla_backend=bool(self.is_mla_backend),
            kv_buf_groups=getattr(self.kv_args, "kv_buf_groups", None),
            total_kv_layers=getattr(self.kv_args, "total_kv_layers", None),
            kv_buffers=[
                {
                    "buffer": index,
                    "ptr": int(ptr),
                    "capacity": int(capacity),
                    "item_len": int(item_len),
                }
                for index, (ptr, capacity, item_len) in enumerate(
                    zip(
                        self.kv_args.kv_data_ptrs,
                        self.kv_args.kv_data_lens,
                        self.kv_args.kv_item_lens,
                    )
                )
            ],
            kv_ptr_count=len(self.kv_args.kv_data_ptrs),
            kv_capacity_count=len(self.kv_args.kv_data_lens),
            kv_item_len_count=len(self.kv_args.kv_item_lens),
        )
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )
        # The Ascend backend optimize batch registration for small memory blocks.
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )
        # Batch register state/extra pool data buffers
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            if component_ptrs and component_lens:
                self.engine.batch_register(component_ptrs, component_lens)
        self._write_diag("manager_register_complete")

    def get_mla_kv_ptrs_with_pp(
        self,
        src_kv_ptrs: List[int],
        dst_kv_ptrs: List[int],
        state_type: Optional[StateType] = None,
    ) -> Tuple[List[int], List[int], int]:
        if state_type is not None:
            # State components (including dSparK draft KV) are independent of
            # the target NPU-MLA latent/K-RoPE grouping below.
            return super().get_mla_kv_ptrs_with_pp(src_kv_ptrs, dst_kv_ptrs, state_type)
        # src_kv_ptrs: k_data, v_data, index_k_data(optional)
        # dst_kv_ptrs: k_data, v_data, index_k_data(optional)
        start_layer = self.kv_args.prefill_start_layer
        kv_buf_groups = getattr(self.kv_args, "kv_buf_groups", 1)
        total_kv_layers = getattr(self.kv_args, "total_kv_layers", 0)
        src_layers = len(src_kv_ptrs) // kv_buf_groups
        # When only speculative-algorithm is enabled for decode
        # the KV has one more layer than prefill.
        # The draft layer needs to be skipped.
        dst_total_layers = (
            min(len(dst_kv_ptrs) // kv_buf_groups, total_kv_layers)
            if total_kv_layers
            else len(dst_kv_ptrs) // kv_buf_groups
        )
        end_layer = start_layer + src_layers
        if src_layers == dst_total_layers:
            sliced_dst_kv_ptrs = dst_kv_ptrs
        else:
            sliced_dst_kv_ptrs = []
            for i in range(kv_buf_groups):
                layer_offset = i * dst_total_layers
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[layer_offset + start_layer : layer_offset + end_layer]
                )
        layers_current_pp_stage = len(src_kv_ptrs)
        return src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        transfer_id = new_transfer_id()
        self._write_diag(
            "send_kvcache_start",
            transfer_id=transfer_id,
            remote_session_id=mooncake_session_id,
            prefill_indices=self._index_summary(prefill_kv_indices),
            decode_indices=self._index_summary(dst_kv_indices),
            dst_ptr_count=len(dst_kv_ptrs),
        )
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        target_info = self.decode_kv_args_table.get(mooncake_session_id)
        if target_info is None:
            self._write_diag(
                "send_kvcache_missing_registration",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
            )
            logger.error(
                "Ascend KV transfer has no registration info for session %s",
                mooncake_session_id,
            )
            return -1

        dst_kv_data_lens = target_info.dst_kv_data_lens
        if not dst_kv_data_lens:
            # Compatibility with an older decode. Boundary checks on the remote
            # side are unavailable, but the transfer can retain its old behavior.
            dst_kv_data_lens = [0] * len(dst_kv_ptrs)
            logger.warning_once(
                "Decode KV registration does not contain per-buffer capacities; "
                "restart decode with the same SGLang revision as prefill to enable "
                "Ascend transfer boundary checks."
            )
        dst_kv_item_lens = target_info.dst_kv_item_lens or [
            target_info.dst_kv_item_len
        ] * len(dst_kv_ptrs)

        if not (
            len(dst_kv_ptrs)
            == len(dst_kv_data_lens)
            == len(dst_kv_item_lens)
        ):
            self._write_diag(
                "send_kvcache_registration_length_mismatch",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                ptr_count=len(dst_kv_ptrs),
                capacity_count=len(dst_kv_data_lens),
                item_len_count=len(dst_kv_item_lens),
            )
            logger.error(
                "Ascend decode KV registration length mismatch for session %s: "
                "ptrs=%d, data_lens=%d, item_lens=%d",
                mooncake_session_id,
                len(dst_kv_ptrs),
                len(dst_kv_data_lens),
                len(dst_kv_item_lens),
            )
            return -1

        if self.pp_size > 1:
            if self.is_mla_backend:
                src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = (
                    self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                )
                src_kv_data_lens, sliced_dst_kv_data_lens, _ = (
                    self.get_mla_kv_ptrs_with_pp(
                        self.kv_args.kv_data_lens, dst_kv_data_lens
                    )
                )
                src_kv_item_lens, sliced_dst_kv_item_lens, _ = (
                    self.get_mla_kv_ptrs_with_pp(
                        self.kv_args.kv_item_lens, dst_kv_item_lens
                    )
                )
            else:
                (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                ) = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
                src_k_data_lens, src_v_data_lens, dst_k_data_lens, dst_v_data_lens, _ = (
                    self.get_mha_kv_ptrs_with_pp(
                        self.kv_args.kv_data_lens, dst_kv_data_lens
                    )
                )
                src_k_item_lens, src_v_item_lens, dst_k_item_lens, dst_v_item_lens, _ = (
                    self.get_mha_kv_ptrs_with_pp(
                        self.kv_args.kv_item_lens, dst_kv_item_lens
                    )
                )
                src_kv_ptrs = src_k_ptrs + src_v_ptrs
                sliced_dst_kv_ptrs = dst_k_ptrs + dst_v_ptrs
                src_kv_data_lens = src_k_data_lens + src_v_data_lens
                sliced_dst_kv_data_lens = dst_k_data_lens + dst_v_data_lens
                src_kv_item_lens = src_k_item_lens + src_v_item_lens
                sliced_dst_kv_item_lens = dst_k_item_lens + dst_v_item_lens
                layers_current_pp_stage *= 2
        else:
            src_kv_ptrs = self.kv_args.kv_data_ptrs
            sliced_dst_kv_ptrs = dst_kv_ptrs
            src_kv_data_lens = self.kv_args.kv_data_lens
            sliced_dst_kv_data_lens = dst_kv_data_lens
            src_kv_item_lens = self.kv_args.kv_item_lens
            sliced_dst_kv_item_lens = dst_kv_item_lens
            layers_current_pp_stage = len(src_kv_ptrs)

        metadata_lengths = {
            "src_ptrs": len(src_kv_ptrs),
            "dst_ptrs": len(sliced_dst_kv_ptrs),
            "src_data_lens": len(src_kv_data_lens),
            "dst_data_lens": len(sliced_dst_kv_data_lens),
            "src_item_lens": len(src_kv_item_lens),
            "dst_item_lens": len(sliced_dst_kv_item_lens),
        }
        if any(length != layers_current_pp_stage for length in metadata_lengths.values()):
            self._write_diag(
                "send_kvcache_pp_metadata_mismatch",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                expected=layers_current_pp_stage,
                actual=metadata_lengths,
            )
            logger.error(
                "Ascend KV metadata does not align after PP slicing for session %s: "
                "expected=%d, actual=%s",
                mooncake_session_id,
                layers_current_pp_stage,
                metadata_lengths,
            )
            return -1

        layers_params = list(
            zip(
                range(layers_current_pp_stage),
                src_kv_ptrs,
                sliced_dst_kv_ptrs,
                src_kv_data_lens,
                sliced_dst_kv_data_lens,
                src_kv_item_lens,
                sliced_dst_kv_item_lens,
            )
        )
        group_descriptors = [
            {
                "group": group_id,
                "src_first_page": int(prefill_index[0]),
                "src_page_count": len(prefill_index),
                "dst_first_page": int(decode_index[0]),
                "dst_page_count": len(decode_index),
            }
            for group_id, (prefill_index, decode_index) in enumerate(
                zip(prefill_kv_blocks, dst_kv_blocks)
            )
        ]
        group_count = len(group_descriptors)
        buffer_descriptors = [
            {
                "batch_buffer_position": position,
                "buffer": int(buffer_id),
                "flat_block_start": position * group_count,
                "flat_block_end_exclusive": (position + 1) * group_count,
                "src_ptr": int(src_ptr),
                "dst_ptr": int(dst_ptr),
                "src_capacity": int(src_data_len),
                "dst_capacity": int(dst_data_len),
                "src_item_len": int(src_item_len),
                "dst_item_len": int(dst_item_len),
            }
            for position, (
                buffer_id,
                src_ptr,
                dst_ptr,
                src_data_len,
                dst_data_len,
                src_item_len,
                dst_item_len,
            ) in enumerate(layers_params)
        ]
        self._write_diag(
            "send_kvcache_plan",
            transfer_id=transfer_id,
            remote_session_id=mooncake_session_id,
            groups=group_descriptors,
            buffers=buffer_descriptors,
            group_count=group_count,
            buffer_count=len(buffer_descriptors),
            combined_block_count=group_count * len(buffer_descriptors),
        )

        def set_transfer_blocks(
            buffer_id: int,
            src_ptr: int,
            dst_ptr: int,
            src_data_len: int,
            dst_data_len: int,
            src_item_len: int,
            dst_item_len: int,
        ) -> List[Tuple[int, int, int]] | None:
            if src_item_len <= 0 or dst_item_len <= 0:
                self._write_diag(
                    "invalid_item_length",
                    transfer_id=transfer_id,
                    remote_session_id=mooncake_session_id,
                    buffer=buffer_id,
                    src_item_len=src_item_len,
                    dst_item_len=dst_item_len,
                )
                logger.error(
                    "Ascend KV buffer %d has invalid item length: src=%d, dst=%d",
                    buffer_id,
                    src_item_len,
                    dst_item_len,
                )
                return None
            if src_item_len != dst_item_len:
                self._write_diag(
                    "item_length_mismatch",
                    transfer_id=transfer_id,
                    remote_session_id=mooncake_session_id,
                    buffer=buffer_id,
                    src_item_len=src_item_len,
                    dst_item_len=dst_item_len,
                )
                logger.error(
                    "Ascend KV buffer %d layout mismatch for session %s: "
                    "src_item_len=%d, dst_item_len=%d. Prefill/decode must use "
                    "the same model, page size, KV dtype, and cache layout.",
                    buffer_id,
                    mooncake_session_id,
                    src_item_len,
                    dst_item_len,
                )
                return None

            transfer_blocks = []
            for group_id, (prefill_index, decode_index) in enumerate(
                zip(prefill_kv_blocks, dst_kv_blocks)
            ):
                src_index = int(prefill_index[0])
                dst_index = int(decode_index[0])
                length = src_item_len * len(prefill_index)
                src_offset = src_index * src_item_len
                dst_offset = dst_index * dst_item_len
                src_end = src_offset + length
                dst_end = dst_offset + length
                if src_index < 0 or src_end > src_data_len:
                    self._write_diag(
                        "source_range_out_of_bounds",
                        transfer_id=transfer_id,
                        remote_session_id=mooncake_session_id,
                        buffer=buffer_id,
                        group=group_id,
                        page=src_index,
                        pages=len(prefill_index),
                        item_len=src_item_len,
                        byte_start=src_offset,
                        byte_end=src_end,
                        capacity=src_data_len,
                    )
                    logger.error(
                        "Ascend KV source range out of bounds before transfer: "
                        "session=%s, buffer=%d, group=%d, page=%d, pages=%d, "
                        "item_len=%d, byte_range=[%d,%d), capacity=%d",
                        mooncake_session_id,
                        buffer_id,
                        group_id,
                        src_index,
                        len(prefill_index),
                        src_item_len,
                        src_offset,
                        src_end,
                        src_data_len,
                    )
                    return None
                if dst_index < 0 or (dst_data_len and dst_end > dst_data_len):
                    self._write_diag(
                        "destination_range_out_of_bounds",
                        transfer_id=transfer_id,
                        remote_session_id=mooncake_session_id,
                        buffer=buffer_id,
                        group=group_id,
                        page=dst_index,
                        pages=len(decode_index),
                        item_len=dst_item_len,
                        byte_start=dst_offset,
                        byte_end=dst_end,
                        capacity=dst_data_len,
                    )
                    logger.error(
                        "Ascend KV destination range out of bounds before transfer: "
                        "session=%s, buffer=%d, group=%d, page=%d, pages=%d, "
                        "item_len=%d, byte_range=[%d,%d), capacity=%d",
                        mooncake_session_id,
                        buffer_id,
                        group_id,
                        dst_index,
                        len(decode_index),
                        dst_item_len,
                        dst_offset,
                        dst_end,
                        dst_data_len,
                    )
                    return None
                src_addr = src_ptr + src_offset
                dst_addr = dst_ptr + dst_offset
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(layer_params) -> int:
            transfer_blocks = set_transfer_blocks(*layer_params)
            if transfer_blocks is None:
                return -1
            buffer_id = int(layer_params[0])
            self._write_diag(
                "memfabric_transfer_start",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                mode="single_buffer",
                buffer=buffer_id,
                block_count=len(transfer_blocks),
            )
            try:
                ret = self._transfer_data(mooncake_session_id, transfer_blocks)
            except Exception as exc:
                self._write_diag(
                    "memfabric_transfer_exception",
                    transfer_id=transfer_id,
                    remote_session_id=mooncake_session_id,
                    mode="single_buffer",
                    buffer=buffer_id,
                    block_count=len(transfer_blocks),
                    exception=repr(exc),
                )
                raise
            self._write_diag(
                "memfabric_transfer_result",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                mode="single_buffer",
                buffer=buffer_id,
                block_count=len(transfer_blocks),
                ret=int(ret),
            )
            return ret

        # Worker function for processing all layers in a batch
        def process_layers(layers_params) -> int:
            transfer_blocks = []
            for layer_params in layers_params:
                layer_blocks = set_transfer_blocks(*layer_params)
                if layer_blocks is None:
                    return -1
                transfer_blocks.extend(layer_blocks)
            self._write_diag(
                "memfabric_transfer_start",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                mode="combined",
                buffer_count=len(layers_params),
                block_count=len(transfer_blocks),
            )
            try:
                ret = self._transfer_data(mooncake_session_id, transfer_blocks)
            except Exception as exc:
                self._write_diag(
                    "memfabric_transfer_exception",
                    transfer_id=transfer_id,
                    remote_session_id=mooncake_session_id,
                    mode="combined",
                    buffer_count=len(layers_params),
                    block_count=len(transfer_blocks),
                    exception=repr(exc),
                )
                raise
            self._write_diag(
                "memfabric_transfer_result",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                mode="combined",
                buffer_count=len(layers_params),
                block_count=len(transfer_blocks),
                ret=int(ret),
            )
            return ret

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(process_layer, layer_params)
                for layer_params in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    self._write_diag(
                        "send_kvcache_result",
                        transfer_id=transfer_id,
                        remote_session_id=mooncake_session_id,
                        ret=int(status),
                    )
                    for f in futures:
                        f.cancel()
                    return status
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            ret = process_layers(layers_params)
            if ret == 0:
                self._write_diag(
                    "send_kvcache_result",
                    transfer_id=transfer_id,
                    remote_session_id=mooncake_session_id,
                    ret=0,
                )
                return 0

            self._write_diag(
                "combined_transfer_failed",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                ret=int(ret),
                buffer_count=len(layers_params),
                group_count=len(prefill_kv_blocks),
                block_count=len(layers_params) * len(prefill_kv_blocks),
            )
            logger.error(
                "Ascend combined KV batch failed: session=%s, ret=%d, "
                "buffers=%d, groups_per_buffer=%d, total_blocks=%d. "
                "Retrying each buffer sequentially to isolate a MemFabric "
                "registration or batch-size failure.",
                mooncake_session_id,
                ret,
                len(layers_params),
                len(prefill_kv_blocks),
                len(layers_params) * len(prefill_kv_blocks),
            )
            for layer_params in layers_params:
                ret = process_layer(layer_params)
                if ret != 0:
                    (
                        buffer_id,
                        _,
                        _,
                        src_data_len,
                        dst_data_len,
                        src_item_len,
                        dst_item_len,
                    ) = layer_params
                    self._write_diag(
                        "per_buffer_retry_failed",
                        transfer_id=transfer_id,
                        remote_session_id=mooncake_session_id,
                        buffer=int(buffer_id),
                        ret=int(ret),
                        group_count=len(prefill_kv_blocks),
                        src_capacity=int(src_data_len),
                        dst_capacity=int(dst_data_len),
                        src_item_len=int(src_item_len),
                        dst_item_len=int(dst_item_len),
                    )
                    logger.error(
                        "Ascend per-buffer KV retry failed: session=%s, "
                        "buffer=%d, ret=%d, groups=%d, src_capacity=%d, "
                        "dst_capacity=%d, src_item_len=%d, dst_item_len=%d",
                        mooncake_session_id,
                        buffer_id,
                        ret,
                        len(prefill_kv_blocks),
                        src_data_len,
                        dst_data_len,
                        src_item_len,
                        dst_item_len,
                    )
                    self._write_diag(
                        "send_kvcache_result",
                        transfer_id=transfer_id,
                        remote_session_id=mooncake_session_id,
                        ret=int(ret),
                    )
                    return ret
            self._write_diag(
                "combined_failed_retries_succeeded",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
            )
            self._write_diag(
                "send_kvcache_result",
                transfer_id=transfer_id,
                remote_session_id=mooncake_session_id,
                ret=0,
                recovered_by_per_buffer_retry=True,
            )
            logger.warning(
                "Ascend combined KV batch failed for session %s but all "
                "per-buffer retries succeeded; continuing the request.",
                mooncake_session_id,
            )
            return 0

        self._write_diag(
            "send_kvcache_result",
            transfer_id=transfer_id,
            remote_session_id=mooncake_session_id,
            ret=0,
        )
        return 0


class AscendKVSender(MooncakeKVSender):
    pass


class AscendKVReceiver(MooncakeKVReceiver):
    pass


class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
