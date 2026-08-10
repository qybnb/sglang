import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.ascend.diagnostics import write_ascend_kv_diag
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/backend/pd_disaggregation.md"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        self.session_id = NetworkAddress(
            self.hostname, self.engine.get_rpc_port()
        ).to_host_port_str()
        write_ascend_kv_diag(
            "engine_created",
            role=self.role,
            hostname=self.hostname,
            npu_id=self.npu_id,
            session_id=self.session_id,
            store_url=self.store_url,
        )
        self.initialize()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            trans_op_type = TransferEngine.TransDataOpType.DEVICE_RDMA
            """with device RDMA for PD transfer"""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        protocol_name = transfer_protocol or "sdma"
        write_ascend_kv_diag(
            "engine_initialize_start",
            role=self.role,
            npu_id=self.npu_id,
            session_id=self.session_id,
            store_url=self.store_url,
            protocol=protocol_name,
        )
        try:
            ret_value = self.engine.initialize(
                self.store_url, self.session_id, self.role, self.npu_id, trans_op_type
            )
        except Exception as exc:
            write_ascend_kv_diag(
                "engine_initialize_exception",
                role=self.role,
                npu_id=self.npu_id,
                session_id=self.session_id,
                exception=repr(exc),
            )
            raise
        write_ascend_kv_diag(
            "engine_initialize_result",
            role=self.role,
            npu_id=self.npu_id,
            session_id=self.session_id,
            protocol=protocol_name,
            ret=int(ret_value),
        )
        if ret_value != 0:
            raise RuntimeError(
                "Ascend Transfer Engine initialization failed: "
                f"ret={ret_value}, role={self.role}, store_url={self.store_url}, "
                f"session_id={self.session_id}, npu_id={self.npu_id}, "
                f"protocol={transfer_protocol or 'sdma'}"
            )

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        registration = [
            {"buffer": index, "ptr": int(ptr), "capacity": int(length)}
            for index, (ptr, length) in enumerate(zip(ptrs, lengths))
        ]
        write_ascend_kv_diag(
            "batch_register_start",
            role=self.role,
            npu_id=self.npu_id,
            session_id=self.session_id,
            buffers=registration,
            ptr_count=len(ptrs),
            length_count=len(lengths),
            total_bytes=sum(lengths),
        )
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception as exc:
            write_ascend_kv_diag(
                "batch_register_exception",
                role=self.role,
                npu_id=self.npu_id,
                session_id=self.session_id,
                buffers=registration,
                exception=repr(exc),
            )
            raise RuntimeError(
                "Ascend memory registration raised an exception: "
                f"buffers={len(ptrs)}, total_bytes={sum(lengths)}"
            ) from exc
        write_ascend_kv_diag(
            "batch_register_result",
            role=self.role,
            npu_id=self.npu_id,
            session_id=self.session_id,
            buffers=registration,
            ret=int(ret_value),
        )
        if ret_value != 0:
            raise RuntimeError(
                "Ascend memory registration failed: "
                f"ret={ret_value}, buffers={len(ptrs)}, "
                f"total_bytes={sum(lengths)}"
            )

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "sdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
