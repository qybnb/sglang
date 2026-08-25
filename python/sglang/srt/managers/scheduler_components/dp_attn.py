from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import ceil_align, require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


logger = logging.getLogger(__name__)

_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()

_KIMI_K3_LOCAL_PREFILL_CP_ARCHS = frozenset(
    {
        "KimiK3ForConditionalGeneration",
        "KimiLinearForCausalLM",
    }
)


def _requires_local_prefill_cp_latch(
    server_args: ServerArgs, model_config: ModelConfig
) -> bool:
    """Whether this worker must latch PCP from the real local batch."""
    if (
        not envs.SGLANG_ENABLE_CP_V2.get()
        or not server_args.enable_prefill_cp
        or server_args.attn_cp_size <= 1
    ):
        return False

    architectures = set(
        getattr(model_config.hf_config, "architectures", None) or []
    )
    text_config = getattr(model_config, "hf_text_config", None)
    architectures.update(getattr(text_config, "architectures", None) or [])
    return bool(architectures & _KIMI_K3_LOCAL_PREFILL_CP_ARCHS)


def _local_prefill_cp_candidate(
    local_batch: Optional[ScheduleBatch],
    num_tokens: int,
    attn_tp_size: int,
) -> bool:
    """Evaluate PCP from the real batch before DP padding fabricates idle work."""
    if local_batch is None or num_tokens <= 0:
        return False

    from sglang.srt.layers.cp.utils import can_cp_v2_apply
    # The ragged CP-v2 path aligns each CP rank's physical rows independently
    # for attention TP.  Do not add a CP-global alignment here: doing so would
    # couple otherwise independent attention-DP replicas through the shared
    # global token table.
    planned_num_tokens = ceil_align(num_tokens, attn_tp_size)
    return can_cp_v2_apply(local_batch, num_tokens=planned_num_tokens)


def _maybe_log_local_prefill_cp_decision(
    *,
    enabled: bool,
    local_batch: Optional[ScheduleBatch],
    local_prefill_cp_active: Optional[bool],
    num_tokens: int,
    attn_cp_size: int,
    attn_dp_rank: int,
) -> None:
    """Emit exactly one PCP decision line for one real local DP batch.

    The caller selects one attention-TP/CP leader per attention-DP replica.
    Keep all diagnostic list construction behind ``enabled`` so production
    runs pay no per-batch logging or statistics overhead when the environment
    variable is unset.
    """
    if not enabled:
        return

    if local_batch is None:
        mode = "NONE"
        num_reqs = 0
        extend_lens = []
        reason = "no_local_batch"
    else:
        forward_mode = getattr(local_batch, "forward_mode", None)
        mode = getattr(forward_mode, "name", str(forward_mode))
        num_reqs = local_batch.batch_size()
        raw_extend_lens = getattr(local_batch, "extend_lens", None)
        extend_lens = (
            [int(length) for length in raw_extend_lens]
            if raw_extend_lens is not None
            else []
        )

        if local_prefill_cp_active:
            reason = "eligible"
        elif forward_mode is None:
            reason = "missing_forward_mode"
        elif not forward_mode.is_context_parallel_extend():
            reason = f"mode_{mode.lower()}"
        elif forward_mode.is_mixed():
            reason = "mixed_batch"
        elif extend_lens and min(extend_lens) < 2 * attn_cp_size:
            reason = (
                f"short_extend_min_{min(extend_lens)}"
                f"_required_{2 * attn_cp_size}"
            )
        else:
            reason = "cp_strategy_ineligible"

    logger.info(
        "[KIMI_K3_PCP_BATCH] dp_rank=%d mode=%s num_reqs=%d "
        "num_tokens=%d extend_lens=%s local_pcp_active=%s reason=%s",
        attn_dp_rank,
        mode,
        num_reqs,
        num_tokens,
        extend_lens,
        local_prefill_cp_active,
        reason,
    )


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int
    can_run_breakable_cuda_graph: bool
    local_prefill_cp_active: Optional[bool]

    # some gathered elements
    tp0_info: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_breakable_cuda_graph),
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_breakable_cuda_graph
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, 7),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks

        # Set fallback values for inactive ranks
        tp_info = global_info_tensor.view(
            self.dp_size * self.tp_size * self.cp_size, 7
        )
        tp_info[tp_active_ranks == 0] = self._get_fallback_tensor(device=device)

        tp0_info = global_info_tensor[:, 0, :]
        self.tp0_info = tp0_info
        # Perform only one Device-to-Host (D2H) memory copy
        cpu_data = tp0_info[:, :2].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        self.can_cuda_graph = bool(tp0_info[:, 2].min().item())
        self.is_extend_in_batch = bool(tp0_info[:, 3].max().item())
        self.can_run_breakable_cuda_graph = bool(tp0_info[:, 6].min().item())
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph
    batch.can_run_dp_breakable_cuda_graph = mlp_sync_info.can_run_breakable_cuda_graph
    batch.local_prefill_cp_active = mlp_sync_info.local_prefill_cp_active


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    latch_local_prefill_cp: bool = False,
    log_local_prefill_cp_decision: bool = False,
    attn_dp_rank: int = -1,
):
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph
    can_run_breakable_cuda_graph = (
        local_batch is not None
        and local_batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
        and not disable_cuda_graph
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    if len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    local_prefill_cp_active = (
        _local_prefill_cp_candidate(local_batch, num_tokens, attn_tp_size)
        if latch_local_prefill_cp
        else None
    )
    _maybe_log_local_prefill_cp_decision(
        enabled=log_local_prefill_cp_decision,
        local_batch=local_batch,
        local_prefill_cp_active=local_prefill_cp_active,
        num_tokens=num_tokens,
        attn_cp_size=attn_cp_size,
        attn_dp_rank=attn_dp_rank,
    )

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        can_run_breakable_cuda_graph=can_run_breakable_cuda_graph,
        local_prefill_cp_active=local_prefill_cp_active,
    )

    if not skip_all_gather:
        mlp_sync_info.all_gather(device=device, group=group)

        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1
        need_idle_batch = dp_size > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    tp_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    offload_tags: set[str]
    ps: ParallelState
    server_args: ServerArgs
    model_config: ModelConfig
    enable_overlap: bool
    spec_algorithm: SpeculativeAlgorithm
    get_require_mlp_sync: Callable[[], bool]

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
            latch_local_prefill_cp=(
                _requires_local_prefill_cp_latch(
                    self.server_args, self.model_config
                )
            ),
            log_local_prefill_cp_decision=(
                envs.SGLANG_KIMI_K3_LOG_PCP_EVERY_BATCH.get()
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            attn_dp_rank=self.ps.attn_dp_rank,
        )

    def maybe_prepare_mlp_sync_batch(
        self,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.get_require_mlp_sync() for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.get_require_mlp_sync():
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
