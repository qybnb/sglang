"""Ascend FuseEP fused dispatch+GEMM+combine forward path.

Follows the mega_moe shape: a free-function bypass invoked from
``FusedMoE.forward`` when ``--moe-a2a-backend ascend_fuseep`` is set, plus a
weight-postprocess helper that NPU quant_methods call from their
``process_weights_after_loading`` when the same backend is selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import FusedMoEMode, npu_format_cast
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import DeepEPMode

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16


def _get_fuseep_buffer(layer: FusedMoE):
    DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_tp_group().device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.LOW_LATENCY,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        layer.num_experts,
    )


def forward_fuseep(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor:
    fuse_mode = envs.SGLANG_NPU_FUSED_MOE_MODE.get()
    if fuse_mode == FusedMoEMode.MEGA_MOE.value:
        # print(f"xxxxx",flush=True)
        # assert layer.weights.w1_scale is not None
        # assert layer.weights.w2_scale is not None

        def to_list(x):
            return x if isinstance(x, list) else [x]

        weight1 = to_list(layer.w13_weight)
        weight2 = to_list(layer.w2_weight)

        weight_scales1 = to_list(layer.w13_weight_scale)
        weight_scales2 = to_list(layer.w2_weight_scale)

        weight_scales1 = [t.squeeze(0) if (t.dim() == 2 and t.shape[0] == 1) else t for t in weight_scales1]
        weight_scales2 = [t.squeeze(0) if (t.dim() == 2 and t.shape[0] == 1) else t for t in weight_scales2]

        # l1_bias = layer.w13_weight_scale
        # l2_bias = layer.w13_weight_scale
        # deepep
        # import deepep.bufferf
        buf = _get_fuseep_buffer(layer)
        # print(f"================= {weight1.shape=} {weight1.dtype=}", flush=True)
        expert_per_rank =max(1,layer.num_experts//int(get_moe_expert_parallel_world_size()))
        num_max_dispatch_tokens_per_rank = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        # There is a trade-off between memory and performance, which can be configured here via settings.
        #max_recv_token_num = max(1, num_max_dispatch_tokens_per_rank * int(get_moe_expert_parallel_world_size()) * min(layer.top_k, expert_per_rank))
        max_recv_token_num = 131072
        out, _ = buf.fused_deep_moe(
            x=hidden_states,
            topk_idx=topk_output.topk_ids.to(torch.int32),
            topk_weights=topk_output.topk_weights.to(torch.float32),
            gmm1_permuted_weight=weight1,
            gmm1_permuted_weight_scale=weight_scales1,
            gmm2_weight=weight2,
            gmm2_weight_scale=weight_scales2,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            backend="mega_moe",
            activation="situ",
            beta=4.0,
            linear_beta=25.0,
            l1_bias=layer.w13_scale_bias,
            l2_bias=layer.w2_scale_bias,
            num_experts=layer.num_experts,
            max_recv_token_num=60000,
            # x_active_mask=x_active_mask,
            # activation_clamp=activation_clamp,
            # weight1_type=layer._mega_moe_weight_type,
            # weight2_type=layer._mega_moe_weight_type,
        )
        return out
    else:
        buf = _get_fuseep_buffer(layer)
        hidden_states, _ = buf.fused_deep_moe(
            hidden_states,
            topk_idx=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            gmm1_permuted_weight=layer.w13_weight,
            gmm1_permuted_weight_scale=layer.w13_weight_scale,
            gmm2_weight=layer.w2_weight,
            gmm2_weight_scale=layer.w2_weight_scale,
            num_max_dispatch_tokens_per_rank=(
                envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
            ),
            num_experts=layer.num_experts,
            fuse_mode=fuse_mode,
        )
        return hidden_states


def _permute_w13_weight_scale(w: torch.Tensor, tile_n: int) -> torch.Tensor:
    if tile_n % 2 != 0:
        raise ValueError(f"tile_n must be even, got {tile_n}")

    *dims, n = w.shape
    if n % tile_n != 0:
        raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

    w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)
    perm_order = list(range(len(dims))) + [-2, -3, -1]
    return w_reshaped.permute(perm_order).reshape(*dims, n)


def _reshape_w13_weight(
    weight: torch.Tensor, dim: int, chunk_size: int = 64
) -> torch.Tensor:
    # Achieving greater computing power through reshape on Ascend.
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)

    if original_shape[dim] % (2 * chunk_size) != 0:
        raise ValueError(
            f"Dimension {dim} size {original_shape[dim]} must be divisible by "
            f"{2 * chunk_size}"
        )

    new_shape = (
        *original_shape[:dim],
        2,
        original_shape[dim] // (2 * chunk_size),
        chunk_size,
        *original_shape[dim + 1 :],
    )

    weight = weight.view(new_shape)
    weight = weight.transpose(dim, dim + 1).contiguous()
    return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])


def _release_weight_cache(weight: torch.Tensor) -> torch.Tensor:
    # .contiguous() introduces additional memory overhead; release with resize_(0)
    origin_weight = weight.data.transpose(1, 2)
    new_weight = origin_weight.contiguous()
    origin_weight.untyped_storage().resize_(0)
    return new_weight


def _scale_from_float_to_int64(scale: torch.Tensor) -> torch.nn.Parameter:
    import numpy as np

    converted = torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).to(scale.device)
    return torch.nn.Parameter(converted, requires_grad=False)


def _force_release_storage(src: torch.Tensor, dependents: list) -> None:
    """Forcibly resize ``src``'s underlying storage to 0 to return memory to
    the NPU caching allocator immediately.

    This is only safe when no tensor in ``dependents`` shares storage with
    ``src``. We guard with a ``data_ptr`` overlap check: if any dependent
    overlaps, we fall back to ``del`` + ``empty_cache`` so the fallback paths
    of ``npu_format_cast`` (which return the input as-is) are not corrupted.
    """
    try:
        src_storage = src.untyped_storage()
        src_ptr = src_storage.data_ptr()
        src_size = src_storage.nbytes()
        if src_size == 0:
            return
        for d in dependents:
            d_storage = d.untyped_storage()
            d_ptr = d_storage.data_ptr()
            d_size = d_storage.nbytes()
            if d_size > 0 and not (
                d_ptr + d_size <= src_ptr or src_ptr + src_size <= d_ptr
            ):
                # Storage overlaps with a dependent; cannot resize safely.
                return
        src_storage.resize_(0)
    except Exception:
        # If any introspection fails, leave the storage untouched rather than
        # risk corrupting the freshly built lists.
        return


def process_fuseep_weights(layer: torch.nn.Module) -> None:
    """Apply the Ascend FuseEP-specific weight layout.

    Replaces NPU quant_method weight layouts with the form required by the
    fused_deep_moe op. Invoked from NPU ``process_weights_after_loading``
    when ``--moe-a2a-backend ascend_fuseep`` is set.
    """
    if envs.SGLANG_NPU_FUSED_MOE_MODE.get() == FusedMoEMode.DISPATCH_FFN_COMBINE.value:
        w13_weight = _release_weight_cache(layer.w13_weight)
        layer.w13_weight.data = npu_format_cast(w13_weight)
        w2_weight = _release_weight_cache(layer.w2_weight)
        layer.w2_weight.data = npu_format_cast(w2_weight)

        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
            layer.w13_weight_scale.data.shape[0], -1
        )
        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )

        layer.w13_weight_scale = _scale_from_float_to_int64(layer.w13_weight_scale.data)
        layer.w2_weight_scale = _scale_from_float_to_int64(layer.w2_weight_scale.data)
    elif envs.SGLANG_NPU_FUSED_MOE_MODE.get() == FusedMoEMode.FUSED_DEEP_MOE.value:
        cpu_w13 = layer.w13_weight.data.transpose(1, 2).cpu()
        layer.w13_weight.data = _reshape_w13_weight(cpu_w13, -1).npu()
        w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
        w13_scale = _permute_w13_weight_scale(w13_scale, 128)
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale.to(torch.float32), requires_grad=False
        )
        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)

        w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale.to(torch.float32), requires_grad=False
        )
    else:
        if not hasattr(layer, "w13_scale_bias"):
            raise RuntimeError(
                "MegaMoe only support W4A8 INT on A2/A3 for weight with w1 scale bias and w2 scale bias."
                "Try to disable MegaMoe to avoid this error."
            )

        # Build each list then release the original tensor immediately so that
        # the large source weights do not coexist with all new lists, keeping
        # peak memory roughly at one copy instead of two. ``_force_release_storage``
        # uses ``storage.resize_(0)`` to return memory to the NPU caching
        # allocator right away (guarded by a data_ptr overlap check so the
        # fallback paths of ``npu_format_cast`` stay safe).
        '''
        def _release_weight_cache_without_trans(weight: torch.Tensor) -> torch.Tensor:
            casted_weight = npu_format_cast(weight.clone()).view(torch.int32)
            #weight.untyped_storage().resize_(0)
            return casted_weight
        w13_weight_src = layer.w13_weight.data
        w13_weight_list = [
            _release_weight_cache_without_trans(weight)
            for weight in w13_weight_src.unbind(dim=0)
        ]

        del layer.w13_weight

        w2_weight_src = layer.w2_weight.data
        w2_weight_list = [
            _release_weight_cache_without_trans(weight)
            for weight in w2_weight_src.unbind(dim=0)
        ]

        del layer.w2_weight

        w13_weight_scale_src = layer.w13_weight_scale.data
        w13_weight_scale_list = [
            t.reshape(-1).view(torch.uint64)
            for t in w13_weight_scale_src.unbind(dim=0)
        ]
        del layer.w13_weight_scale

        w2_weight_scale_src = layer.w2_weight_scale.data
        w2_weight_scale_list = [
            t.reshape(-1).view(torch.uint64)
            for t in w2_weight_scale_src.unbind(dim=0)
        ]
        del layer.w2_weight_scale

        w13_scale_bias_src = layer.w13_scale_bias.data
        w13_scale_bias_list = [
            t.reshape(-1).to(torch.float32)
            for t in w13_scale_bias_src.unbind(dim=0)
        ]
        del layer.w13_scale_bias

        w2_scale_bias_src = layer.w2_scale_bias.data
        w2_scale_bias_list = [
            t.reshape(-1).to(torch.float32)
            for t in w2_scale_bias_src.unbind(dim=0)
        ]
        _force_release_storage(w2_scale_bias_src, w2_scale_bias_list)
        del layer.w2_scale_bias

        layer.w13_weight = w13_weight_list
        layer.w13_weight_scale = w13_weight_scale_list
        layer.w2_weight = w2_weight_list
        layer.w2_weight_scale = w2_weight_scale_list
        layer.w13_scale_bias = w13_scale_bias_list
        layer.w2_scale_bias = w2_scale_bias_list
        
           # print(f"=================209 ======= ", flush=True)
        layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
        layer.cann_mega_moe_w13_weight_list = [weight.clone().view(torch.int32) for weight in layer.w13_weight.data.unbind(dim=0)]
        layer.cann_mega_moe_w2_weight_list = [weight.clone().view(torch.int32) for weight in layer.w2_weight.data.unbind(dim=0)]

        layer.cann_mega_moe_w13_weight_scale_list = [t.reshape(-1) for t in layer.w13_weight_scale.data.unbind(dim=0)]
        layer.cann_mega_moe_w2_weight_scale_list = [t.reshape(-1) for t in layer.w2_weight_scale.data.unbind(dim=0)]
        if not hasattr(layer, "w13_scale_bias"):
            raise RuntimeError(
                "MegaMoe only support W4A8 INT on A2/A3 for weight with w1 scale bias and w2 scale bias."
                "Try to disable MegaMoe to avoid this error."
            )
        layer.cann_mega_moe_w13_scale_bias_list = [t.reshape(-1) for t in layer.w13_scale_bias.data.unbind(dim=0)]
        layer.cann_mega_moe_w2_scale_bias_list = [t.reshape(-1) for t in layer.w2_scale_bias.data.unbind(dim=0)]
        del layer.w13_weight
        del layer.w2_weight
        del layer.w13_weight_scale
        del layer.w2_weight_scale
        del layer.w13_scale_bias
        del layer.w2_scale_bias
        layer.w13_weight = layer.cann_mega_moe_w13_weight_list
        layer.w13_weight_scale = layer.cann_mega_moe_w13_weight_scale_list
        layer.w2_weight = layer.cann_mega_moe_w2_weight_list
        layer.w2_weight_scale = layer.cann_mega_moe_w2_weight_scale_list

        def cast_bias_to_fp32(bias):
            lst = bias if isinstance(bias, list) else [bias]
            return [t if t.dtype == torch.float32 else t.to(torch.float32) for t in lst]

        layer.w13_scale_bias = cast_bias_to_fp32(layer.cann_mega_moe_w13_scale_bias_list)
        layer.w2_scale_bias = cast_bias_to_fp32(layer.cann_mega_moe_w2_scale_bias_list)
        '''
        layer.cann_mega_moe_w13_weight_list = [
            npu_format_cast(weight.clone()).view(torch.int32)
            for weight in layer.w13_weight.data.unbind(dim=0)
        ]

        layer.cann_mega_moe_w2_weight_list = [
            npu_format_cast(weight.clone()).view(torch.int32)
            for weight in layer.w2_weight.data.unbind(dim=0)
        ]

        layer.cann_mega_moe_w13_weight_scale_list = [
            t.reshape(-1).view(torch.uint64)
            for t in layer.w13_weight_scale.data.unbind(dim=0)
        ]
        layer.cann_mega_moe_w2_weight_scale_list = [
            t.reshape(-1).view(torch.uint64)
            for t in layer.w2_weight_scale.data.unbind(dim=0)
        ]

        if not hasattr(layer, "w13_scale_bias"):
            raise RuntimeError(
                "MegaMoe only support W4A8 INT on A2/A3 for weight with w1 scale bias and w2 scale bias."
                "Try to disable MegaMoe to avoid this error."
            )
        layer.cann_mega_moe_w13_scale_bias_list = [
            t.reshape(-1).to(torch.float32)
            for t in layer.w13_scale_bias.data.unbind(dim=0)
        ]
        layer.cann_mega_moe_w2_scale_bias_list = [
            t.reshape(-1).to(torch.float32)
            for t in layer.w2_scale_bias.data.unbind(dim=0)
        ]

        del layer.w13_weight
        del layer.w2_weight
        del layer.w13_weight_scale
        del layer.w2_weight_scale
        del layer.w13_scale_bias
        del layer.w2_scale_bias

        layer.w13_weight = layer.cann_mega_moe_w13_weight_list
        layer.w13_weight_scale = layer.cann_mega_moe_w13_weight_scale_list
        layer.w2_weight = layer.cann_mega_moe_w2_weight_list
        layer.w2_weight_scale = layer.cann_mega_moe_w2_weight_scale_list
        layer.w13_scale_bias = layer.cann_mega_moe_w13_scale_bias_list
        layer.w2_scale_bias = layer.cann_mega_moe_w2_scale_bias_list
    if hasattr(layer, "w13_weight_offset"):
        layer.w13_weight_offset = torch.nn.Parameter(
            layer.w13_weight_offset.data.squeeze(-1).contiguous(),
            requires_grad=False,
        )
    if hasattr(layer, "w2_weight_offset"):
        layer.w2_weight_offset = torch.nn.Parameter(
            layer.w2_weight_offset.data.squeeze(-1).contiguous(),
            requires_grad=False,
        )
