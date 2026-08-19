from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk_delta_h import (
    CHUNK_SIZE as KDA_CHUNK_SIZE,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kda_cp import (
    all_gather_cp_heads,
    build_kda_fla_cp_context,
    head_to_sequence_a2a,
    kda_use_fla_prefill_cp,
    kda_use_prefill_cp,
    prepare_kda_cp_conv_states,
    sequence_to_head_a2a,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

# Ascend computes the convolution window in the weight dtype and assigns the
# updated window back into the persistent cache, with PyTorch casting to the
# cache dtype on assignment.
if is_npu():
    from sglang.srt.hardware_backend.npu.kernels.causal_conv1d_verify import (
        causal_conv1d_linear_verify_npu,
    )
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
        torch_causal_conv1d_update_npu,
    )

    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel


def _npu_causal_conv1d_linear_verify(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
) -> torch.Tensor:
    """Run linear-chain verify conv while preserving every rollback window.

    The Ascend causal-conv wrapper does not expose the tree/scratch arguments
    supported by the Triton implementation. DSpark verification is a linear
    chain, so advancing one token at a time is equivalent and lets us save the
    post-token window required by speculative state commit.
    """
    batch_size, _, num_tokens = x.shape
    cache_indices = conv_state_indices[:batch_size]
    scratch_indices = intermediate_state_indices[:batch_size]
    state = conv_state.index_select(0, cache_indices)

    if state.shape[-1] + 1 != weight.shape[-1]:
        raise ValueError(
            "KDA NPU verify conv window mismatch: "
            f"state={state.shape[-1]}, weight={weight.shape[-1]}"
        )

    outputs = []
    for step in range(num_tokens):
        window = torch.cat((state, x[:, :, step : step + 1]), dim=-1)
        out = torch.sum(window.to(weight.dtype) * weight.unsqueeze(0), dim=-1)
        if bias is not None:
            out = out + bias
        out = torch.nn.functional.silu(out).to(x.dtype)
        outputs.append(out.unsqueeze(-1))

        state = window[:, :, 1:].to(conv_state.dtype)
        intermediate_conv_window[:, step].index_copy_(
            0, scratch_indices, state.to(intermediate_conv_window.dtype)
        )

    conv_state.index_copy_(0, cache_indices, state)
    return torch.cat(outputs, dim=-1)


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashkda():
            from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
                FlashKDAKernel,
            )

            self.extend_kernel = FlashKDAKernel()
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            cutedsl_kernel = CuteDSLKDAKernel()
            if getattr(cutedsl_kernel, "supports_prefill", False):
                # SM100 chunk prefill pipeline.
                self.extend_kernel = cutedsl_kernel
            else:
                # CuTe DSL prefill kernels need SM100 (Blackwell); on older GPUs
                # fall back to the Triton chunk kernel.
                self.extend_kernel = triton_kernel
                rank0_log(
                    "KDA cutedsl prefill needs SM100; falling back to Triton extend."
                )
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton', 'flashkda', or 'cutedsl' "
                "(cutedsl prefill needs SM100)."
            )

        # K3 DSpark target verify always uses the rollback-capable Triton
        # kernel, independently from the decode/prefill backend.
        self.verify_kernel = triton_kernel
        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if the decode
        kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            lower_bound=lower_bound,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


from torch.nn import functional as F


def causal_conv1d_fn_native(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
        if out.ndim == 3:
            out = out.squeeze(0)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn_npu_old(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = -1,
    **kwargs,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref_b = []
    assert query_start_loc[-1] <= x.shape[-1], f"{query_start_loc=}, {x.shape=}"
    for i in range(query_start_loc.numel() - 1):
        out_ref_b.append(
            causal_conv1d_fn_native(
                x[..., query_start_loc[i] : query_start_loc[i + 1]],
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]].unsqueeze(0),
                initial_states=(
                    conv_states[cache_indices[i]].unsqueeze(0)
                    if has_initial_state[i]
                    else None
                ),
            )
        )
    out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=-1)
    if x.shape[-1] > query_start_loc[-1]:
        pad_seqlen = x.shape[-1] - query_start_loc[-1]
        out_ref_tensor = torch.cat(
            [
                out_ref_tensor,
                out_ref_tensor.new_zeros([*out_ref_tensor.shape[:-1], pad_seqlen]),
            ],
            dim=-1,
        )
    return out_ref_tensor


def ragged_verify_dense_scatter_indices(
    *, query_start_loc: torch.Tensor, seq_len: int, draft_token_num: int
) -> torch.Tensor:
    batch_size = query_start_loc.shape[0] - 1
    token_pos = torch.arange(seq_len, device=query_start_loc.device, dtype=torch.int32)
    token_slots = torch.searchsorted(query_start_loc[1:], token_pos, right=True)
    return (
        token_slots * draft_token_num
        + (token_pos - query_start_loc[token_slots]).to(torch.int64)
    ).clamp_(max=batch_size * draft_token_num)


def build_dspark_verify_scratch_indices(
    *, num_rows: int, scratch_slots: int, device: torch.device
) -> torch.Tensor:
    """Map verify rows to speculative scratch, reserving the last slot as pad.

    DP-attention graph buckets are expressed in global batch sizes, while each
    DP worker allocates speculative Mamba scratch for only its local request
    capacity.  Consequently a graph bucket can have more rows than locally
    usable scratch slots (for example global BS4 with DP4 has one real row per
    worker).  The extra graph rows are padding and may safely share the final
    sentinel slot; real rows remain mapped one-to-one from zero.

    The returned tensor is deliberately precomputed before graph capture so
    the target-verify kernels see a stable pointer during capture and replay.
    """
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}")
    if scratch_slots < 1:
        raise ValueError(f"scratch_slots must be positive, got {scratch_slots}")
    return torch.arange(num_rows, dtype=torch.int32, device=device).clamp_max_(
        scratch_slots - 1
    )


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        conv_states = model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
        # The shared prefix-cache tracker expects the convolution window to be
        # the last dimension. NPU already stores KDA conv state as
        # [layer, pool, channels, window]; other devices use
        # [layer, pool, window, channels].
        self.conv_states_shape = (
            conv_states.shape
            if is_npu()
            else torch.Size(
                (
                    *conv_states.shape[:-2],
                    conv_states.shape[-1],
                    conv_states.shape[-2],
                )
            )
        )
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)
        self._dspark_target_verify = model_runner.spec_algorithm.is_dspark()
        self.verify_intermediate_state_indices = self._make_verify_scratch_indices(
            self.req_to_token_pool.size
        )
        self._kda_fla_cp_logged = False

    def _make_verify_scratch_indices(self, num_rows: int) -> torch.Tensor:
        if not self._dspark_target_verify:
            return torch.arange(num_rows, dtype=torch.int32, device=self.device)
        speculative_cache = (
            self.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )
        scratch_slots = speculative_cache.intermediate_ssm.shape[1]
        return build_dspark_verify_scratch_indices(
            num_rows=num_rows,
            scratch_slots=scratch_slots,
            device=self.device,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        if not self._dspark_target_verify:
            return

        # Keep eager capacity when it is larger, but extend the stable index
        # template to every captured graph row.  Rows beyond local speculative
        # capacity map to the scratch pool's final sentinel slot.
        num_rows = max(max_bs, self.verify_intermediate_state_indices.numel())
        self.verify_intermediate_state_indices = self._make_verify_scratch_indices(
            num_rows
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            track_mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[
                0
            ]
            self.forward_metadata.conv_states_mask_indices = (
                self.forward_metadata.mamba_track_indices[track_mask_indices]
            )

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """Select the exact KDA recurrent state at each cache boundary.

        KDA's chunk kernel stores ``h[i]`` at the start of chunk ``i`` and
        writes the state after the complete extend into ``ssm_states``.  A
        cache boundary can be chunk-aligned while still preceding the end of
        the extend (for example, cache token 192 from a 196-token prefill).
        Such a boundary must use ``h[3]``, not the final state at token 196.
        """
        chunk_size = KDA_CHUNK_SIZE
        track_mask = forward_batch.mamba_track_mask.cpu()
        extend_lens = forward_batch.extend_seq_lens.cpu()
        track_indices = forward_batch.mamba_track_indices.cpu()
        cache_indices = mamba_cache_indices.cpu()
        track_lens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        num_h_states = (extend_lens - 1) // chunk_size + 1
        h_offsets = torch.zeros_like(num_h_states)
        h_offsets[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        lens_masked = (track_lens - prefix_lens)[track_mask]
        extend_masked = extend_lens[track_mask]
        offsets_masked = h_offsets[track_mask]
        dst_masked = track_indices[track_mask]

        # Only the actual end of this extend is represented by the in-place
        # final state. Every earlier cache boundary is represented by h.
        use_final = lens_masked == extend_masked
        final_src = cache_indices[track_mask][use_final]
        final_dst = dst_masked[use_final]

        use_h = ~use_final
        h_src = offsets_masked[use_h] + lens_masked[use_h] // chunk_size
        h_dst = dst_masked[use_h]

        return (
            h_src.to(self.device, non_blocking=True),
            h_dst.to(self.device, non_blocking=True),
            final_src.to(self.device, non_blocking=True),
            final_dst.to(self.device, non_blocking=True),
        )

    @staticmethod
    def _channel_first_conv_states(conv_states: torch.Tensor) -> torch.Tensor:
        """Return convolution state in [pool, channels, window] layout.

        The NPU memory pool already reverses the generic Kimi state shape to
        this layout for the Ascend causal-conv kernels. Other platforms keep
        the generic [pool, window, channels] layout and need a transpose.
        """
        return conv_states if is_npu() else conv_states.transpose(-1, -2)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # ReplaySSM ring: per-layer ring slices + the once-per-forward per-row
        # write cursor. All None unless --enable-linear-replayssm, so packed_decode
        # falls through to the byte-identical legacy KDA path. KDA ships WITHOUT
        # radix coordination for now, so force_flush is None/zeroed (the ring
        # flushes only at the natural write_pos == L-1 wrap; set in the shared
        # HybridLinearAttn metadata, which zeroes force_flush for KDA models).
        # NOTE: ReplaySSM decode is a GDN (scalar-gate) bandwidth win; on KDA the
        # per-K g_cache is K x larger and the reconstruction refolds the per-K
        # decay every step, so it is correct but SLOWER than packed (a measured
        # decode regression). Kept wired for correctness + the spec-decode path;
        # not recommended for KDA decode. Revisit on Blackwell (more tensor-core
        # throughput may flip the compute/bandwidth tradeoff).
        replayssm_write_pos = getattr(
            self.forward_metadata, "replayssm_write_pos", None
        )
        replayssm_force_flush = getattr(
            self.forward_metadata, "replayssm_force_flush", None
        )
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        conv_states = self._channel_first_conv_states(conv_states)
        if is_npu():
            # The Ascend wrapper computes the updated window in the FP32 weight
            # dtype and writes it back with advanced indexing. K3 stores the
            # persistent cache in BF16, so that assignment fails during graph
            # capture. Converting the whole cache pool to FP32 avoids the dtype
            # error but only updates a temporary tensor, leaving decode with a
            # stale prefill window. Compute on the active rows, then explicitly
            # cast and commit the updated window to the persistent cache.
            assert isinstance(mixed_qkv, torch.Tensor)
            # Slot 0 is reserved by MambaSlotAllocator as the dummy write target
            # for padded rows. Eager attention-TP padding uses -1 while Ascend
            # graph replay already maps padding to 0, so normalize both forms
            # before the ordinary PyTorch gather/scatter operations.
            safe_cache_indices = cache_indices.clamp_min(0).to(torch.int64)
            active_conv_states = conv_states.index_select(0, safe_cache_indices)
            qkv, updated_conv_states = torch_causal_conv1d_update_npu(
                mixed_qkv.unsqueeze(-1),
                active_conv_states,
                layer.conv_weights,
                bias=layer.bias,
                activation="silu",
            )
            conv_states.index_copy_(
                0, safe_cache_indices, updated_conv_states.to(conv_states.dtype)
            )
            qkv = qkv.squeeze(-1)
        else:
            qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                activation="silu",
                conv_state_indices=cache_indices,
            )

        # Skip split + reshape by consuming the packed mixed_qkv directly in a
        # single fused Triton kernel (KDA per-K gate variant of GDN PR #20627).
        #
        # The packed kernel hard-assumes one token per sequence (T=1): it has no
        # query_start_loc / per-sequence loop. forward_decode is only entered in
        # decode mode (see HybridLinearAttnBackend.forward dispatch), where each
        # request contributes exactly one token, so #tokens == #requests. Multi-
        # token-per-seq speculative paths (target_verify / draft_extend) go
        # through forward_extend instead. Assert the invariant so a future
        # routing change fails loudly rather than silently corrupting state.
        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            ret = self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return ret

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        ret = self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            lower_bound=getattr(layer, "lower_bound", None),
        )
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        # print(f"decode {torch.distributed.get_rank()=}: {layer.layer_id=}, KDA extend: q={torch.sum(q)=}, {torch.sum(conv_states)=}, {torch.sum(ret)=}", flush=True)
        return ret

    @staticmethod
    def _qkv_channels_to_heads(
        tensor: torch.Tensor,
        layer: RadixLinearAttention,
        *,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """Convert ``[..., Q|K|V channels]`` to ``[..., heads, Q|K|V dims]``."""
        if num_heads is None:
            num_heads = layer.num_q_heads
        splits = [
            num_heads * layer.head_q_dim,
            num_heads * layer.head_k_dim,
            num_heads * layer.head_v_dim,
        ]
        q, k, v = tensor.split(splits, dim=-1)
        q = q.unflatten(-1, (num_heads, layer.head_q_dim))
        k = k.unflatten(-1, (num_heads, layer.head_k_dim))
        v = v.unflatten(-1, (num_heads, layer.head_v_dim))
        if not (layer.num_q_heads == layer.num_k_heads == layer.num_v_heads):
            raise NotImplementedError(
                "KDA prefill CP currently requires equal Q/K/V head counts."
            )
        return torch.cat((q, k, v), dim=-1)

    @staticmethod
    def _heads_to_qkv_channels(
        tensor: torch.Tensor,
        layer: RadixLinearAttention,
    ) -> torch.Tensor:
        """Inverse of :meth:`_qkv_channels_to_heads` for a chosen head count."""
        q, k, v = tensor.split(
            [layer.head_q_dim, layer.head_k_dim, layer.head_v_dim], dim=-1
        )
        return torch.cat(
            (
                q.flatten(-2),
                k.flatten(-2),
                v.flatten(-2),
            ),
            dim=-1,
        )

    def _transpose_kda_cp_inputs(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        """Run the sequence->head A2A for projected QKV and KDA gates."""
        parallel = get_parallel()
        cp_size = parallel.attn_cp_size
        if layer.num_q_heads % cp_size != 0:
            raise ValueError(
                "KDA prefill CP requires local KDA heads divisible by CP size: "
                f"heads={layer.num_q_heads}, cp_size={cp_size}."
            )

        qkv_by_head = self._qkv_channels_to_heads(mixed_qkv, layer)
        qkv_by_head = sequence_to_head_a2a(qkv_by_head, forward_batch)

        if a.ndim != 4 or a.shape[0] != 1:
            raise ValueError(f"Unexpected KDA gate shape under CP: {tuple(a.shape)}")
        if b.ndim != 3 or b.shape[0] != 1:
            raise ValueError(f"Unexpected KDA beta shape under CP: {tuple(b.shape)}")
        gates = torch.cat((a.squeeze(0), b.squeeze(0).unsqueeze(-1)), dim=-1)
        gates = sequence_to_head_a2a(gates, forward_batch)

        heads_per_cp_rank = layer.num_q_heads // cp_size
        mixed_qkv = self._heads_to_qkv_channels(qkv_by_head, layer)
        a = gates[..., : layer.head_k_dim].unsqueeze(0)
        b = gates[..., layer.head_k_dim].unsqueeze(0)
        head_start = parallel.attn_cp_rank * heads_per_cp_rank
        return mixed_qkv, a, b, head_start, heads_per_cp_rank

    def _track_kda_cp_conv_state(
        self,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        num_heads: int,
    ) -> None:
        """Save a full-head convolution checkpoint from the CP head shard."""
        tracked = mixed_qkv[self.forward_metadata.track_conv_indices]
        tracked_by_head = self._qkv_channels_to_heads(
            tracked, layer, num_heads=num_heads
        )
        tracked_full = all_gather_cp_heads(tracked_by_head, head_dim=-2)
        tracked_full = self._heads_to_qkv_channels(tracked_full, layer).transpose(
            -1, -2
        )
        conv_states[self.forward_metadata.conv_states_mask_indices] = tracked_full.to(
            conv_states.dtype, copy=False
        )

    def _sync_kda_cp_final_states(
        self,
        layer: RadixLinearAttention,
        cache_indices: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        head_start: int,
        num_heads: int,
    ) -> None:
        """Replicate final conv/KDA states so existing PD transfer stays valid."""
        valid = cache_indices >= 0
        if not bool(valid.any()):
            return
        active_indices = cache_indices[valid]

        # Conv cache is channel-major [slot, Q|K|V channels, window]. Convert
        # through a per-head representation so one collective reconstructs all
        # three Q/K/V channel blocks.
        active_conv = conv_states.index_select(0, active_indices).transpose(-1, -2)
        active_conv_by_head = self._qkv_channels_to_heads(active_conv, layer)
        active_conv_local = active_conv_by_head.narrow(-2, head_start, num_heads)
        active_conv_full = all_gather_cp_heads(active_conv_local, head_dim=-2)
        active_conv_full = self._heads_to_qkv_channels(
            active_conv_full, layer
        ).transpose(-1, -2)
        conv_states.index_copy_(
            0, active_indices, active_conv_full.to(conv_states.dtype, copy=False)
        )

        active_ssm_local = (
            ssm_states.index_select(0, active_indices)
            .narrow(1, head_start, num_heads)
            .contiguous()
        )
        active_ssm_full = all_gather_cp_heads(active_ssm_local, head_dim=1)
        ssm_states.index_copy_(
            0, active_indices, active_ssm_full.to(ssm_states.dtype, copy=False)
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        if self._dspark_target_verify and forward_batch.forward_mode.is_target_verify():
            return self._forward_dspark_target_verify(
                layer, forward_batch, mixed_qkv, a, b
            )
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = self._channel_first_conv_states(mamba_cache_params.conv[0])

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0
        use_kda_cp = kda_use_prefill_cp(forward_batch)
        use_fla_cp = kda_use_fla_prefill_cp(forward_batch)
        use_kda_a2a = use_kda_cp and not use_fla_cp
        head_start = 0
        num_cp_heads = layer.num_q_heads
        fla_cp_context = None
        fla_conv_initial_states = None

        if use_fla_cp:
            fla_cp_context = build_kda_fla_cp_context(
                forward_batch, device=mixed_qkv.device
            )
            fla_conv_initial_states = prepare_kda_cp_conv_states(
                mixed_qkv,
                conv_states,
                cache_indices,
                fla_cp_context,
                has_initial_state=has_initial_state,
            )
            if not self._kda_fla_cp_logged:
                rank0_log(
                    "KDA prefill CP backend: FLA PR 691 affine-state "
                    "composition (zigzag adapter)"
                )
                self._kda_fla_cp_logged = True
        elif use_kda_a2a:
            mixed_qkv, a, b, head_start, num_cp_heads = self._transpose_kda_cp_inputs(
                layer, forward_batch, mixed_qkv, a, b
            )

        # Save the raw QKV convolution window at the last cacheable chunk
        # boundary. The shared metadata indexes flattened input tokens and the
        # destination ping-pong slots selected by MambaRadixCache.
        if self.forward_metadata.has_mamba_track_mask and not use_fla_cp:
            if use_kda_a2a:
                self._track_kda_cp_conv_state(
                    layer, mixed_qkv, conv_states, num_cp_heads
                )
            else:
                mixed_qkv_to_track = mixed_qkv[
                    self.forward_metadata.track_conv_indices
                ].transpose(-1, -2)
                conv_states[self.forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track.to(conv_states.dtype, copy=False)
                )

        splits = (
            [
                num_cp_heads * layer.head_q_dim,
                num_cp_heads * layer.head_k_dim,
                num_cp_heads * layer.head_v_dim,
            ]
            if use_kda_a2a
            else [layer.q_dim, layer.k_dim, layer.v_dim]
        )
        q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
        full_splits = [layer.q_dim, layer.k_dim, layer.v_dim]
        q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
            full_splits, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_states.split(
            full_splits, dim=-2
        )
        if layer.bias is not None:
            q_bias, k_bias, v_bias = layer.bias.split(full_splits, dim=0)
        else:
            q_bias, k_bias, v_bias = None, None, None

        if use_kda_a2a:
            q_channel_start = head_start * layer.head_q_dim
            k_channel_start = head_start * layer.head_k_dim
            v_channel_start = head_start * layer.head_v_dim
            q_channels = num_cp_heads * layer.head_q_dim
            k_channels = num_cp_heads * layer.head_k_dim
            v_channels = num_cp_heads * layer.head_v_dim

            q_conv_weight = q_conv_weight.narrow(0, q_channel_start, q_channels)
            k_conv_weight = k_conv_weight.narrow(0, k_channel_start, k_channels)
            v_conv_weight = v_conv_weight.narrow(0, v_channel_start, v_channels)
            q_conv_state = q_conv_state.narrow(-2, q_channel_start, q_channels)
            k_conv_state = k_conv_state.narrow(-2, k_channel_start, k_channels)
            v_conv_state = v_conv_state.narrow(-2, v_channel_start, v_channels)
            if q_bias is not None:
                q_bias = q_bias.narrow(0, q_channel_start, q_channels)
                k_bias = k_bias.narrow(0, k_channel_start, k_channels)
                v_bias = v_bias.narrow(0, v_channel_start, v_channels)

        if use_fla_cp:
            assert fla_conv_initial_states is not None
            (
                q_conv_run_state,
                k_conv_run_state,
                v_conv_run_state,
            ) = fla_conv_initial_states.split(full_splits, dim=1)
        else:
            q_conv_run_state = q_conv_state
            k_conv_run_state = k_conv_state
            v_conv_run_state = v_conv_state

        def run_conv(x, weight, bias, state):
            # Note: at this point x is [C, T] (channel-first from
            # mixed_qkv.transpose(0,1).split).  causal_conv1d_fn expects
            # [C, T] channel-first input.  The output is transposed back to
            # [T, C] for downstream consumers.

            if use_fla_cp:
                assert fla_cp_context is not None
                local_indices = torch.arange(
                    fla_cp_context.num_local_segments,
                    device=cache_indices.device,
                    dtype=cache_indices.dtype,
                )
                state_work = state.to(weight.dtype).contiguous()
                out = causal_conv1d_fn(
                    x.to(weight.dtype),
                    weight,
                    bias,
                    activation="silu",
                    conv_states=state_work,
                    has_initial_state=torch.ones(
                        fla_cp_context.num_local_segments,
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    cache_indices=local_indices,
                    query_start_loc=fla_cp_context.local_cu_seqlens,
                    seq_lens_cpu=list(fla_cp_context.local_segment_lens),
                )
                return out.to(x.dtype).transpose(0, 1)

            if is_npu():
                # The Ascend varlen wrapper creates its padding buffer in the
                # weight dtype.  K3 stores conv weights in FP32 and activations /
                # cache in BF16, so adapt both inputs through a small active-row
                # FP32 state and explicitly cast the results back.
                local_indices = torch.arange(
                    cache_indices.shape[0],
                    device=cache_indices.device,
                    dtype=cache_indices.dtype,
                )
                state_work = state[cache_indices].to(weight.dtype).contiguous()
                out = causal_conv1d_fn(
                    x.to(weight.dtype),
                    weight,
                    bias,
                    activation="silu",
                    conv_states=state_work,
                    has_initial_state=has_initial_state,
                    cache_indices=local_indices,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                )
                state.index_copy_(0, cache_indices, state_work.to(state.dtype))
                return out.to(x.dtype).transpose(0, 1)

            return causal_conv1d_fn(
                x,
                weight,
                bias,
                activation="silu",
                conv_states=state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)

        q = run_conv(q, q_conv_weight, q_bias, q_conv_run_state)
        k = run_conv(k, k_conv_weight, k_bias, k_conv_run_state)
        v = run_conv(v, v_conv_weight, v_bias, v_conv_run_state)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        if use_kda_a2a:
            A_log = layer.A_log.narrow(2, head_start, num_cp_heads)
            dt_bias = (
                layer.dt_bias.view(layer.num_q_heads, layer.head_k_dim)
                .narrow(0, head_start, num_cp_heads)
                .flatten()
            )
            # A head slice of the full state pool is not contiguous: its
            # per-slot stride still contains all heads, while the KDA kernels
            # assume a densely packed [slot, local_head, K, V] pool. Compact
            # only this batch's active states and remap the slot indices.
            valid_cache_indices = cache_indices >= 0
            safe_cache_indices = cache_indices.clamp_min(0)
            ssm_states_for_kernel = (
                ssm_states.index_select(0, safe_cache_indices)
                .narrow(1, head_start, num_cp_heads)
                .contiguous()
            )
            if not bool(valid_cache_indices.all()):
                ssm_states_for_kernel[~valid_cache_indices].zero_()
            kernel_cache_indices = torch.arange(
                cache_indices.shape[0],
                device=cache_indices.device,
                dtype=cache_indices.dtype,
            )
        else:
            A_log = layer.A_log
            dt_bias = layer.dt_bias
            ssm_states_for_kernel = ssm_states
            kernel_cache_indices = cache_indices

        kernel_query_start_loc = (
            fla_cp_context.local_cu_seqlens if use_fla_cp else query_start_loc
        )

        core_attn_out, _, h = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states_for_kernel,
            cache_indices=kernel_cache_indices,
            query_start_loc=kernel_query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=getattr(layer, "lower_bound", None),
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            # target_verify / draft_extend_v2 also reach forward_extend; they must
            # stay rollback-able, so a kernel that commits state in place (e.g.
            # FlashKDA) must not run for them.
            is_spec_decode=(
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ),
            cp_context=fla_cp_context,
        )
        if use_kda_a2a:
            active_state_indices = cache_indices[valid_cache_indices]
            if active_state_indices.numel() > 0:
                ssm_states.narrow(1, head_start, num_cp_heads).index_copy_(
                    0,
                    active_state_indices,
                    ssm_states_for_kernel[valid_cache_indices],
                )
            self._sync_kda_cp_final_states(
                layer,
                cache_indices,
                conv_states,
                ssm_states,
                head_start,
                num_cp_heads,
            )
            if h is not None:
                # chunk_kda returns [1, num_chunks, heads, K, V].
                h = all_gather_cp_heads(h, head_dim=2)

        if h is not None and not use_fla_cp:
            # This branch's KDA kernels and recurrent-state pool both use the
            # [K, V] matrix layout.
            self._track_mamba_state_extend(
                forward_batch,
                h,
                ssm_states,
                self.forward_metadata,
            )
        if use_kda_a2a:
            core_attn_out = head_to_sequence_a2a(
                core_attn_out.squeeze(0), forward_batch
            ).unsqueeze(0)
        return core_attn_out

    def _forward_dspark_target_verify(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        fm = self.forward_metadata
        seq_len = mixed_qkv.shape[0]
        query_start_loc = fm.query_start_loc
        cache_indices = fm.mamba_cache_indices
        retrieve_next_token = fm.retrieve_next_token
        retrieve_next_sibling = fm.retrieve_next_sibling
        retrieve_parent_token = fm.retrieve_parent_token

        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        intermediate_state_cache = getattr(cache, "intermediate_ssm", None)
        if intermediate_state_cache is None:
            raise RuntimeError(
                "KDA DSpark target verify requires speculative Mamba scratch."
            )
        conv_states = cache.conv[0]
        ssm_states = cache.temporal
        intermediate_conv = cache.intermediate_conv_window[0]
        intermediate_indices = self.verify_intermediate_state_indices
        draft_token_num = forward_batch.spec_info.draft_token_num
        ragged_layout = getattr(forward_batch.spec_info, "ragged_verify_layout", None)

        batch_size = query_start_loc.shape[0] - 1
        num_dense = batch_size * draft_token_num
        if ragged_layout is None and seq_len == num_dense:
            dense_indices = None
            dense_qkv = mixed_qkv.view(batch_size, draft_token_num, -1)
        else:
            dense_indices = ragged_verify_dense_scatter_indices(
                query_start_loc=query_start_loc,
                seq_len=seq_len,
                draft_token_num=draft_token_num,
            )
            dense = mixed_qkv.new_zeros(num_dense + 1, mixed_qkv.shape[-1])
            dense.index_copy_(0, dense_indices, mixed_qkv)
            dense_qkv = dense[:num_dense].view(batch_size, draft_token_num, -1)

        if is_npu():
            processed = causal_conv1d_linear_verify_npu(
                dense_qkv.transpose(1, 2).contiguous(),
                conv_states,
                layer.conv_weights,
                layer.bias,
                cache_indices[:batch_size],
                intermediate_conv,
                intermediate_indices[:batch_size],
                activation="silu",
                update_persistent_state=False,
            )
        else:
            processed = causal_conv1d_update(
                dense_qkv.transpose(1, 2),
                conv_states.transpose(-1, -2),
                layer.conv_weights,
                layer.bias,
                activation="silu",
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv.transpose(-1, -2),
                intermediate_state_indices=intermediate_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
        flat = processed.transpose(1, 2).reshape(batch_size * draft_token_num, -1)
        if dense_indices is not None:
            padded = flat.new_zeros(flat.shape[0] + 1, flat.shape[1])
            padded[: flat.shape[0]] = flat
            flat = padded[dense_indices]

        q, k, v = flat.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
        out = self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_state_cache,
            intermediate_state_indices=intermediate_indices,
            cache_steps=draft_token_num,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=getattr(layer, "lower_bound", None),
        )
        if dense_indices is not None:
            covered = dense_indices < (batch_size * draft_token_num)
            out = torch.where(covered.view(1, -1, 1, 1), out, 0.0)
        return out
