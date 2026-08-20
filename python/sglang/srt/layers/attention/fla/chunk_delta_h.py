# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.srt.layers.attention.fla.op import exp, safe_exp
from sglang.srt.layers.attention.fla.utils import is_nvidia_hopper

NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8, 16]
CHUNK_SIZE = 64


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_affine_h_kernel(
    k,
    v,
    w,
    gk,
    affine,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Compute the additive half of a segment's recurrent affine map.

    This is the multi-segment Ascend adaptation of FLA PR 691's
    ``pre_process_fwd_kernel_stage1``.  Unlike the ordinary state kernel it
    keeps only the final state of every segment and therefore does not write a
    ``[num_chunks, H, K, V]`` intermediate tensor.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    segment_len = eos - bos
    num_chunks = tl.cdiv(segment_len, BT)

    affine += ((i_n * H + i_h) * K * (V + K)).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    h0 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        h3 = tl.zeros([64, BV], dtype=tl.float32)

    for chunk_id in range(num_chunks):
        w0_ptr = tl.make_block_ptr(
            w,
            (segment_len, K),
            (stride_w, 1),
            (chunk_id * BT, 0),
            (BT, 64),
            (1, 0),
        )
        w0 = tl.load(w0_ptr, boundary_check=(0, 1))
        value = tl.dot(w0, h0.to(w0.dtype))
        if K > 64:
            w1_ptr = tl.make_block_ptr(
                w,
                (segment_len, K),
                (stride_w, 1),
                (chunk_id * BT, 64),
                (BT, 64),
                (1, 0),
            )
            w1 = tl.load(w1_ptr, boundary_check=(0, 1))
            value += tl.dot(w1, h1.to(w1.dtype))
        if K > 128:
            w2_ptr = tl.make_block_ptr(
                w,
                (segment_len, K),
                (stride_w, 1),
                (chunk_id * BT, 128),
                (BT, 64),
                (1, 0),
            )
            w2 = tl.load(w2_ptr, boundary_check=(0, 1))
            value += tl.dot(w2, h2.to(w2.dtype))
        if K > 192:
            w3_ptr = tl.make_block_ptr(
                w,
                (segment_len, K),
                (stride_w, 1),
                (chunk_id * BT, 192),
                (BT, 64),
                (1, 0),
            )
            w3 = tl.load(w3_ptr, boundary_check=(0, 1))
            value += tl.dot(w3, h3.to(w3.dtype))

        value_ptr = tl.make_block_ptr(
            v,
            (segment_len, V),
            (stride_v, 1),
            (chunk_id * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        value = tl.load(value_ptr, boundary_check=(0, 1)) - value

        last_token = min((chunk_id + 1) * BT, segment_len) - 1
        key_offsets = tl.arange(0, 64)
        gk0 = tl.load(
            gk + (bos + last_token) * H * K + i_h * K + key_offsets,
            mask=key_offsets < K,
            other=0.0,
        )
        h0 *= exp(gk0)[:, None]
        if K > 64:
            key_offsets1 = 64 + key_offsets
            gk1 = tl.load(
                gk + (bos + last_token) * H * K + i_h * K + key_offsets1,
                mask=key_offsets1 < K,
                other=0.0,
            )
            h1 *= exp(gk1)[:, None]
        if K > 128:
            key_offsets2 = 128 + key_offsets
            gk2 = tl.load(
                gk + (bos + last_token) * H * K + i_h * K + key_offsets2,
                mask=key_offsets2 < K,
                other=0.0,
            )
            h2 *= exp(gk2)[:, None]
        if K > 192:
            key_offsets3 = 192 + key_offsets
            gk3 = tl.load(
                gk + (bos + last_token) * H * K + i_h * K + key_offsets3,
                mask=key_offsets3 < K,
                other=0.0,
            )
            h3 *= exp(gk3)[:, None]

        value = value.to(k.dtype.element_ty)
        k0_ptr = tl.make_block_ptr(
            k,
            (K, segment_len),
            (1, stride_k),
            (0, chunk_id * BT),
            (64, BT),
            (0, 1),
        )
        key0 = tl.load(k0_ptr, boundary_check=(0, 1))
        h0 += tl.dot(key0, value)
        if K > 64:
            k1_ptr = tl.make_block_ptr(
                k,
                (K, segment_len),
                (1, stride_k),
                (64, chunk_id * BT),
                (64, BT),
                (0, 1),
            )
            key1 = tl.load(k1_ptr, boundary_check=(0, 1))
            h1 += tl.dot(key1, value)
        if K > 128:
            k2_ptr = tl.make_block_ptr(
                k,
                (K, segment_len),
                (1, stride_k),
                (128, chunk_id * BT),
                (64, BT),
                (0, 1),
            )
            key2 = tl.load(k2_ptr, boundary_check=(0, 1))
            h2 += tl.dot(key2, value)
        if K > 192:
            k3_ptr = tl.make_block_ptr(
                k,
                (K, segment_len),
                (1, stride_k),
                (192, chunk_id * BT),
                (64, BT),
                (0, 1),
            )
            key3 = tl.load(k3_ptr, boundary_check=(0, 1))
            h3 += tl.dot(key3, value)

    out0 = tl.make_block_ptr(
        affine,
        (K, V),
        (V + K, 1),
        (0, i_v * BV),
        (64, BV),
        (1, 0),
    )
    tl.store(out0, h0.to(out0.dtype.element_ty), boundary_check=(0, 1))
    if K > 64:
        out1 = tl.make_block_ptr(
            affine,
            (K, V),
            (V + K, 1),
            (64, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(out1, h1.to(out1.dtype.element_ty), boundary_check=(0, 1))
    if K > 128:
        out2 = tl.make_block_ptr(
            affine,
            (K, V),
            (V + K, 1),
            (128, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(out2, h2.to(out2.dtype.element_ty), boundary_check=(0, 1))
    if K > 192:
        out3 = tl.make_block_ptr(
            affine,
            (K, V),
            (V + K, 1),
            (192, i_v * BV),
            (64, BV),
            (1, 0),
        )
        tl.store(out3, h3.to(out3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_affine(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.LongTensor,
) -> torch.Tensor:
    """Return per-segment ``[H | M]`` without materializing chunk states.

    The output satisfies ``state_out = M @ state_in + H`` and is laid out as
    ``[segments, heads, K, V + K]``.  It is intentionally FP32 because the
    cross-rank affine composition is numerically sensitive.
    """
    if cu_seqlens is None:
        raise ValueError("KDA CP affine preprocessing requires cu_seqlens")
    if gk is None:
        raise ValueError("KDA CP affine preprocessing requires per-key gates")
    _, token_count, key_heads, key_dim = k.shape
    num_heads = u.shape[-2]
    value_dim = u.shape[-1]
    num_segments = len(cu_seqlens) - 1
    if key_dim > 256:
        raise ValueError("KDA CP affine preprocessing supports K <= 256")
    if num_heads % key_heads != 0:
        raise ValueError("KDA CP affine preprocessing requires H divisible by Hg")

    affine = torch.empty(
        num_segments,
        num_heads,
        key_dim,
        value_dim + key_dim,
        dtype=torch.float32,
        device=k.device,
    )
    chunk_gated_delta_rule_fwd_affine_h_kernel[
        (triton.cdiv(value_dim, 64), num_segments * num_heads)
    ](
        k=k,
        v=u,
        w=w,
        gk=gk,
        affine=affine,
        cu_seqlens=cu_seqlens,
        T=token_count,
        H=num_heads,
        Hg=key_heads,
        K=key_dim,
        V=value_dim,
        BT=CHUNK_SIZE,
        BV=64,
        num_warps=4,
        num_stages=2,
    )

    # BiShengIR cannot currently keep PR 691's KxK dot accumulator in CC
    # across a dynamic chunk loop.  Reuse the established recurrent kernel for
    # M, but suppress both its per-chunk state stores and its value loads.  This
    # retains the proven Ascend code generation while avoiding the two largest
    # sources of overhead in the original correctness-first adaptation.
    affine_indices = torch.arange(num_segments, dtype=torch.int32, device=k.device)
    transition = torch.eye(key_dim, dtype=torch.float32, device=k.device).view(
        1, 1, key_dim, key_dim
    )
    transition = transition.expand(num_segments, num_heads, key_dim, key_dim).clone()
    chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=transition,
        initial_state_indices=affine_indices,
        save_new_value=False,
        cu_seqlens=cu_seqlens,
        store_chunk_state=False,
        zero_value=True,
        block_value=64,
    )
    affine[..., value_dim:].copy_(transition)
    return affine


# @triton.autotune(
#     configs=[
#         triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
#         for num_warps in [2, 4]
#         for num_stages in [2, 3, 4]
#         for BV in [32, 64]
#     ],
#     key=["H", "K", "V", "BT", "USE_G"],
#     use_cuda_graph=use_cuda_graph,
# )
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STORE_CHUNK_STATE: tl.constexpr,
    ZERO_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    index = tl.load(initial_state_indices + i_n).to(tl.int32)
    h0 = initial_state + index * stride_h
    ht = initial_state + index * stride_h
    if USE_INITIAL_STATE:
        h0 = h0 + i_h * K * V
    if INPLACE_UPDATE:
        ht = ht + i_h * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if STORE_CHUNK_STATE:
            p_h1 = tl.make_block_ptr(
                h + i_t * stride_h,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (64, BV),
                (1, 0),
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t * stride_h,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        if ZERO_VALUE:
            b_v = -b_v
        else:
            p_v = tl.make_block_ptr(
                v,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * safe_exp(b_g_last - b_g)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    # epilogue
    if INPLACE_UPDATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    store_chunk_state: bool = True,
    zero_value: bool = False,
    block_value: int = 32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is not None
        else None
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    if block_value not in (32, 64):
        raise ValueError(
            f"KDA state kernel block_value must be 32 or 64, got {block_value}"
        )

    h = k.new_empty(B, NT, H, K, V) if store_chunk_state else k.new_empty(1)

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, block_value), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=block_value,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_UPDATE=True,
        SAVE_NEW_VALUE=v_new is not None,
        STORE_CHUNK_STATE=store_chunk_state,
        ZERO_VALUE=zero_value,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=2,
    )
    return (h if store_chunk_state else None), v_new
