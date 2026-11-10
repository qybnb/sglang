"""Run the GLM-5.2 C8 operator and DCP merge checks on Ascend 950."""

import argparse
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
PAGE = 128
TOPK = 2048
LATENT = 512
ROPE = 64
SCALE = 1 / math.sqrt(LATENT + ROPE)


def report_env():
    print("Python", sys.version, flush=True)
    if shutil.which("npu-smi"):
        subprocess.run(["npu-smi", "info"], check=False)
    print(
        json.dumps(
            {
                name: os.environ.get(name)
                for name in (
                    "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH",
                    "ASCEND_RT_VISIBLE_DEVICES",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if ascend:
        for name in ("version.info", "version.cfg", "ascend_toolkit_install.info"):
            file = Path(ascend) / name
            if file.is_file():
                print(file, file.read_text(errors="replace").strip(), flush=True)

    missing = []
    for name in (
        "sglang",
        "torch",
        "torch_npu",
        "sgl_kernel_npu",
    ):
        try:
            module = importlib.import_module(name)
            print(
                name,
                getattr(module, "__version__", ""),
                getattr(module, "__file__", ""),
                flush=True,
            )
        except Exception as exc:
            missing.append(name)
            print(f"{name}: {exc!r}", flush=True)

    if "torch_npu" not in missing and "torch" not in missing:
        import torch
        import torch_npu

        for name in (
            "npu_dynamic_quant",
            "npu_quant_lightning_indexer",
            "npu_kv_quant_sparse_flash_attention",
            "npu_attention_update",
        ):
            print(
                f"torch_npu.{name}: {callable(getattr(torch_npu, name, None))}",
                flush=True,
            )
        print(
            f"indexer alias={getattr(torch_npu, 'npu_quant_lightning_indexer', None)!r}; "
            f"native={getattr(torch.ops.npu, 'npu_quant_lightning_indexer', None)!r}",
            flush=True,
        )
        native = getattr(torch.ops.npu, "sgl_kv_quant_sparse_flash_attention", None)
        print(f"Native DCP FP8 attention: {callable(native)}", flush=True)
        if native is not None:
            print(native.default._schema, flush=True)
    print(
        "The numeric subcommands check whether the operators actually launch.",
        flush=True,
    )


def fp8_cpu(tensor):
    import torch

    return tensor.view(torch.uint8).cpu().view(torch.float8_e4m3fn).float()


def write_indexer_cache(k, scale, decode_kv_len=None):
    import torch

    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    # Match the pool's layer-major layout. Use the second slot to exercise a
    # nonzero storage offset through both scatter writes and the native reader.
    pool = object.__new__(NPUMLATokenToKVPool)
    pool.dtype = pool.store_dtype = k.dtype
    pool.layer_transfer_counter = None
    pool.index_head_dim = k.shape[-1]
    pool.index_page_size = PAGE
    pool.index_size = (k.shape[0] - 1) * PAGE
    pool.index_page_padding = 1
    pool.indexer_layer_id_to_slot = {0: 0, 1: 1}
    pool.index_k_buffer = k.new_zeros((2, *k.shape))
    pool.index_k_scale_buffer = scale.new_zeros((2, *scale.shape))

    def write_slots(start, stop):
        loc = torch.arange(start, stop, device=k.device, dtype=torch.int64)
        pool.set_index_k_scale_buffer(1, loc, scale.flatten(0, 1)[start:stop].clone())
        pool.set_index_k_buffer(1, loc, k.flatten(0, 1)[start:stop].clone())

    if decode_kv_len is None:
        write_slots(PAGE, k.shape[0] * PAGE)
    else:
        decode_slot = PAGE + decode_kv_len - 1
        write_slots(PAGE, decode_slot)
        torch.npu.synchronize()
        print(f"indexer cache append decode slot={decode_slot}", flush=True)
        write_slots(decode_slot, decode_slot + 1)
    return pool.get_index_k_buffer(1), pool.get_index_k_scale_buffer(1)


def check_indexer(
    device,
    score_atol,
    lengths=(127, 129, 2305, 2561),
    *,
    write_cache=False,
    decode_indexer_shape=None,
):
    import torch

    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
        _quantize_npu_indexer_activation,
        create_npu_hadamard_128,
    )

    heads, dim, topk = decode_indexer_shape or (64, 128, TOPK)
    # Decode appends to the prefix and uses the same request's page table.
    decode_kv_len = lengths[0] if decode_indexer_shape is not None else None
    lengths = torch.tensor(lengths, dtype=torch.int32)
    rows = lengths.numel()
    pages = math.ceil(int(lengths.max()) / PAGE)
    page_ids = torch.arange(1, pages + 1, dtype=torch.int32)
    if decode_kv_len is None:
        page_ids = page_ids.flip(0)
    tables = torch.stack([page_ids.roll(row) for row in range(rows)])
    q = torch.randn(rows, heads, dim).to(device, torch.bfloat16)
    k = torch.randn(pages + 1, PAGE, 1, dim).to(device, torch.bfloat16)
    weights = (torch.rand(rows, heads) / heads).to(device, torch.bfloat16)
    hadamard = create_npu_hadamard_128(dim, device)
    q8, sq = _quantize_npu_indexer_activation(q, hadamard, torch.float8_e4m3fn)
    k8, sk = _quantize_npu_indexer_activation(k, hadamard, torch.float8_e4m3fn)
    key, key_scale = k8, sk
    if write_cache or decode_kv_len is not None:
        torch.npu.synchronize()
        print(f"indexer cache write lengths={lengths.tolist()}", flush=True)
        key, key_scale = write_indexer_cache(k8, sk, decode_kv_len)
        torch.npu.synchronize()
        # Use the pre-scatter tensors as the score reference. Check the cache
        # first, so a scatter failure cannot look like a Top-K selection error.
        valid_slots = slice(
            PAGE, None if decode_kv_len is None else PAGE + decode_kv_len
        )
        torch.testing.assert_close(
            key.flatten(0, 1)[valid_slots].view(torch.uint8).cpu(),
            k8.flatten(0, 1)[valid_slots].view(torch.uint8).cpu(),
        )
        torch.testing.assert_close(
            key_scale.flatten(0, 1)[valid_slots].cpu(),
            sk.flatten(0, 1)[valid_slots].cpu(),
            rtol=0,
            atol=0,
        )
        print(
            f"indexer native call Q={tuple(q8.shape)} table={tuple(tables.shape)} "
            f"topk={topk} K storage_offset={key.storage_offset()}",
            flush=True,
        )
    result = torch.ops.npu.npu_quant_lightning_indexer(
        query=q8,
        key=key,
        weights=weights,
        query_dequant_scale=sq,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=torch.arange(
            1, rows + 1, dtype=torch.int32, device=device
        ),
        actual_seq_lengths_key=lengths.to(device),
        block_table=tables.to(device),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
        query_quant_mode=0,
        key_quant_mode=0,
    ).cpu()
    if result.shape != (rows, 1, topk) or result.dtype != torch.int32:
        raise AssertionError(f"Unexpected indexer output {result.shape} {result.dtype}")
    q_ref = fp8_cpu(q8) * sq.float().cpu().unsqueeze(-1)
    k_ref = (fp8_cpu(k8) * sk.float().cpu().unsqueeze(-1)).squeeze(2)
    weights = weights.float().cpu()
    for row, length in enumerate(lengths.tolist()):
        picked = result[row, 0].long()
        count = min(length, topk)
        ids = picked[:count]
        if (
            not bool(((ids >= 0) & (ids < length)).all())
            or ids.unique().numel() != count
            or not bool((picked[count:] == -1).all())
        ):
            raise AssertionError(
                f"Invalid Top-K range, count, duplicates or padding at row {row}"
            )
        keys = k_ref[tables[row].long()].reshape(-1, dim)[:length]
        scores = ((q_ref[row] @ keys.T).relu() * weights[row, :, None]).sum(0)
        gold = scores.topk(count)
        # Compare scores at the selection boundary, where exact set equality
        # can penalize equally good choices on different kernels.
        gap = max(0.0, (gold.values[-1] - scores[ids].min()).item())
        overlap = len(set(gold.indices.tolist()) & set(ids.tolist())) / count
        print(
            f"indexer row={row} len={length} overlap={overlap:.4f} boundary_gap={gap:.6f}"
        )
        if gap > score_atol:
            raise AssertionError(f"Top-K boundary gap {gap} exceeds {score_atol}")


def pack_kv(latent, rope, device):
    import torch

    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    pool = SimpleNamespace(
        dtype=torch.float8_e4m3fn, kv_lora_rank=LATENT, qk_rope_head_dim=ROPE
    )
    packed = NPUMLATokenToKVPool._pack_dsa_fp8_kv_cache(
        pool, latent.to(device), rope.to(device)
    ).view(-1, PAGE, 1, LATENT + 2 * ROPE + 16)
    raw = packed.view(torch.uint8).cpu()
    scale = raw[..., LATENT + 2 * ROPE :].contiguous().view(torch.float32)
    k = raw[..., :LATENT].contiguous().view(torch.float8_e4m3fn).float()
    k = (k.unflatten(-1, (4, 128)) * scale.unsqueeze(-1)).flatten(-2)
    kr = raw[..., LATENT : LATENT + 2 * ROPE].contiguous().view(torch.bfloat16)
    return packed, k.squeeze(2), kr.float().squeeze(2)


def fixture():
    import torch

    # Two verify chains straddle page ownership transitions. Rows 0 and 2
    # have no valid Top-K on rank 1 and rank 0, respectively.
    lengths = torch.tensor([128, 129, 2048, 2049], dtype=torch.int32)
    indices = torch.full((4, 1, TOPK), -1, dtype=torch.int32)
    for row, positions in enumerate(
        ([3, 29, 97, 127], [128, 3, 96, 127], [135, 221, 255], range(2048, 0, -1))
    ):
        indices[row, 0, : len(positions)] = torch.tensor(positions)
    page_ids = list(range(17, 0, -1))
    return SimpleNamespace(
        q=(torch.randn(4, 32, LATENT) * 0.5).bfloat16(),
        qr=(torch.randn(4, 32, ROPE) * 0.5).bfloat16(),
        latent=(torch.randn(len(page_ids) + 1, PAGE, 1, LATENT) * 0.4).bfloat16(),
        rope=(torch.randn(len(page_ids) + 1, PAGE, 1, ROPE) * 0.4).bfloat16(),
        lengths=lengths,
        indices=indices,
        page_ids=page_ids,
        q_lengths=torch.arange(1, 5, dtype=torch.int32),
    )


def attention_reference(data, key, rope, table, indices):
    import torch

    out = torch.zeros_like(data.q, dtype=torch.float32)
    maxes = torch.full(out.shape[:2], -torch.inf)
    sums = torch.zeros_like(maxes)
    k = key[table].flatten(0, 1)
    kr = rope[table].flatten(0, 1)
    for row, ids in enumerate(indices[:, 0]):
        chosen = ids[ids >= 0].long()
        if chosen.numel():
            logits = (
                data.q[row].float() @ k[chosen].T + data.qr[row].float() @ kr[chosen].T
            ) * SCALE
            maxes[row] = logits.amax(-1)
            sums[row] = (logits - maxes[row, :, None]).exp().sum(-1)
            out[row] = logits.softmax(-1) @ k[chosen]
    return out, maxes, sums


def check_close(label, got, want, *, atol, rtol):
    import torch

    got, want = got.float().cpu(), want.float().cpu()
    finite = want.isfinite()
    error = (got[finite] - want[finite]).abs()
    print(
        f"{label}: max_abs_error={error.max().item() if error.numel() else 0.0:.6g}",
        flush=True,
    )
    torch.testing.assert_close(
        got, want, atol=atol, rtol=rtol, msg=lambda msg: f"{label}\n{msg}"
    )


def check_qsfa(data, device, args):
    import torch

    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    packed, key, rope = pack_kv(data.latent, data.rope, device)
    table = torch.tensor(
        [data.page_ids] * data.q.shape[0], dtype=torch.int32, device=device
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(block_tables=table), qk_rope_head_dim=ROPE
    )
    result = AscendAttnBackend._forward_fp8_sparse(
        backend,
        data.q.to(device),
        data.qr.to(device),
        packed,
        data.indices.to(device),
        SimpleNamespace(scaling=SCALE),
        data.q_lengths,
        data.lengths,
    )
    ref, _, _ = attention_reference(data, key, rope, data.page_ids, data.indices)
    check_close(
        "ordinary QSFA / PCP-shaped metadata",
        result,
        ref,
        atol=args.atol,
        rtol=args.rtol,
    )


def dcp_shard(data, rank, device, args, *, verify):
    import torch

    from sglang.srt.hardware_backend.npu.attention.dsa_dcp import (
        forward_dcp_sparse_attention,
        load_dcp_fp8_sparse_attention,
    )
    from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices

    native = load_dcp_fp8_sparse_attention()
    packed, key, rope = pack_kv(data.latent, data.rope, device)
    selected = [0, *data.page_ids[rank::2]]
    packed_local = (
        packed.view(torch.uint8).cpu()[selected].to(device).view(torch.float8_e4m3fn)
    )
    local_lens = torch.tensor(
        [
            sum(
                min(PAGE, max(0, length - page * PAGE))
                for page in range(rank, len(data.page_ids), 2)
            )
            for length in data.lengths.tolist()
        ],
        dtype=torch.int32,
        device=device,
    )
    table = torch.tensor(
        [list(range(1, len(selected)))] * data.q.shape[0],
        dtype=torch.int32,
        device=device,
    )
    local_topk = remap_dcp_sparse_indices(
        data.indices.to(device), 2, rank, interleave_size=PAGE
    )
    global_topk = data.indices.clone()
    for row, ids in enumerate(data.indices[:, 0]):
        kept = [p for p in ids.tolist() if p >= 0 and p // PAGE % 2 == rank]
        global_topk[row].fill_(-1)
        global_topk[row, 0, : len(kept)] = torch.tensor(kept, dtype=torch.int32)
        expected = [p // PAGE // 2 * PAGE + p % PAGE for p in kept] + [-1] * (
            TOPK - len(kept)
        )
        if local_topk[row, 0].tolist() != expected:
            raise AssertionError(f"Rank {rank}, row {row}: DCP remap mismatch")
    if verify:
        metadata = SimpleNamespace(
            dcp_spec_block_tables=table, dcp_spec_seq_lens=local_lens
        )
    else:
        metadata = SimpleNamespace(dcp_block_tables=table, dcp_seq_lens=local_lens)
    stats = []

    def capture_stats(**kwargs):
        out, maxes, sums = native(**kwargs)
        stats.extend((maxes, sums))
        return out, maxes, sums

    out, lse = forward_dcp_sparse_attention(
        q_nope=data.q.to(device),
        q_rope=data.qr.to(device),
        k_nope=packed_local,
        k_rope=packed_local[..., :0],
        topk_indices=local_topk,
        actual_seq_lengths_query=data.q_lengths,
        forward_metadata=metadata,
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: verify)
        ),
        speculative_num_draft_tokens=2 if verify else None,
        scaling=SCALE,
        fp8_attention=capture_stats,
    )
    ref, maxes, sums = attention_reference(data, key, rope, data.page_ids, global_topk)
    ref_lse = maxes + sums.log()
    valid = global_topk[:, 0, 0] >= 0
    label = f"Native FP8 {'verify' if verify else 'decode'} rank={rank}"
    check_close(label + " output", out, ref, atol=args.atol, rtol=args.rtol)
    check_close(
        label + " max",
        stats[0].cpu()[0, valid],
        maxes[valid],
        atol=args.lse_atol,
        rtol=args.rtol,
    )
    check_close(
        label + " sum",
        stats[1].cpu()[0, valid],
        sums[valid],
        atol=args.lse_atol,
        rtol=args.rtol,
    )
    check_close(label + " LSE", lse, ref_lse, atol=args.lse_atol, rtol=args.rtol)
    if (
        not torch.all(out.cpu()[~valid] == 0)
        or not torch.isneginf(lse.cpu()[~valid]).all()
    ):
        raise AssertionError(f"{label}: empty rank rows must be exactly zero / -inf")
    full_ref, _, _ = attention_reference(data, key, rope, data.page_ids, data.indices)
    return out, lse, full_ref


def check_dcp(data, device, args):
    import torch

    for verify in (False, True):
        out0, lse0, full = dcp_shard(data, 0, device, args, verify=verify)
        out1, lse1, _ = dcp_shard(data, 1, device, args, verify=verify)
        weight0, weight1 = torch.stack((lse0.cpu(), lse1.cpu())).softmax(0).unbind(0)
        merged = out0.float().cpu() * weight0.unsqueeze(
            -1
        ) + out1.float().cpu() * weight1.unsqueeze(-1)
        check_close(
            "Native FP8 two-shard CPU merge",
            merged,
            full,
            atol=args.atol,
            rtol=args.rtol,
        )


def check_dcp2(data, device, args):
    import torch.distributed as dist

    from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla_npu

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("Launch dcp2 with torchrun --standalone --nproc-per-node=2")
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        rank = dist.get_rank()
        out, lse, full = dcp_shard(data, rank, device, args, verify=True)
        group = SimpleNamespace(world_size=2, all_to_all_single=dist.all_to_all_single)
        merged = cp_lse_ag_out_rs_mla_npu(out, lse, group)
        expected = full.chunk(2, dim=1)[rank]
        check_close(
            f"HCCL merge rank={rank}", merged, expected, atol=args.atol, rtol=args.rtol
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "env",
            "indexer",
            "indexer-cache",
            "indexer-decode",
            "qsfa",
            "dcp",
            "dcp2",
        ),
    )
    parser.add_argument(
        "--model", type=Path, help="Local checkpoint directory for indexer-decode"
    )
    parser.add_argument(
        "--device", type=int, default=int(os.environ.get("LOCAL_RANK", "0"))
    )
    parser.add_argument("--seed", type=int, default=531)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--lse-atol", type=float, default=0.03)
    parser.add_argument("--topk-score-atol", type=float, default=0.05)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "python"))
    if args.case == "env":
        report_env()
        return
    if args.case == "indexer-decode":
        if args.model is None or not (args.model / "config.json").is_file():
            parser.error(
                "indexer-decode requires --model pointing to a local config.json"
            )
        config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        decode_indexer_shape = tuple(
            config[key] for key in ("index_n_heads", "index_head_dim", "index_topk")
        )

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    name = torch.npu.get_device_name(args.device)
    print(f"case={args.case} device={device} name={name} seed={args.seed}", flush=True)
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        if args.case == "indexer":
            check_indexer(device, args.topk_score_atol)
        elif args.case == "indexer-cache":
            check_indexer(
                device, args.topk_score_atol, lengths=(2, 4, 6), write_cache=True
            )
            check_indexer(device, args.topk_score_atol, write_cache=True)
        elif args.case == "indexer-decode":
            for kv_len in (98, 4098):
                check_indexer(
                    device,
                    args.topk_score_atol,
                    lengths=(kv_len,),
                    decode_indexer_shape=decode_indexer_shape,
                )
        elif args.case == "qsfa":
            check_qsfa(fixture(), device, args)
        elif args.case == "dcp":
            check_dcp(fixture(), device, args)
        else:
            check_dcp2(fixture(), device, args)
    torch.npu.synchronize()
    print(f"PASS {args.case}", flush=True)


if __name__ == "__main__":
    main()
