"""Opt-in diagnostics for the A5 validation branch."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def describe(tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tensor.stride(),
        "offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def sample(tensor):
    if tensor.numel() <= 16:
        return tensor.tolist()
    return dict(
        count=tensor.numel(), first=tensor[:8].tolist(), last=tensor[-8:].tolist()
    )


class IndexerDiagnostic:
    def __init__(self, directory, batch, layer):
        self.directory = Path(directory)
        self.capture = (self.directory / "capture").is_file()
        self.decode = batch.forward_mode.is_decode()
        self.rank = torch.distributed.get_rank()
        self.context = dict(
            pid=os.getpid(),
            rank=self.rank,
            layer=layer,
            mode=batch.forward_mode.name,
            capture=self.capture,
        )
        if batch.seq_lens_cpu is not None:
            self.context["batch_seq_lens"] = batch.seq_lens_cpu.tolist()

    def log(self, stage, **fields):
        print(
            "QLI_DIAG " + json.dumps(dict(self.context, stage=stage, **fields)),
            flush=True,
        )

    def fence(self, stage):
        if self.capture:
            self.log(stage + "_wait")
            torch.npu.synchronize()
            self.log(stage + "_ok")

    def cache_write(self, loc, k, scale, pool):
        self.fence("before_scatter")
        if self.capture:
            self.slots = loc.flatten().cpu().long()
            self.log(
                "scatter_inputs",
                slots=sample(self.slots),
                k=describe(k),
                scale=describe(scale),
            )
            capacity = pool.index_k_buffer.shape[1] * pool.index_page_size
            if (self.slots < 0).any() or (self.slots >= capacity).any():
                raise ValueError(f"Index K scatter slot outside [0, {capacity})")
        else:
            self.log("scatter_enter", k=describe(k), slots=describe(loc))

    def cache_written(self, k, scale, pool, layer):
        self.fence("key_scatter")
        if self.capture:
            # Slot zero is used for padding. Compare the rows with a real page.
            valid = self.slots >= pool.index_page_size
            slots = self.slots[valid]
            key = pool.get_index_k_buffer(layer).view(torch.uint8).flatten(0, 1).cpu()
            key_scale = pool.get_index_k_scale_buffer(layer).flatten(0, 1).cpu()
            expected_k = k.view(torch.uint8).reshape(-1, 1, k.shape[-1]).cpu()[valid]
            expected_scale = scale.reshape(-1, 1).cpu()[valid]
            k_ok = torch.equal(key.index_select(0, slots), expected_k)
            scale_ok = torch.equal(key_scale.index_select(0, slots), expected_scale)
            self.log("scatter_readback", key_match=k_ok, scale_match=scale_ok)
        else:
            self.log("scatter_enqueued")

    def call(self, **kwargs):
        self.fence("before_indexer")
        meta = {
            k: describe(v) for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        if self.capture:
            q_lens = kwargs["actual_seq_lengths_query"].cpu()
            kv_lens = kwargs["actual_seq_lengths_key"].cpu()
            table = kwargs["block_table"].cpu()
            page = kwargs["key"].shape[1]
            issues = []
            if q_lens.numel() != kv_lens.numel() or table.shape[0] != kv_lens.numel():
                issues.append("query lengths, KV lengths and page-table rows disagree")
            if q_lens.numel() and q_lens[-1] > kwargs["query"].shape[0]:
                issues.append("Q prefix sum exceeds the physical Q row count")
            if table.ndim == 2 and table.shape[0] == kv_lens.numel():
                pages = (kv_lens.long() + page - 1) // page
                if (pages > table.shape[1]).any():
                    issues.append("KV length requires more pages than the table has")
                valid_pages = table[torch.arange(table.shape[1]) < pages[:, None]]
                if ((valid_pages < 0) | (valid_pages >= kwargs["key"].shape[0])).any():
                    issues.append("an active page id is outside the K cache")
            expected_slots = None
            if self.decode and table.ndim == 2 and table.shape[0] == kv_lens.numel():
                col = (kv_lens.long() - 1) // page
                valid = (kv_lens > 0) & (col < table.shape[1])
                if valid.any():
                    row = torch.arange(len(kv_lens))[valid]
                    expected_slots = (
                        table[row, col[valid]].long() * page
                        + (kv_lens[valid] - 1) % page
                    ).tolist()
                    if len(self.slots) == len(kv_lens):
                        if expected_slots != self.slots[valid].tolist():
                            issues.append(
                                "decode scatter slots disagree with the page table"
                            )
            self.log(
                "metadata",
                q_lens=sample(q_lens),
                kv_lens=sample(kv_lens),
                table=table.tolist() if self.decode else table[:4].tolist(),
                expected_decode_slots=expected_slots,
                issues=issues,
            )

        snapshot = None
        if self.capture and self.decode:
            snapshot = self.directory / f"rank-{self.rank}.pt"
            tmp = snapshot.with_suffix(".tmp")
            payload = {}
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float8_e4m3fn:
                        value = value.view(torch.uint8)
                    value = value.detach().cpu().contiguous().clone()
                payload[name] = value
            finite = {
                name: bool(torch.isfinite(payload[name].float()).all())
                for name in ("weights", "query_dequant_scale")
            }
            finite["query"] = bool(
                torch.isfinite(payload["query"].view(torch.float8_e4m3fn).float()).all()
            )
            self.log("input_values", finite=finite)
            torch.save(dict(args=payload, meta=meta, context=self.context), tmp)
            tmp.replace(snapshot)
        indexer = torch.ops.npu.npu_quant_lightning_indexer
        self.log("native_enter", operator=str(indexer), tensors=meta)
        result = indexer(**kwargs)
        self.fence("indexer")
        self.log("native_exit")
        if snapshot is not None:
            snapshot.unlink()
        return result


def replay(path):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    args = snapshot["args"]
    device = snapshot["meta"]["query"]["device"]
    torch.npu.set_device(device)
    for name, info in snapshot["meta"].items():
        data = args[name]
        extent = (
            info["offset"]
            + 1
            + sum(
                (size - 1) * stride
                for size, stride in zip(info["shape"], info["stride"], strict=True)
            )
        )
        storage = torch.empty(extent, dtype=data.dtype, device=device)
        view = storage.as_strided(info["shape"], info["stride"], info["offset"])
        view.copy_(data.to(device))
        args[name] = (
            view.view(torch.float8_e4m3fn)
            if info["dtype"] == "torch.float8_e4m3fn"
            else view
        )
    torch.npu.synchronize()
    indexer = torch.ops.npu.npu_quant_lightning_indexer
    print(f"QLI replay ENTER operator={indexer} {snapshot['context']}", flush=True)
    result = indexer(**args)
    torch.npu.synchronize()
    print(f"QLI replay PASS {tuple(result.shape)}", flush=True)


if __name__ == "__main__":
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
    replay(sys.argv[1])
