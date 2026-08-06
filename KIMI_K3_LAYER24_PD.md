# Kimi-K3 24-layer single-node PD deployment

This recipe targets one server with 16 Ascend NPUs and 64 GiB HBM per NPU.
It creates a prefix-pruned Kimi-K3 checkpoint and runs:

- NPU 0-7: Prefill, TP=8
- NPU 8-15: Decode, TP=8
- Local Ascend transfer store and PD router

## 1. Create the 24-layer checkpoint

Use a complete local Kimi-K3 ModelSlim checkpoint as `FULL_MODEL_PATH`.
The output must be a new or empty directory.

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3

export FULL_MODEL_PATH=/path/to/full/Kimi-K3-int4
export REDUCED_MODEL_PATH=/path/to/Kimi-K3-int4-layer24

python3 python/sglang/srt/debug_utils/kimi_k3_layer_pruner.py \
  --input "$FULL_MODEL_PATH" \
  --output "$REDUCED_MODEL_PATH" \
  --keep-num-layers 24 \
  --link-mode hardlink
```

`hardlink` does not duplicate shard data and remains valid if the original
directory is renamed. Both directories must be on the same filesystem. The
tool exits instead of unexpectedly copying hundreds of GB if a hard link
cannot be created; use `--link-mode symlink` or `--link-mode copy` explicitly
when that is what you want.

The tool changes the nested `text_config.num_hidden_layers`, truncates the
one-based `kda_layers` and `full_attn_layers` lists, filters the safetensors
index, and retains all tokenizer, vision, projector, embedding, norm, and
language-head tensors.

For Kimi-K3, 24 is a clean boundary: it contains six complete
`3 KDA + 1 MLA` groups and two complete 12-layer Attention-Residual groups.

## 2. Start Prefill, Decode, and Router

Open three terminals. All three commands must use the exact same
`REDUCED_MODEL_PATH`.

Terminal 1:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/Kimi-K3-int4-layer24 ./run_16p_pd_layer24.sh prefill
```

Terminal 2:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/Kimi-K3-int4-layer24 ./run_16p_pd_layer24.sh decode
```

Terminal 3, after both model servers are ready:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/Kimi-K3-int4-layer24 ./run_16p_pd_layer24.sh router
```

The OpenAI-compatible endpoint is `http://127.0.0.1:6688/v1`.

## 3. Smoke test

```bash
curl http://127.0.0.1:6688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/Kimi-K3-int4-layer24",
    "messages": [{"role": "user", "content": "用一句话介绍月球。"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Memory tuning

Defaults are deliberately conservative:

- `MEM_FRACTION_STATIC=0.84`
- `MAX_TOTAL_TOKENS=32768`
- `MAX_RUNNING_REQUESTS=16`
- `CHUNKED_PREFILL_SIZE=4096`
- BF16 KDA/SSM state

If either role reports out-of-memory while loading, first create a 20-layer
checkpoint and use the same launcher:

```bash
python3 python/sglang/srt/debug_utils/kimi_k3_layer_pruner.py \
  --input "$FULL_MODEL_PATH" \
  --output /path/to/Kimi-K3-int4-layer20 \
  --keep-num-layers 20 \
  --link-mode hardlink
```

If loading succeeds but cache allocation fails, lower
`MAX_TOTAL_TOKENS` to `16384` and `MAX_RUNNING_REQUESTS` to `8`. If there is
substantial free HBM after warmup, increase `MEM_FRACTION_STATIC` gradually,
then increase token capacity.

Prefix layer pruning is intended to make the serving topology runnable.
Removing 69 of 93 trained layers causes a major quality loss; useful model
quality requires continued pretraining or teacher distillation after pruning.
