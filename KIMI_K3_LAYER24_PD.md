# Kimi-K3 24-layer single-node PD deployment

This recipe targets one server with 16 Ascend NPUs and 64 GiB HBM per NPU.
It uses the original checkpoint but loads only a prefix of language layers:

- NPU 0-7: Prefill, TP=8
- NPU 8-15: Decode, TP=8
- Local Ascend transfer store and PD router

## 1. Use the original full checkpoint

No reduced checkpoint is generated, and the original weight directory is not
modified. Set `MODEL_PATH` to the complete local Kimi-K3 ModelSlim checkpoint.
At startup, the launcher overrides only the in-memory
`text_config.num_hidden_layers`:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3

export MODEL_PATH=/path/to/full/Kimi-K3-int4
export NUM_HIDDEN_LAYERS=24
```

The model creates only the first 24 language layers. During loading, SGLang
auto-detects the original `*.safetensors.index.json`, skips shards containing
only language layers 24 and above, and filters removed tensor names before
calling `get_tensor` for mixed ModelSlim shards. Global tensors are still
loaded from the original directory. No config, index, or safetensors file is
written.

For Kimi-K3, 24 is a clean boundary: it contains six complete
`3 KDA + 1 MLA` groups and two complete 12-layer Attention-Residual groups.

## 2. Start Prefill, Decode, and Router

Open three terminals. Prefill and decode must use the same complete
`MODEL_PATH` and the same `NUM_HIDDEN_LAYERS`.

Terminal 1:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/full/Kimi-K3-int4 NUM_HIDDEN_LAYERS=24 \
  ./run_16p_pd_layer24.sh prefill
```

Terminal 2:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/full/Kimi-K3-int4 NUM_HIDDEN_LAYERS=24 \
  ./run_16p_pd_layer24.sh decode
```

Terminal 3, after both model servers are ready:

```bash
cd /home/q00886407/kimi0730/sglang-kimiK3
MODEL_PATH=/path/to/full/Kimi-K3-int4 ./run_16p_pd_layer24.sh router
```

The OpenAI-compatible endpoint is `http://127.0.0.1:6688/v1`.

## 3. Smoke test

```bash
curl http://127.0.0.1:6688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/path/to/full/Kimi-K3-int4",
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

If either role reports out-of-memory while loading, retry with 20 layers;
the same original checkpoint remains unchanged:

```bash
MODEL_PATH=/path/to/full/Kimi-K3-int4 NUM_HIDDEN_LAYERS=20 \
  ./run_16p_pd_layer24.sh prefill
```

If loading succeeds but cache allocation fails, lower
`MAX_TOTAL_TOKENS` to `16384` and `MAX_RUNNING_REQUESTS` to `8`. If there is
substantial free HBM after warmup, increase `MEM_FRACTION_STATIC` gradually,
then increase token capacity.

Runtime prefix-layer loading is intended to make the serving topology runnable.
Removing 69 of 93 trained layers causes a major quality loss; useful model
quality requires continued pretraining or teacher distillation after pruning.
