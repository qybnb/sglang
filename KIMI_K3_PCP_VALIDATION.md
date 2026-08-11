# Kimi K3 PCP/PD 验证手册

本文用于验证三个不同层次的问题：

1. 功能：Prefill CP、MLA/KDA、PD KV 传输和 Decode 能否稳定完成。
2. 实现精度：相同裁层模型在 PCP 开关前后的 token 和 logprob 是否一致。
3. 性能趋势：在相同硬件、相同层数和相同请求下，PCP 对 TTFT 和输入吞吐的影响。

裁层模型不能用于 GPQA 等业务精度结论，也不能代替完整 93 层模型的最终性能验收。

## 1. 测试配置

使用同一份 checkpoint、相同 `NUM_HIDDEN_LAYERS`、相同 Prefill/Decode
硬件和相同请求，依次重启并测试以下配置：

| 配置 | `ENABLE_PCP` | KDA | MLA | 用途 |
|---|---:|---|---|---|
| A | 0 | 不进入 CP | 不进入 CP | PCP-off 基线 |
| B | 1 | `a2a` | `allgather` | PCP 基础实现 |
| C | 1 | `fla` | `ring` | PCP 优化实现 |

推荐先用前 12 层。12 层包含三组 `3 KDA + 1 MLA`，能够覆盖 KDA、MLA
和一次完整的 Attention-Residual 周期。绝对最小功能测试可以使用 4 层，完整
业务精度必须恢复 93 层。

## 2. 单机 8 NPU

单机拓扑：

```text
NPU 0-3: Prefill, world=4
NPU 4-7: Decode,  world=4

PCP off: Prefill CP1 / attention-TP4 -> Decode CP1 / attention-TP4
PCP on : Prefill CP2 / attention-TP2 -> Decode CP1 / attention-TP4
```

在启动 Prefill、Decode 的终端中设置相同的基础环境：

```bash
cd /home/q00886407/K3-pcp/sglang-kimiK3

export MODEL_PATH=/path/to/Kimi-K3
export NODE_IP=<本机业务IP>
export PREFILL_HOST=${NODE_IP}
export DECODE_HOST=${NODE_IP}
export PREFILL_LOCAL_IP=${NODE_IP}
export DECODE_LOCAL_IP=${NODE_IP}
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=4

export TP_SIZE=4
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=1
export PREFILL_CP_SIZE=2
export NUM_HIDDEN_LAYERS=12
export MAX_TOTAL_TOKENS=8192
export MAX_RUNNING_REQUESTS=4
export CHUNKED_PREFILL_SIZE=2048
export PAGE_SIZE=128
```

每次只选择一个配置：

```bash
# A: PCP off
export ENABLE_PCP=0
export RUN_TAG=A_pcp_off

# B: PCP + A2A/allgather
export ENABLE_PCP=1
export PREFILL_CP_SIZE=2
export KDA_CP_BACKEND=a2a
export MLA_CP_BACKEND=allgather
export RUN_TAG=B_pcp_a2a_allgather

# C: PCP + FLA/ring
export ENABLE_PCP=1
export PREFILL_CP_SIZE=2
export KDA_CP_BACKEND=fla
export MLA_CP_BACKEND=ring
export RUN_TAG=C_pcp_fla_ring
```

正式加载模型前可以仅检查解析后的并行配置，不启动服务：

```bash
CONFIG_ONLY=1 ./run_16p_pd_layer24.sh prefill
CONFIG_ONLY=1 ./run_16p_pd_layer24.sh decode
```

确认输出中的 CP、attention-TP、NPU 范围和 `RUN_TAG` 后，再去掉
`CONFIG_ONLY=1` 正式启动。

三个终端分别运行：

```bash
./run_16p_pd_layer24.sh prefill
```

```bash
./run_16p_pd_layer24.sh decode
```

```bash
./run_16p_pd_layer24.sh router
```

切换 A/B/C 时必须停止并重启 Prefill、Decode 和 Router，不能只重启 Prefill。

如果 12 层仍然 OOM，先使用：

```bash
export NUM_HIDDEN_LAYERS=4
export MAX_TOTAL_TOKENS=4096
export MAX_RUNNING_REQUESTS=2
```

## 3. 双机各 8 NPU

双机测试使用现有默认规模：

```text
Prefill 节点: world=8, DP1
  PCP off: CP1 / attention-TP8
  PCP on : CP4 / attention-TP2

Decode 节点: world=8, DP2, CP1 / attention-TP4
```

仓库已经提供三份 12 层固定配置脚本。模型默认路径为
`/home/weights/Kimi-K3-int4`，IP 默认沿用 `80.5.17.37/38`；只有实际环境不同
时才需要通过 `MODEL_PATH`、`PREFILL_HOST` 和 `DECODE_HOST` 覆盖。

| 配置 | 直接执行的脚本 |
|---|---|
| A：PCP off | `run_16p_pd_layer12_pcp_off.sh` |
| B：A2A/allgather | `run_16p_pd_layer12_pcp_a2a.sh` |
| C：FLA/ring | `run_16p_pd_layer12_pcp_fla_ring.sh` |

例如测试 C 配置，在 Prefill 节点执行：

```bash
./run_16p_pd_layer12_pcp_fla_ring.sh prefill
```

在 Decode 节点执行：

```bash
./run_16p_pd_layer12_pcp_fla_ring.sh decode
```

在 Router 所在节点执行：

```bash
./run_16p_pd_layer12_pcp_fla_ring.sh router
```

切换配置时，三处使用同一份 A、B 或 C 脚本，并完整重启 Prefill、Decode 和
Router。三份脚本固定相同的 TP/DP、层数、page size、cache 上限和 chunk size，
避免终端遗留的环境变量污染 A/B/C 对比。正式启动前也可以执行：

```bash
CONFIG_ONLY=1 ./run_16p_pd_layer12_pcp_off.sh prefill
CONFIG_ONLY=1 ./run_16p_pd_layer12_pcp_a2a.sh prefill
CONFIG_ONLY=1 ./run_16p_pd_layer12_pcp_fla_ring.sh prefill
```

以下环境变量方式保留用于修改机器地址、显存配置或其他实验参数；常规双机
12 层 A/B/C 验证直接使用上面的固定脚本即可。

两台机器都设置：

```bash
export MODEL_PATH=/path/to/Kimi-K3
export PREFILL_HOST=<Prefill节点IP>
export DECODE_HOST=<Decode节点IP>
export TP_SIZE=8
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=2
export NUM_HIDDEN_LAYERS=12
export MAX_TOTAL_TOKENS=16384
export MAX_RUNNING_REQUESTS=8
export CHUNKED_PREFILL_SIZE=4096
export PAGE_SIZE=128
```

Prefill 节点额外设置：

```bash
export PREFILL_LOCAL_IP=${PREFILL_HOST}
export PREFILL_BASE_GPU_ID=0
```

Decode 节点额外设置：

```bash
export DECODE_LOCAL_IP=${DECODE_HOST}
export DECODE_BASE_GPU_ID=0
```

A 配置使用：

```bash
export ENABLE_PCP=0
export RUN_TAG=A_pcp_off
```

B/C 配置增加：

```bash
export ENABLE_PCP=1
export PREFILL_CP_SIZE=4
```

并分别选择 `a2a/allgather` 或 `fla/ring`。启动角色的命令与单机相同。

## 4. 功能和稳定性测试

测试脚本使用 Router 的 `/generate` 接口发送精确 2048-token 请求，每个请求
包含不同的 token 后缀，避免所有请求命中相同前缀缓存：

```bash
python3 scripts/kimi_k3_pcp_validation.py smoke \
  --base-url http://127.0.0.1:6688 \
  --tokenizer "${MODEL_PATH}" \
  --tag "${RUN_TAG}" \
  --input-len 2048 \
  --output-len 32 \
  --num-requests 8 \
  --concurrency 4 \
  --output "logs/kimi_k3_pcp_validation/${RUN_TAG}_smoke.json"
```

通过标准：8 个请求全部成功。然后检查 Prefill 诊断：

```bash
python3 scripts/kimi_k3_pcp_validation.py diag \
  --diag-dir "logs/kimi_k3_layer24_pd/${RUN_TAG}_prefill_kv_diag_<时间>" \
  --role prefill \
  --service-log "logs/kimi_k3_layer24_pd/${RUN_TAG}_prefill_<时间>.log" \
  --output "logs/kimi_k3_pcp_validation/${RUN_TAG}_prefill_diag.json"
```

检查 Decode：

```bash
python3 scripts/kimi_k3_pcp_validation.py diag \
  --diag-dir "logs/kimi_k3_layer24_pd/${RUN_TAG}_decode_kv_diag_<时间>" \
  --role decode \
  --service-log "logs/kimi_k3_layer24_pd/${RUN_TAG}_decode_<时间>.log" \
  --output "logs/kimi_k3_pcp_validation/${RUN_TAG}_decode_diag.json"
```

诊断工具会检查：

- `is_mla_backend=true`
- `kv_buf_groups=2`
- Prefill 至少产生一个 `send_kvcache_plan`
- 没有越界、item-length mismatch、MemFabric 非零返回值和 transfer failed

双机时分别在 Prefill 和 Decode 机器执行对应的诊断命令。

## 5. 实现精度 A/B

在 A 服务运行时采集：

```bash
python3 scripts/kimi_k3_pcp_validation.py accuracy \
  --base-url http://127.0.0.1:6688 \
  --tokenizer "${MODEL_PATH}" \
  --tag A_pcp_off \
  --input-lens 2048,8192 \
  --output-len 32 \
  --prefill-logprob-tokens 64 \
  --top-logprobs 5 \
  --output logs/kimi_k3_pcp_validation/A_accuracy.json
```

重启为 B 或 C 后，使用完全相同的长度参数重新采集：

```bash
python3 scripts/kimi_k3_pcp_validation.py accuracy \
  --base-url http://127.0.0.1:6688 \
  --tokenizer "${MODEL_PATH}" \
  --tag C_pcp_fla_ring \
  --input-lens 2048,8192 \
  --output-len 32 \
  --prefill-logprob-tokens 64 \
  --top-logprobs 5 \
  --output logs/kimi_k3_pcp_validation/C_accuracy.json
```

比较结果：

```bash
python3 scripts/kimi_k3_pcp_validation.py compare-accuracy \
  logs/kimi_k3_pcp_validation/A_accuracy.json \
  logs/kimi_k3_pcp_validation/C_accuracy.json \
  --output logs/kimi_k3_pcp_validation/A_vs_C_accuracy.json
```

默认判据：

- 首个输出 token 一致
- 输出 token 匹配率至少 95%
- Prefill 尾部 token logprob 最大绝对差不超过 0.15
- 相同输出 token 的 logprob 最大绝对差不超过 0.15

阈值是初始工程门槛，不是最终精度规范。应先记录 A 自身重复运行的自然波动，再决定
是否收紧。Prefill logprob 的比较比生成文本更重要，因为生成一旦在接近打平的 token
处发生分叉，后续 token 就不再处于相同自回归输入上。

## 6. 性能 A/B

测试脚本复用 `sglang.benchmark.serving`，使用原生 `/generate`、精确 token IDs、
固定输出长度和 `ignore_eos`，主要观察 TTFT 和输入吞吐。

A 配置：

```bash
python3 scripts/kimi_k3_pcp_validation.py perf \
  --base-url http://127.0.0.1:6688 \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --tag A_pcp_off \
  --input-lens 1024,4096,8192 \
  --output-lens 1,32 \
  --concurrencies 1,4 \
  --num-prompts 8 \
  --output-file logs/kimi_k3_pcp_validation/A_perf.jsonl
```

C 配置使用相同矩阵：

```bash
python3 scripts/kimi_k3_pcp_validation.py perf \
  --base-url http://127.0.0.1:6688 \
  --model "${MODEL_PATH}" \
  --tokenizer "${MODEL_PATH}" \
  --tag C_pcp_fla_ring \
  --input-lens 1024,4096,8192 \
  --output-lens 1,32 \
  --concurrencies 1,4 \
  --num-prompts 8 \
  --output-file logs/kimi_k3_pcp_validation/C_perf.jsonl
```

比较：

```bash
python3 scripts/kimi_k3_pcp_validation.py compare-perf \
  logs/kimi_k3_pcp_validation/A_perf.jsonl \
  logs/kimi_k3_pcp_validation/C_perf.jsonl \
  --output logs/kimi_k3_pcp_validation/A_vs_C_perf.json
```

输出中的正收益含义：

- `ttft_gain_pct > 0`：候选配置 TTFT 更低
- `input_throughput_gain_pct > 0`：候选配置输入吞吐更高
- `request_throughput_gain_pct > 0`：候选配置请求吞吐更高
- `tpot_gain_pct` 理论上应接近 0，因为 Decode 没有开启 CP

脚本会在 JSONL 中保留逐请求错误；任一对比配置有请求失败时，`compare-perf`
直接返回失败，不使用残缺请求计算出的吞吐作为收益结论。

建议每个配置完整重复三次，比较中位数。短输入可能因为通信开销而没有收益，PCP
是否有价值主要看 8K、16K、32K 及目标并发下的曲线。裁层结果只能作为趋势，最终
结论必须在完整 93 层、相同总硬件预算下重跑。

## 7. 结果归档

每个配置至少保存：

```text
RUN_TAG_prefill_*.log
RUN_TAG_decode_*.log
RUN_TAG_prefill_kv_diag_*/
RUN_TAG_decode_kv_diag_*/
RUN_TAG_smoke.json
RUN_TAG_accuracy.json
RUN_TAG_perf.jsonl
RUN_TAG_perf.jsonl.log
```

不要把 A/B/C 的结果写入同一个不带 tag 的目录。对比前确认两边使用同一 Git
commit、checkpoint、裁层数、page size、请求集合和硬件数量。
