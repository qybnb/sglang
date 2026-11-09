# GLM-5.2 A5 C8 验证

测试分支基于 SGLang PR #49265 的 `61a8d60365a06723e09dcd379c5d70ed23e45c08`。C8 指 `fp8_e4m3` KV cache。Target 和 NEXTN draft 的 KV dtype 分别配置，模型权重和量化方式在对照组之间固定。

## 0. 准备

在已有可运行 SGLang 的 A5 Linux 容器里执行。使用与该提交兼容的 torch、torch_npu、CANN 和 sgl-kernel-npu。整网测试使用含 NEXTN 权重的 GLM-5.2 本地 checkpoint，TP 取能装下模型的卡数。单算子用一张卡，HCCL 合并用两张卡。

```bash
git clone --single-branch --branch npu-dsa-fp8-a5-validation \
  https://github.com/qybnb/sglang.git sglang-glm52-c8
cd sglang-glm52-c8
set -o pipefail
T=test/manual/ascend/glm52_c8
R="$PWD/glm52-c8-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$R"
git rev-parse HEAD | tee "$R/git-sha.txt"
python3 "$T/check_ops.py" env 2>&1 | tee "$R/env.log"
```

脚本从当前 checkout 导入 SGLang。若需构建或安装 Python 依赖，按所用 NPU 镜像的安装方式准备。运行前按本机可用设备设置 `ASCEND_RT_VISIBLE_DEVICES`。算子脚本的 `--device` 使用进程内可见卡号。

DCP C8 需要 `cann_ops_transformer` 注册的 QSFA V2，算子需能在本容器真正启动。先在 `env.log` 看模块路径和 schema，再由下节的 `dcp` 命令确认运行。PCP C8 所需的两个算子通过 `torch_npu` 调用。

## 1. 单算子与两卡合并

按下列顺序逐条运行。某一步失败时保留对应日志，再定位该层。

```bash
python3 "$T/check_ops.py" indexer 2>&1 | tee "$R/indexer.log"
python3 "$T/check_ops.py" qsfa    2>&1 | tee "$R/qsfa.log"
python3 "$T/check_ops.py" dcp     2>&1 | tee "$R/dcp-op.log"
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$T/check_ops.py" dcp2 2>&1 | tee "$R/dcp2-hccl.log"
```

| 用例 | 实际调用与输入 | 通过标准 |
| --- | --- | --- |
| `indexer` | PCP 形状的本地 Q 和逐行页表，FP8 Index K，独立 Q/K scale，128-token 页，短请求和超过 2048 的请求 | CPU 用解量化后的 Q/K 算分数。检查下标范围、数量、唯一性和 `-1` padding，Top-K 入选下标的分数离参考边界最多 0.05 |
| `qsfa` | 复用 SGLang 的 656-byte KV packer 和普通 QSFA helper，传 PCP 形状的 metadata，覆盖满 2048 Top-K | 输出与同一份 packed KV 解量化后的 CPU FP32 参考一致 |
| `dcp` | 在一张卡上串行跑两个 rank 的 QSFA V2，分别走 decode 和 target verify，覆盖跨页边界、空 Top-K、局部 KV 长度为 0 | 分别比较 output、max、sum、LSE。空行严格为 output=0、LSE=-inf。CPU 合并两个 partial 后与全量参考一致 |
| `dcp2` | 两卡 HCCL，QSFA V2 走 verify 形状，调用生产 DCP merge | 每个 rank 的合并输出与全量参考相应的 head slice 一致 |

排查 PCP C8 在 quant indexer 处崩溃时，可单独运行 `ASCEND_LAUNCH_BLOCKING=1 python3 "$T/check_ops.py" indexer-cache`。它无需权重，先用生产 pool 方法 scatter 写入 FP8 K 和 scale，再从第二层的带偏移视图调用 native 算子，分别覆盖全部是短行以及存在超过 2048 token 的行。

默认 output 容差为 `atol=0.02, rtol=0.03`，softmax 统计及 LSE 的绝对容差为 0.03。脚本会打印最大误差，这些值是首轮筛查阈值。遇到失败先保留默认阈值结果，再按需用 `--atol`、`--rtol`、`--lse-atol` 或 `--topk-score-atol` 做诊断。

## 2. 单机整网

设置本机模型路径和 TP。默认模型权重量化为 `modelslim`，按 checkpoint 实际格式调整 `--quantization`，传 `auto` 时交给 SGLang 识别。可以用 `--print-command` 先检查命令。默认监听本机 31265，使用空闲端口。

```bash
MODEL=/models/GLM-5.2
TP=8
COMMON=(--model "$MODEL" --tp "$TP" --cp-size 2 --out-dir "$R/model")

# 先验 target，eager，逐组只改变 CP/DCP 拓扑。
python3 "$T/run_model.py" run "${COMMON[@]}" --cases base pcp dcp

# 再验 NEXTN，先用 BF16 draft KV，再单独打开 draft C8。
python3 "$T/run_model.py" run "${COMMON[@]}" --cases \
  base-nextn-bf16 pcp-nextn-bf16 dcp-nextn-bf16 \
  base-nextn-fp8 pcp-nextn-fp8 dcp-nextn-fp8

# 最后开 decode graph。Prefill 保持 eager，方便归因。
python3 "$T/run_model.py" run "${COMMON[@]}" --graph decode --cases \
  base pcp dcp base-nextn-fp8 pcp-nextn-fp8 dcp-nextn-fp8
```

脚本依次启动各组，在当前终端实时打印 SGLang 服务日志，并同步保存到各组的 `server.log`。请求覆盖短请求、4097-token fresh prefill、追加 257-token 后的 prefix hit，以及长短不同的双请求 batch。检查请求完成、生成长度、有限 logprob、前缀命中、NEXTN verify 次数和 decode graph counter。每组还保存 `command.json`、`server_info.json`、`results.json` 和前后两份 Prometheus 指标。某个服务退出后可以用 `--cases` 单独重跑，给同名用例指定新的 `--out-dir`。

默认 `--chunk-size 8192`。要覆盖跨 chunk prefill，把三组放在同一新输出目录用 `--chunk-size 2048` 重跑。如果有 logprob 的请求不进入某个 graph 路径，可对 eager 与 graph 同时使用 `--skip-logprobs` 做 token 对照，并确认 graph counter 增长。

### 对比结果

```bash
M="$R/model"
python3 "$T/run_model.py" compare "$M/base-eager/results.json" \
  "$M/pcp-eager/results.json" "$M/dcp-eager/results.json"
python3 "$T/run_model.py" compare "$M/base-nextn-bf16-eager/results.json" \
  "$M/pcp-nextn-bf16-eager/results.json" "$M/dcp-nextn-bf16-eager/results.json"
python3 "$T/run_model.py" compare "$M/base-nextn-fp8-eager/results.json" \
  "$M/pcp-nextn-fp8-eager/results.json" "$M/dcp-nextn-fp8-eager/results.json"
python3 "$T/run_model.py" compare "$M/dcp-eager/results.json" \
  "$M/dcp-nextn-bf16-eager/results.json" "$M/dcp-nextn-fp8-eager/results.json"
python3 "$T/run_model.py" compare "$M/dcp-nextn-fp8-eager/results.json" \
  "$M/dcp-nextn-fp8-decode/results.json"
```

`compare` 先核对请求 hash，只在生成历史一致的前缀上比较 logprob。默认 token 完全一致且最大 logprob 差不超过 0.15 才报 `PASS`。出现 `REVIEW` 时退出码为 2，输出保留共同 token 数和误差。需要单独看 draft KV 的影响时，在同一拓扑内比较 `nextn-bf16` 和 `nextn-fp8`，并记录 accept length。这个矩阵验证功能和数值回归，任务准确率另用固定数据集验收。

## 3. 已有多机服务、真实任务与 PD

如果模型需多机才能加载，用已验证的部署命令启动服务，保持 checkpoint、TP/DP/EP、量化和其他配置一致。按上面的矩阵只切换 PCP、DCP、NEXTN 和 draft KV dtype，然后用同一套 client 采集结果。服务要提供 SGLang 原生 `/generate`，开启 NEXTN 时传 `--expect-spec`。需要验证 prefix hit 时加 `--require-cache-hit`。

```bash
URL=http://127.0.0.1:30000
python3 "$T/run_model.py" client --url "$URL" --tokenizer "$MODEL" \
  --out-dir "$R/external-candidate" --expect-spec

# 在基线和候选服务上分别跑同样的真实任务，比较 accuracy 和完成率。
PYTHONPATH="$PWD/python:${PYTHONPATH:-}" python3 -m sglang.test.run_eval \
  --base-url "$URL" --eval-name gsm8k --num-examples 200 --num-threads 8 \
  --temperature 0 --max-tokens 2048 2>&1 | tee "$R/gsm8k-candidate.log"
```

PD 使用现有 Ascend transfer 配置和 router，先跑真实 Prefill + Decode 的 CP1/DCP1 对照，再分别测试 P 侧 PCP、D 侧 DCP，最后测 P 侧 PCP 加 D 侧 DCP。两侧 target KV dtype 和 draft KV dtype 要对应一致。请求指向 router，比较生成结果，保存 P/D 两侧的日志和 cache transfer 证据。单机 runner 不管理多机 worker、router 或 MemFabric 服务。
