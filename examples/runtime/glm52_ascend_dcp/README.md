# GLM-5.2 W4A8 在 Ascend A3 上运行 P/D + DCP

## 单机 fake Decode

只测 Decode 时，在一台可见 16 张 A3 卡的机器上运行 `decode_fake.sh`。脚本使用 TP16、Attention DP4、EP16，默认 DCP4。投机参数与真实 Decode 一致，使用 NEXTN、3 步、top-k 1、4 个 draft tokens 和 Spec V2。`fake` 后端模拟 Prefill 的 KV 交接，客户端直接请求本机 Decode，适合测 Decode 容量和性能。生成内容基于合成 KV。

先按下文的容器和 kernel 安装步骤准备这一台机器。进入本分支的仓库目录后执行：

```bash
S=examples/runtime/glm52_ascend_dcp
export MODEL_PATH=/home/weights/GLM-5.2-w4a8

bash "$S/decode_fake.sh" check
bash "$S/decode_fake.sh"

# DCP1 对照，在 DCP4 服务退出后执行
DCP_SIZE=1 bash "$S/decode_fake.sh"
```

服务监听 `127.0.0.1:18004`。可用 `FAKE_PORT`、`MEM_FRACTION_STATIC` 和 `MAX_RUNNING_REQUESTS` 覆盖默认值 `18004`、`0.90` 和 `128`。`MAX_RUNNING_REQUESTS` 按 4 个 Attention DP 组均分。需要显式指定 HCCL/Gloo 网卡时，在本机设置 `D_NIC`。

`check` 会显示投机步数、top-k 和 draft tokens。参数解析会把 NEXTN 归一化为 `EAGLE`。

在同一容器的另一个终端发一条冒烟请求，检查响应的 `meta_info.completion_tokens`：

```bash
python3 -c 'import json; print(json.dumps({"input_ids": [123] * 128, "sampling_params": {"max_new_tokens": 8, "ignore_eos": True, "temperature": 0}}))' |
  curl --noproxy '*' -fsS "http://127.0.0.1:${FAKE_PORT:-18004}/generate" \
    -H 'Content-Type: application/json' --data-binary @-
```

## 双机真实 P/D

使用两台 AArch64 Ascend A3 服务器，每台容器可见 16 张卡。Prefill 使用 TP4 × PP4、Attention DP4。Decode 使用 TP16、Attention DP4、EP16，默认 DCP4。P/D KV 传输使用 Ascend MemFabric SDMA。

P/D 两端使用相同的 NEXTN 配置，3 步、top-k 1、4 个 draft tokens。Decode 开启 Spec V2。

## 1. 准备容器和权重

参考镜像标签为 `20261022-cann9.2.0-a3`。参考环境使用 CANN 9.2.0、Python 3.12、torch 2.13.0、torch-npu 2.13.0、Triton-Ascend 3.6.0 和 sglang-router 0.3.2。容器需带 CANN/ATB、与 CANN 匹配的 DeepEP 和 Ascend MemFabric，并可见本机 NPU。使用 host 网络，在容器内确认 P/D 地址可达。两端都需要挂载 GLM-5.2 W4A8 权重。

在两台机器的容器中拉取本分支。源码构建还需要 Git、CMake、C++ 工具链、pybind11、Python wheel 打包依赖，以及访问 GitHub 和 GitCode 的网络。

```bash
git clone --depth 1 --branch glm52-ascend-dcp https://github.com/qybnb/sglang.git sglang-glm52
cd sglang-glm52
S=examples/runtime/glm52_ascend_dcp
bash "$S/setup-kernel.sh"
```

安装脚本从 `sgl-kernel-npu` 的 `bebbbc732974cf955bd4dc9e54c9e4b8210f5e5f` 构建 A3 wheel，并将它安装到独立的 runtime。也可以在一端按该提交构建 wheel，把 wheel 复制到另一端后执行 `bash "$S/setup-kernel.sh" /path/to/sgl_kernel_npu-2026.9.0-cp312-cp312-linux_aarch64.whl`。wheel 应与容器的 Python、torch/torch-npu 和 CANN 匹配。

默认 runtime 放在 `${XDG_CACHE_HOME:-$HOME/.cache}/sglang/glm52-ascend-dcp`。如需换目录，在安装和启动前都导出同一个 `GLM52_RUNTIME_DIR`。启动脚本通过 `PYTHONPATH` 加载本 checkout 和这个 kernel，无需覆盖容器自带的 SGLang 或 kernel 包。CANN 或 ATB 的环境脚本位置不同，可以设置 `CANN_ENV_SCRIPT` 和 `ATB_ENV_SCRIPT`。

## 2. 设置地址并检查

在两端的启动终端设置以下环境变量。用 `ip -br -4 addr` 确认各自用于通信的地址和网卡。`MODEL_PATH` 使用容器内的实际路径。

```bash
export P_IP=10.10.10.21 D_IP=10.10.10.22
export P_NIC=eth1 D_NIC=eth1
export MODEL_PATH=/home/weights/GLM-5.2-w4a8
S=examples/runtime/glm52_ascend_dcp

# Prefill 节点
bash "$S/prefill.sh" check

# Decode 节点
bash "$S/decode.sh" check
```

`check` 导入 SGLang 和 kernel，并解析当前模型的启动参数。P 应显示 `tp_size=4`、`pp_size=4`、`dcp_size=1`，D 应显示 `tp_size=16`、`attn_dp_size=4`、`dcp_size=4`。需要 DCP1 对照时，用 `DCP_SIZE=1 bash "$S/decode.sh" check` 或 `DCP_SIZE=1 bash "$S/decode.sh"`。

| 变量 | 默认端口 | 用途 |
| --- | --- | --- |
| `P_PORT` | 8003 | Prefill HTTP |
| `P_BOOTSTRAP_PORT` | 8998 | Prefill bootstrap |
| `MF_PORT` | 24667 | Prefill MemFabric store，P/D 均引用 |
| `D_PORT` | 8004 | Decode HTTP |
| `D_DIST_PORT` | 5000 | Decode 分布式初始化 |
| `ROUTER_PORT` | 5599 | Router HTTP |

需要调整时在所有引用该端口的节点和 Router 上同步设置对应变量。`ROUTER_HOST` 默认 `127.0.0.1`。

## 3. 启动和冒烟验证

```bash
# Prefill 节点
bash "$S/prefill.sh"

# Decode 节点
bash "$S/decode.sh"

# P/D 就绪后，在 Prefill 节点另开终端，使用相同地址配置
bash "$S/router.sh"
```

在 Router 节点运行下面的命令，检查 Decode 的 `dcp_size` 和请求实际返回的生成内容。

```bash
curl --noproxy '*' -fsS "http://$D_IP:${D_PORT:-8004}/server_info" |
  python3 -c 'import json, sys; print("dcp_size =", json.load(sys.stdin)["dcp_size"])'

curl --noproxy '*' -fsS "http://127.0.0.1:${ROUTER_PORT:-5599}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-w4a8","messages":[{"role":"user","content":"Reply with the word OK."}],"max_tokens":16,"temperature":0}'
```

Decode 支持按阶段触发的 10 步 rank 0 同步 profiling。输出位置使用请求的 `output_dir`，省略时沿用 `SGLANG_TORCH_PROFILER_DIR` 或 `/tmp`。
