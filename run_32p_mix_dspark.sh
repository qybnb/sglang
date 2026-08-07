#!/bin/bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=1
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=1
# export TRITON_ALL_BLOCKS_PARALLEL=1
MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
# DRAFT_MODEL_PATH=/home/weights/RadixArk-Kimi-K3-DSpark
DRAFT_MODEL_PATH=/home/weights/Kimi-K3-DSpark

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HCCL_BUFFSIZE=2000
export DEEPEP_NORMAL_LONG_SEQ_ROUND=64
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=512

export HCCL_OP_EXPANSION_MODE=AIV
#depdency--依赖
export ASCEND_CUSTOM_OPP_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=/home/z30071866/cann9.1.0/cann-9.1.0-beta.3/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
#switch---融合算子开关
# export SGLANG_NPU_FUSED_MOE_MODE=3

# Decode profiling
export ENABLE_PROFILING=0
export PROFILING_STAGE=decode
export PROFILING_BS=1
export PROFILING_STEP=5

export PYTHONPATH=/home/q00886407/kimi0730/sglang-kimiK3/python:$PYTHONPATH
# export SGLANG_DSPARK_DEBUG_TRACE=${SGLANG_DSPARK_DEBUG_TRACE:-1}

D_IP=('192.168.25.209' '192.168.25.212' '192.168.25.216' '192.168.25.217')
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_RAGGED_VERIFY_MODE=static

        sglang serve \
            --model-loader-extra-config '{"enable_multithread_load": true}' \
            --dist-init-addr 192.168.25.209:5000 --nnodes 4 --node-rank $i \
            --model-path $MODEL_PATH \
            --tokenizer-path $MODEL_PATH \
            --trust-remote-code \
            --attention-backend ascend \
            --device npu \
            --quantization modelslim \
            --dtype bfloat16 \
            --tp-size 64 \
            --disable-radix-cache \
	        --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
            --mem-fraction-static 0.78 \
            --chunked-prefill-size 8192 \
            --cuda-graph-bs 1 4 16 \
            --max-running-requests 64 \
            --host 0.0.0.0 \
            --port 30000 \
            --reasoning-parser kimi_k3 \
	        --moe-a2a-backend deepep \
            --deepep-mode auto \
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \
            --watchdog-timeout 9000 2>&1 | tee \
                "${LOG_DIR}/run_32p_mix_rank${i}_${LOCAL_HOST1}_$(date +%Y-%m-%d_%H-%M-%S).log"
        status=${PIPESTATUS[0]}
        exit "$status"
    fi
done

exit 1

# spec options
            --speculative-algorithm DSPARK \
            --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
            --speculative-dspark-block-size 7 \
            --speculative-draft-attention-backend ascend \
            --speculative-eagle-topk 1 \
            --speculative-draft-model-quantization unquant \

python -m sglang.bench_serving \
  --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset-name random \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --max-concurrency 1 \
  --random-input-len 8000 \
  --random-output-len 1000 \
  --num-prompts 1 \
  --disable-ignore-eos \
  --random-range-ratio 1 


 python -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 127.0.0.1 \
    --port 30000 \
    --model /home/weights/Kimi-K3-w4a8-int-moe \
    --tokenizer /home/weights/Kimi-K3-w4a8-int-moe \
    --served-model-name Kimi-K3-w4a8-int-moe \
    --max-concurrency 1 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 1 \
    --disable-ignore-eos \
    --random-range-ratio 1

    
curl -s http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/weights/Kimi-K3-w4a8-int-8cards-quarot-all-0722",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 20,
    "temperature": 0
  }'


curl --location 'http://127.0.0.1:30000/generate' \
--header 'Content-Type: application/json' \
--data '{
    "text": "The capital of France is?",
    "sampling_params": {
        "temperature": 0.8,
        "max_new_tokens": 100
    }
}'