#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${P_NIC:?Set P_NIC to the Prefill communication interface}"
prepare_server
export HCCL_SOCKET_IFNAME="$P_NIC" GLOO_SOCKET_IFNAME="$P_NIC"
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 TASK_QUEUE_ENABLE=2
export DEEPEP_HCCL_BUFFSIZE=2500 DEEP_USE_ALLTOALL_MODE=1
export SGLANG_PP_LAYER_PARTITION=18,20,24,16

run_server "${1:-serve}" \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode prefill \
    --host "$P_IP" --port "$P_PORT" \
    --disaggregation-bootstrap-port "$P_BOOTSTRAP_PORT" \
    --trust-remote-code \
    --nnodes 1 --node-rank 0 \
    --tp-size 4 --pp-size 4 --dp-size 4 --enable-dp-attention \
    --mem-fraction-static 0.72 \
    --attention-backend ascend --device npu --quantization modelslim \
    --disaggregation-transfer-backend ascend \
    --max-running-requests 16 \
    --served-model-name glm-5.2-w4a8 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --enable-metrics \
    --chunked-prefill-size 16384 --max-prefill-tokens 180000 \
    --context-length 220000 \
    --moe-a2a-backend deepep --deepep-mode normal \
    --disable-shared-experts-fusion --disable-cuda-graph \
    --dtype bfloat16 --enable-dp-lm-head --moe-dense-tp-size 1 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
