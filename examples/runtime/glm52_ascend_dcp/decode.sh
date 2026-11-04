#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${D_IP:?Set D_IP to the Decode node address}"
: "${D_NIC:?Set D_NIC to the Decode communication interface}"
prepare_server
export HCCL_SOCKET_IFNAME="$D_NIC" GLOO_SOCKET_IFNAME="$D_NIC"
export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export DEEPEP_HCCL_BUFFSIZE=400 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export TASK_QUEUE_ENABLE=0

run_server "${1:-serve}" \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode decode \
    --host "$D_IP" --port "$D_PORT" \
    --trust-remote-code \
    --dist-init-addr "$D_IP:$D_DIST_PORT" \
    --nnodes 1 --node-rank 0 \
    --tp-size 16 --dp-size 4 --dcp-size "${DCP_SIZE:-4}" --enable-dp-attention --ep-size 16 \
    --mem-fraction-static 0.84 --max-running-requests 32 \
    --attention-backend ascend --device npu --quantization modelslim \
    --served-model-name glm-5.2-w4a8 \
    --reasoning-parser glm45 --tool-call-parser glm47 \
    --enable-metrics \
    --moe-a2a-backend deepep --deepep-mode low_latency \
    --cuda-graph-max-bs-decode 8 \
    --disaggregation-transfer-backend ascend \
    --watchdog-timeout 9000 --context-length 220000 \
    --disable-shared-experts-fusion --dtype bfloat16 \
    --load-balance-method round_robin \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
