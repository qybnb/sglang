#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

max_running=${MAX_RUNNING_REQUESTS:-128}
if [[ ! "$max_running" =~ ^[1-9][0-9]*$ ]] || (( max_running % 4 != 0 )); then
    echo "MAX_RUNNING_REQUESTS must be a positive multiple of 4" >&2
    exit 2
fi

prepare_server_runtime
unset ASCEND_MF_STORE_URL ASCEND_MF_TRANSFER_PROTOCOL
if [[ -n "${D_NIC:-}" ]]; then
    export HCCL_SOCKET_IFNAME="$D_NIC" GLOO_SOCKET_IFNAME="$D_NIC"
fi
export TASK_QUEUE_ENABLE=0 DEEPEP_HCCL_BUFFSIZE=400
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$((max_running / 4))

run_server "${1:-serve}" \
    --model-path "$MODEL_PATH" --served-model-name glm-5.2-fake-capacity \
    --host 127.0.0.1 --port "${FAKE_PORT:-18004}" \
    --disaggregation-mode decode --disaggregation-transfer-backend fake \
    --nnodes 1 --node-rank 0 \
    --tp-size 16 --attn-dp-size 4 --ep-size 16 --dcp-size "${DCP_SIZE:-4}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
    --max-running-requests "$max_running" \
    --attention-backend ascend --device npu --quantization modelslim \
    --moe-a2a-backend deepep --deepep-mode low_latency \
    --context-length 220000 --cuda-graph-max-bs-decode 8 \
    --watchdog-timeout 9000 --disable-shared-experts-fusion \
    --dtype bfloat16 --trust-remote-code \
    --enable-metrics --decode-log-interval 1
