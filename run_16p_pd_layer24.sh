#!/usr/bin/env bash
#
# Single-node Kimi-K3 PD deployment for 16 Ascend NPUs with 64 GiB HBM each.
# GPU/NPU 0-7: prefill, 8-15: decode.
#
# Usage:
#   MODEL_PATH=/path/to/full/Kimi-K3 ./run_16p_pd_layer24.sh prefill
#   MODEL_PATH=/path/to/full/Kimi-K3 ./run_16p_pd_layer24.sh decode
#   MODEL_PATH=/path/to/full/Kimi-K3 ./run_16p_pd_layer24.sh router
#
set -euo pipefail

ROLE="${1:-}"
if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" && "${ROLE}" != "router" ]]; then
    echo "Usage: MODEL_PATH=/path/to/model $0 {prefill|decode|router}" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required." >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
    exit 2
fi

# SGLang probes its own loopback HTTP endpoint during startup. Proxy variables
# can redirect that warmup request to an unrelated router/proxy and make an
# otherwise healthy server terminate after the warmup timeout.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30100}"
ROUTER_PORT="${ROUTER_PORT:-6688}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MF_STORE_PORT="${MF_STORE_PORT:-24669}"
TP_SIZE="${TP_SIZE:-8}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-24}"
if [[ ! "${NUM_HIDDEN_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_HIDDEN_LAYERS must be a positive integer." >&2
    exit 2
fi
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.84}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
PAGE_SIZE="${PAGE_SIZE:-128}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_layer24_pd}"
mkdir -p "${LOG_DIR}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/nnal/atb/set_env.sh
    set -u
fi

export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-64}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1200}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"
export ASCEND_MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://127.0.0.1:${MF_STORE_PORT}}"

COMMON_ARGS=(
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${MODEL_PATH}"
    --json-model-override-args "{\"text_config\":{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS}}}"
    --trust-remote-code
    --attention-backend ascend
    --device npu
    --quantization modelslim
    --dtype bfloat16
    --tp-size "${TP_SIZE}"
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    --moe-a2a-backend deepep
    --watchdog-timeout 9000
    --host 127.0.0.1
)

case "${ROLE}" in
    prefill)
        LOG_FILE="${LOG_DIR}/prefill_$(date '+%Y-%m-%d_%H-%M-%S').log"
        echo "Starting Kimi-K3 ${NUM_HIDDEN_LAYERS}-layer prefill on NPU 0-7; log=${LOG_FILE}"
        python3 -m sglang.launch_server \
            "${COMMON_ARGS[@]}" \
            --base-gpu-id 0 \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
            --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
            --deepep-mode normal \
            --disable-cuda-graph \
            --port "${PREFILL_PORT}" 2>&1 | tee "${LOG_FILE}"
        ;;
    decode)
        LOG_FILE="${LOG_DIR}/decode_$(date '+%Y-%m-%d_%H-%M-%S').log"
        echo "Starting Kimi-K3 ${NUM_HIDDEN_LAYERS}-layer decode on NPU 8-15; log=${LOG_FILE}"
        python3 -m sglang.launch_server \
            "${COMMON_ARGS[@]}" \
            --base-gpu-id 8 \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend ascend \
            --disaggregation-decode-extra-slots 8 \
            --chunked-prefill-size -1 \
            --deepep-mode low_latency \
            --cuda-graph-bs 1 4 8 16 \
            --port "${DECODE_PORT}" 2>&1 | tee "${LOG_FILE}"
        ;;
    router)
        echo "Starting PD router on port ${ROUTER_PORT}"
        python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --policy cache_aware \
            --prefill "http://127.0.0.1:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
            --decode "http://127.0.0.1:${DECODE_PORT}" \
            --host 0.0.0.0 \
            --port "${ROUTER_PORT}"
        ;;
esac
