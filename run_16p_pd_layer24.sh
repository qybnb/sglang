#!/usr/bin/env bash
#
# Two-node Kimi-K3 PD deployment with 8 Ascend NPUs per node.
# 80.5.17.37, NPU 0-7: prefill.
# 80.5.17.38, NPU 0-7: decode.
#
# Usage:
#   MODEL_PATH=/path/to/full/Kimi-K3 ./run_16p_pd_layer24.sh prefill
#   MODEL_PATH=/path/to/full/Kimi-K3 ./run_16p_pd_layer24.sh decode
#   ./run_16p_pd_layer24.sh router
#
# A/B validation:
#   ENABLE_PCP=0 RUN_TAG=A_pcp_off ... ./run_16p_pd_layer24.sh prefill
#   ENABLE_PCP=1 RUN_TAG=C_pcp_optimized \
#     KDA_CP_BACKEND=fla MLA_CP_BACKEND=ring ... \
#     ./run_16p_pd_layer24.sh prefill
#
set -euo pipefail

ROLE="${1:-}"
if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" && "${ROLE}" != "router" ]]; then
    echo "Usage: MODEL_PATH=/path/to/model $0 {prefill|decode} | $0 router" >&2
    exit 2
fi
CONFIG_ONLY="${CONFIG_ONLY:-0}"
if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
    echo "CONFIG_ONLY must be 0 or 1 (got ${CONFIG_ONLY})." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

PREFILL_HOST="${PREFILL_HOST:-80.5.17.37}"
DECODE_HOST="${DECODE_HOST:-80.5.17.38}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-0.0.0.0}"
# Set these only when the address visible inside the process/container must be
# selected explicitly. Otherwise SGLang auto-detects a bindable local address.
PREFILL_LOCAL_IP="${PREFILL_LOCAL_IP:-}"
DECODE_LOCAL_IP="${DECODE_LOCAL_IP:-}"
PREFILL_BASE_GPU_ID="${PREFILL_BASE_GPU_ID:-0}"
DECODE_BASE_GPU_ID="${DECODE_BASE_GPU_ID:-0}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ "${ROLE}" != "router" && "${CONFIG_ONLY}" != "1" ]]; then
    if [[ -z "${MODEL_PATH}" ]]; then
        echo "MODEL_PATH is required." >&2
        exit 2
    fi
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
        exit 2
    fi
fi

# SGLang probes its own loopback HTTP endpoint during startup. Proxy variables
# can redirect that warmup request to an unrelated router/proxy and make an
# otherwise healthy server terminate after the warmup timeout.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${PREFILL_HOST},${DECODE_HOST}"
export no_proxy="${NO_PROXY}"

PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_PORT="${DECODE_PORT:-30100}"
ROUTER_PORT="${ROUTER_PORT:-6688}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MF_STORE_PORT="${MF_STORE_PORT:-24669}"
TP_SIZE="${TP_SIZE:-8}"
ENABLE_PCP="${ENABLE_PCP:-1}"
if [[ "${ENABLE_PCP}" != "0" && "${ENABLE_PCP}" != "1" ]]; then
    echo "ENABLE_PCP must be 0 or 1 (got ${ENABLE_PCP})." >&2
    exit 2
fi
# Prefill CP consumes ranks from the attention-parallel dimension. TP8/DP2/CP4
# would collapse attention TP to 1 and replicate the full attention weights on
# every rank, which does not fit K3-24L on a 64 GiB NPU. Keep two-way attention
# TP on PCP prefill while preserving the validated DP2 CP-off baseline.
if [[ "${ENABLE_PCP}" == "1" ]]; then
    PREFILL_DP_SIZE_DEFAULT=1
else
    PREFILL_DP_SIZE_DEFAULT="${DP_SIZE:-2}"
fi
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-${PREFILL_DP_SIZE_DEFAULT}}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-${DP_SIZE:-2}}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-24}"
if [[ ! "${NUM_HIDDEN_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_HIDDEN_LAYERS must be a positive integer." >&2
    exit 2
fi
for role_dp_name in PREFILL_DP_SIZE DECODE_DP_SIZE; do
    role_dp_size="${!role_dp_name}"
    if [[ ! "${role_dp_size}" =~ ^[1-9][0-9]*$ ]] || (( TP_SIZE % role_dp_size != 0 )); then
        echo "${role_dp_name} must be positive and evenly divide TP_SIZE" \
            "(got ${role_dp_name}=${role_dp_size}, TP_SIZE=${TP_SIZE})." >&2
        exit 2
    fi
done
PREFILL_ATTN_PARALLEL_SIZE=$((TP_SIZE / PREFILL_DP_SIZE))
if [[ "${ENABLE_PCP}" == "1" ]]; then
    PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-4}"
    if [[ ! "${PREFILL_CP_SIZE}" =~ ^[2-9][0-9]*$ ]] \
        || (( PREFILL_ATTN_PARALLEL_SIZE % PREFILL_CP_SIZE != 0 )); then
        echo "With ENABLE_PCP=1, PREFILL_CP_SIZE must be greater than 1 and divide" \
            "TP_SIZE / PREFILL_DP_SIZE (got CP=${PREFILL_CP_SIZE}," \
            "TP=${TP_SIZE}, Prefill-DP=${PREFILL_DP_SIZE})." >&2
        exit 2
    fi
else
    # Keep the same total Prefill world size for a fair A/B comparison.  CP1
    # means all attention-parallel ranks are ordinary attention-TP ranks.
    PREFILL_CP_SIZE=1
fi
PREFILL_ATTN_TP_SIZE=$((PREFILL_ATTN_PARALLEL_SIZE / PREFILL_CP_SIZE))
if [[ "${ENABLE_PCP}" == "1" ]] && (( PREFILL_ATTN_TP_SIZE < 2 )); then
    echo "Prefill attention TP must be at least 2 for K3-24L on 64 GiB NPU; " \
        "TP${TP_SIZE}/DP${PREFILL_DP_SIZE}/CP${PREFILL_CP_SIZE} would replicate full KDA attention weights." >&2
    exit 2
fi
# Prefill CP increases resident attention weights and owns three HCCL
# communicators.  Keep its 93% cache profile isolated from the CP-off baseline,
# which uses the pre-PCP 84% default.
if [[ "${ENABLE_PCP}" == "1" ]]; then
    PREFILL_MEM_FRACTION_STATIC_DEFAULT=0.93
else
    PREFILL_MEM_FRACTION_STATIC_DEFAULT=0.84
fi
PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-${MEM_FRACTION_STATIC:-${PREFILL_MEM_FRACTION_STATIC_DEFAULT}}}"
DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-${MEM_FRACTION_STATIC:-0.84}}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MLA_CP_BACKEND="${MLA_CP_BACKEND:-allgather}"
if [[ "${MLA_CP_BACKEND}" != "allgather" && "${MLA_CP_BACKEND}" != "ring" ]]; then
    echo "MLA_CP_BACKEND must be allgather or ring (got ${MLA_CP_BACKEND})." >&2
    exit 2
fi
KDA_CP_BACKEND="${KDA_CP_BACKEND:-fla}"
if [[ "${KDA_CP_BACKEND}" != "a2a" && "${KDA_CP_BACKEND}" != "fla" ]]; then
    echo "KDA_CP_BACKEND must be a2a or fla (got ${KDA_CP_BACKEND})." >&2
    exit 2
fi
PAGE_SIZE="${PAGE_SIZE:-128}"
RUN_TAG="${RUN_TAG:-pcp${ENABLE_PCP}_cp${PREFILL_CP_SIZE}_${KDA_CP_BACKEND}_${MLA_CP_BACKEND}}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_layer24_pd}"
if [[ "${ROLE}" != "router" ]]; then
    # Direct JSONL diagnostics bypass Python/stdout logging and use one file
    # per scheduler process.  Keep each launch separate for easy collection.
    export SGLANG_ASCEND_KV_DIAG_DIR="${SGLANG_ASCEND_KV_DIAG_DIR:-${LOG_DIR}/${RUN_TAG}_${ROLE}_kv_diag_$(date '+%Y-%m-%d_%H-%M-%S')}"
fi

if [[ "${CONFIG_ONLY}" != "1" ]]; then
    mkdir -p "${LOG_DIR}"
    if [[ "${ROLE}" != "router" ]]; then
        mkdir -p "${SGLANG_ASCEND_KV_DIAG_DIR}"
    fi
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
fi

export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-64}"
# Prefill always keeps DP-attention token scatter for DeepEP, matching the last
# validated pre-PCP Kimi configuration.  PCP additionally shards each chunk
# across CP.  DeepEP's capacity is per rank, so derive its rounds from the
# resulting local token count rather than the scheduler's global chunk size.
if [[ "${ROLE}" == "prefill" ]]; then
    if (( CHUNKED_PREFILL_SIZE <= 0 )); then
        echo "Prefill requires a positive CHUNKED_PREFILL_SIZE" \
            "(got ${CHUNKED_PREFILL_SIZE})." >&2
        exit 2
    fi
    # DP-attention first divides the configured chunk across DP replicas and
    # attention TP then scatters each replica's tokens.  PCP DP1 similarly
    # divides over CP x attention-TP.  Both result in chunk / TP local tokens.
    PREFILL_LOCAL_MAX_TOKENS="$(( \
        (CHUNKED_PREFILL_SIZE + TP_SIZE - 1) / TP_SIZE \
    ))"
    DEEPEP_PREFILL_TOKENS_PER_ROUND="${DEEPEP_PREFILL_TOKENS_PER_ROUND:-512}"
    if [[ ! "${DEEPEP_PREFILL_TOKENS_PER_ROUND}" =~ ^[1-9][0-9]*$ ]]; then
        echo "DEEPEP_PREFILL_TOKENS_PER_ROUND must be a positive integer" \
            "(got ${DEEPEP_PREFILL_TOKENS_PER_ROUND})." >&2
        exit 2
    fi
    DEEPEP_PREFILL_ROUNDS="$(( \
        (PREFILL_LOCAL_MAX_TOKENS + DEEPEP_PREFILL_TOKENS_PER_ROUND - 1) \
        / DEEPEP_PREFILL_TOKENS_PER_ROUND \
    ))"
    # Set the operator-facing variables from one internally consistent pair;
    # stale generic DeepEP variables must not override the computed local size.
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_PREFILL_TOKENS_PER_ROUND}"
    export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_PREFILL_ROUNDS}"
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
    if [[ "${ENABLE_PCP}" == "1" ]]; then
        export HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE:-400}}"
        PREFILL_TOKEN_LAYOUT="pcp_scattered"
    else
        export HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE:-1200}}"
        PREFILL_TOKEN_LAYOUT="dp_attn_scattered"
    fi
else
    export HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE:-1200}}"
fi
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"
# Both PD nodes must register with the same MemFabric config store.  The store
# is created by the rank-0 prefill process and therefore lives on PREFILL_HOST.
export ASCEND_MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://${PREFILL_HOST}:${MF_STORE_PORT}}"
# A3/SuperPod uses the default SDMA MemFabric path. Atlas A2 deployments with
# a configured RoCE fabric can override this with device_rdma.
export ASCEND_MF_TRANSFER_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-sdma}"
export ASCEND_MF_LOG_LEVEL="${ASCEND_MF_LOG_LEVEL:-1}"
if [[ "${ROLE}" == "prefill" && -n "${PREFILL_LOCAL_IP}" ]]; then
    export SGLANG_HOST_IP="${PREFILL_LOCAL_IP}"
elif [[ "${ROLE}" == "decode" && -n "${DECODE_LOCAL_IP}" ]]; then
    export SGLANG_HOST_IP="${DECODE_LOCAL_IP}"
else
    unset SGLANG_HOST_IP
fi
# K3 PCP uses the model-boundary CP-v2 path: shard embeddings before the text
# backbone and gather hidden states before logits. Explicitly disable it in the
# CP1 baseline so the A/B run cannot accidentally enter a CP implementation.
if [[ "${ENABLE_PCP}" == "1" && "${ROLE}" == "prefill" ]]; then
    export SGLANG_ENABLE_CP_V2="${SGLANG_ENABLE_CP_V2:-1}"
else
    export SGLANG_ENABLE_CP_V2=0
fi

PREFILL_DP_ARGS=(
    --enable-dp-attention
    --enable-dp-lm-head
    --dp-size "${PREFILL_DP_SIZE}"
)
PREFILL_CP_ARGS=()
if [[ "${ENABLE_PCP}" == "1" ]]; then
    PREFILL_CP_ARGS=(
        --attn-cp-size "${PREFILL_CP_SIZE}"
        --enable-prefill-cp
        --cp-strategy zigzag
        --mla-cp-backend "${MLA_CP_BACKEND}"
        --kda-cp-backend "${KDA_CP_BACKEND}"
    )
fi

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
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    --moe-a2a-backend deepep
    --watchdog-timeout 9000
)

case "${ROLE}" in
    prefill)
        LOG_FILE="${LOG_DIR}/${RUN_TAG}_prefill_$(date '+%Y-%m-%d_%H-%M-%S').log"
        echo "Starting Kimi-K3 ${NUM_HIDDEN_LAYERS}-layer prefill at ${PREFILL_HOST} (bind=${PREFILL_BIND_HOST}) on NPU ${PREFILL_BASE_GPU_ID}-$((PREFILL_BASE_GPU_ID + TP_SIZE - 1))" \
            "(RUN_TAG=${RUN_TAG}, PCP=${ENABLE_PCP}, TP=${TP_SIZE}, DP=${PREFILL_DP_SIZE}, CP=${PREFILL_CP_SIZE}, attention-TP=${PREFILL_ATTN_TP_SIZE}," \
            "MLA-CP=${MLA_CP_BACKEND}, KDA-CP=${KDA_CP_BACKEND}, mem-fraction=${PREFILL_MEM_FRACTION_STATIC}," \
            "HCCL=${HCCL_BUFFSIZE}MB, MF=${ASCEND_MF_TRANSFER_PROTOCOL}," \
            "token-layout=${PREFILL_TOKEN_LAYOUT}, local-max-token=${PREFILL_LOCAL_MAX_TOKENS}," \
            "DeepEP-round=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS}x${DEEPEP_NORMAL_LONG_SEQ_ROUND});" \
            "log=${LOG_FILE}; kv_diag=${SGLANG_ASCEND_KV_DIAG_DIR}"
        if [[ "${CONFIG_ONLY}" == "1" ]]; then
            exit 0
        fi
        python3 -m sglang.launch_server \
            "${COMMON_ARGS[@]}" \
            --host "${PREFILL_BIND_HOST}" \
            --base-gpu-id "${PREFILL_BASE_GPU_ID}" \
            --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC}" \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
            "${PREFILL_DP_ARGS[@]}" \
            "${PREFILL_CP_ARGS[@]}" \
            --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
            --deepep-mode normal \
            --disable-cuda-graph \
            --port "${PREFILL_PORT}" 2>&1 | tee "${LOG_FILE}"
        ;;
    decode)
        LOG_FILE="${LOG_DIR}/${RUN_TAG}_decode_$(date '+%Y-%m-%d_%H-%M-%S').log"
        echo "Starting Kimi-K3 ${NUM_HIDDEN_LAYERS}-layer decode at ${DECODE_HOST} (bind=${DECODE_BIND_HOST}) on NPU ${DECODE_BASE_GPU_ID}-$((DECODE_BASE_GPU_ID + TP_SIZE - 1))" \
            "(RUN_TAG=${RUN_TAG}, PCP=0, TP=${TP_SIZE}, DP=${DECODE_DP_SIZE}, CP=1, attention-TP=$((TP_SIZE / DECODE_DP_SIZE))," \
            "MF=${ASCEND_MF_TRANSFER_PROTOCOL}); log=${LOG_FILE}; kv_diag=${SGLANG_ASCEND_KV_DIAG_DIR}"
        if [[ "${CONFIG_ONLY}" == "1" ]]; then
            exit 0
        fi
        python3 -m sglang.launch_server \
            "${COMMON_ARGS[@]}" \
            --host "${DECODE_BIND_HOST}" \
            --base-gpu-id "${DECODE_BASE_GPU_ID}" \
            --enable-dp-attention \
            --enable-dp-lm-head \
            --dp-size "${DECODE_DP_SIZE}" \
            --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC}" \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend ascend \
            --disaggregation-decode-extra-slots 8 \
            --chunked-prefill-size -1 \
            --deepep-mode low_latency \
            --cuda-graph-bs 8 \
            --port "${DECODE_PORT}" 2>&1 | tee "${LOG_FILE}"
        ;;
    router)
        echo "Starting PD router for RUN_TAG=${RUN_TAG} at ${ROUTER_HOST}:${ROUTER_PORT}; prefill=${PREFILL_HOST}:${PREFILL_PORT}, decode=${DECODE_HOST}:${DECODE_PORT}"
        if [[ "${CONFIG_ONLY}" == "1" ]]; then
            exit 0
        fi
        python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --policy cache_aware \
            --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
            --decode "http://${DECODE_HOST}:${DECODE_PORT}" \
            --host "${ROUTER_HOST}" \
            --port "${ROUTER_PORT}"
        ;;
esac
