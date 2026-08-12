#!/usr/bin/env bash
#
# Four-node, full-layer Kimi-K3 unified serving profile used to compare PCP
# numerical accuracy without PD disaggregation.  Run the same command on all
# four nodes; the local node rank is resolved from CLUSTER_NODES.
#
# Prefer the mode-specific wrappers below for normal use:
#   ./run_4node_full_pcp_off.sh
#   ./run_4node_full_pcp_a2a.sh
#   ./run_4node_full_pcp_fla_ring.sh
#
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
    pcp_off|pcp_a2a|pcp_fla_ring) ;;
    *)
        echo "Usage: $0 {pcp_off|pcp_a2a|pcp_fla_ring}" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

# The previous four-node deployment used the 192.168.25.* fabric.  Override
# CLUSTER_NODES if the new hosts really use a different subnet.
CLUSTER_NODES="${CLUSTER_NODES:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
IFS=',' read -r -a NODE_IPS <<< "${CLUSTER_NODES}"
NNODES="${#NODE_IPS[@]}"
if (( NNODES != 4 )); then
    echo "CLUSTER_NODES must contain exactly four comma-separated IPs" \
        "(got ${NNODES}: ${CLUSTER_NODES})." >&2
    exit 2
fi

CONFIG_ONLY="${CONFIG_ONLY:-0}"
if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
    echo "CONFIG_ONLY must be 0 or 1 (got ${CONFIG_ONLY})." >&2
    exit 2
fi

LOCAL_NODE_IP="${LOCAL_NODE_IP:-}"
if [[ -z "${LOCAL_NODE_IP}" ]]; then
    HOST_IPS=" $(hostname -I 2>/dev/null || true) "
    for candidate in "${NODE_IPS[@]}"; do
        if [[ "${HOST_IPS}" == *" ${candidate} "* ]]; then
            LOCAL_NODE_IP="${candidate}"
            break
        fi
    done
fi
if [[ -z "${LOCAL_NODE_IP}" ]]; then
    echo "This host does not own an IP from CLUSTER_NODES=${CLUSTER_NODES}." >&2
    echo "Set LOCAL_NODE_IP explicitly only when container IP discovery is unavailable." >&2
    exit 2
fi

NODE_RANK=-1
for i in "${!NODE_IPS[@]}"; do
    if [[ "${LOCAL_NODE_IP}" == "${NODE_IPS[$i]}" ]]; then
        NODE_RANK="${i}"
        break
    fi
done
if (( NODE_RANK < 0 )); then
    echo "LOCAL_NODE_IP=${LOCAL_NODE_IP} is not present in ${CLUSTER_NODES}." >&2
    exit 2
fi

MASTER_ADDR="${MASTER_ADDR:-${NODE_IPS[0]}}"
DIST_INIT_PORT="${DIST_INIT_PORT:-5000}"
SERVER_PORT="${SERVER_PORT:-30000}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
TP_SIZE="${TP_SIZE:-$((NNODES * NPUS_PER_NODE))}"
CP_SIZE="${CP_SIZE:-4}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"

for value_name in NPUS_PER_NODE TP_SIZE CP_SIZE; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer (got ${value})." >&2
        exit 2
    fi
done
if (( TP_SIZE != NNODES * NPUS_PER_NODE )); then
    echo "TP_SIZE must equal NNODES * NPUS_PER_NODE for this launcher" \
        "(got TP=${TP_SIZE}, nodes=${NNODES}, NPU/node=${NPUS_PER_NODE})." >&2
    exit 2
fi

PCP_ARGS=()
case "${MODE}" in
    pcp_off)
        ENABLE_PCP=0
        DP_SIZE="${DP_SIZE:-4}"
        EFFECTIVE_CP_SIZE=1
        RUN_TAG="${RUN_TAG:-A_pcp_off_full}"
        ;;
    pcp_a2a)
        ENABLE_PCP=1
        DP_SIZE="${DP_SIZE:-1}"
        EFFECTIVE_CP_SIZE="${CP_SIZE}"
        RUN_TAG="${RUN_TAG:-B_pcp_a2a_allgather_full}"
        PCP_ARGS=(
            --attn-cp-size "${EFFECTIVE_CP_SIZE}"
            --enable-prefill-cp
            --cp-strategy zigzag
            --kda-cp-backend a2a
            --mla-cp-backend allgather
        )
        ;;
    pcp_fla_ring)
        ENABLE_PCP=1
        DP_SIZE="${DP_SIZE:-1}"
        EFFECTIVE_CP_SIZE="${CP_SIZE}"
        RUN_TAG="${RUN_TAG:-C_pcp_fla_ring_full}"
        PCP_ARGS=(
            --attn-cp-size "${EFFECTIVE_CP_SIZE}"
            --enable-prefill-cp
            --cp-strategy zigzag
            --kda-cp-backend fla
            --mla-cp-backend ring
        )
        ;;
esac

if [[ ! "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || (( TP_SIZE % (DP_SIZE * EFFECTIVE_CP_SIZE) != 0 )); then
    echo "TP_SIZE must be divisible by DP_SIZE * CP_SIZE" \
        "(TP=${TP_SIZE}, DP=${DP_SIZE}, CP=${EFFECTIVE_CP_SIZE})." >&2
    exit 2
fi
ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE / EFFECTIVE_CP_SIZE))

if [[ "${CONFIG_ONLY}" != "1" && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
NO_PROXY_LIST="127.0.0.1,localhost,${CLUSTER_NODES//,/,}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${NO_PROXY_LIST}"
export no_proxy="${NO_PROXY}"

LOCAL_IFACE="${LOCAL_IFACE:-}"
if [[ -z "${LOCAL_IFACE}" ]] && command -v ip >/dev/null 2>&1; then
    LOCAL_IFACE="$(
        ip -o -4 addr show | awk -v target="${LOCAL_NODE_IP}" '
            { split($4, addr, "/") }
            addr[1] == target { print $2; exit }
        '
    )"
fi
if [[ -z "${LOCAL_IFACE}" && "${CONFIG_ONLY}" != "1" ]]; then
    echo "Cannot resolve the network interface for ${LOCAL_NODE_IP}." >&2
    echo "Set LOCAL_IFACE to the HCCL/GLOO interface name." >&2
    exit 2
fi
LOCAL_IFACE="${LOCAL_IFACE:-CONFIG_ONLY_IFACE}"

if [[ "${CONFIG_ONLY}" != "1" ]]; then
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

export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${LOCAL_IFACE}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${LOCAL_IFACE}}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-43000}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"
if [[ "${ENABLE_PCP}" == "1" ]]; then
    export SGLANG_ENABLE_CP_V2=1
else
    export SGLANG_ENABLE_CP_V2=0
fi

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
PAGE_SIZE="${PAGE_SIZE:-128}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_full}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_node${NODE_RANK}_$(date '+%Y-%m-%d_%H-%M-%S').log"

LAUNCH_CMD=(
    python3 -m sglang.launch_server
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --dist-init-addr "${MASTER_ADDR}:${DIST_INIT_PORT}"
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${MODEL_PATH}"
    --trust-remote-code
    --attention-backend ascend
    --device npu
    --quantization modelslim
    --dtype bfloat16
    --tp-size "${TP_SIZE}"
    --enable-dp-attention
    --enable-dp-lm-head
    --dp-size "${DP_SIZE}"
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    --moe-a2a-backend deepep
    --deepep-mode auto
    --disable-radix-cache
    --disable-cuda-graph
    --watchdog-timeout 9000
    --host 0.0.0.0
    --port "${SERVER_PORT}"
    "${PCP_ARGS[@]}"
)

echo "Kimi-K3 four-node full-model launch: mode=${MODE}, node=${LOCAL_NODE_IP}," \
    "rank=${NODE_RANK}/${NNODES}, master=${MASTER_ADDR}:${DIST_INIT_PORT}," \
    "TP=${TP_SIZE}, DP=${DP_SIZE}, CP=${EFFECTIVE_CP_SIZE}, attention-TP=${ATTN_TP_SIZE}," \
    "interface=${LOCAL_IFACE}, model=${MODEL_PATH}, log=${LOG_FILE}"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${LAUNCH_CMD[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${LOG_DIR}"
"${LAUNCH_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
