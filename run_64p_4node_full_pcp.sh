#!/usr/bin/env bash
#
# Four-node, 64-rank Kimi-K3 unified serving profile for full-model PCP tests.
# This is intentionally not a PD deployment: one distributed service performs
# both prefill and decode. PCP is active only for extend/prefill forward modes.
#
# The checkpoint config is used as-is. In particular, this script never passes
# --json-model-override-args and therefore never truncates num_hidden_layers.
#
# Usage on every node (run the same profile on all four nodes):
#   ./run_64p_4node_full_pcp.sh off
#   ./run_64p_4node_full_pcp.sh a2a
#   ./run_64p_4node_full_pcp.sh fla
#
# Common overrides:
#   MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
#   NODE_IPS=80.5.17.37,80.5.17.38,80.5.17.33,80.5.17.35
#   NET_IFACE=enp196s0f0
#   NODE_RANK=0  # optional; otherwise inferred from NODE_IPS and hostname -I
#
set -euo pipefail

PROFILE="${1:-}"
case "${PROFILE}" in
    off|a2a|fla) ;;
    *)
        echo "Usage: MODEL_PATH=/path/to/Kimi-K3 $0 {off|a2a|fla}" >&2
        exit 2
        ;;
esac

CONFIG_ONLY="${CONFIG_ONLY:-0}"
if [[ "${CONFIG_ONLY}" != "0" && "${CONFIG_ONLY}" != "1" ]]; then
    echo "CONFIG_ONLY must be 0 or 1 (got ${CONFIG_ONLY})." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
if [[ "${CONFIG_ONLY}" != "1" && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
    exit 2
fi
CHECKPOINT_LAYERS=unknown
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    CHECKPOINT_LAYERS="$("${PYTHON_BIN}" -c \
        'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("text_config", c).get("num_hidden_layers", "unknown"))' \
        "${MODEL_PATH}/config.json")"
fi

# Default rank order for the current four-node unified Kimi-K3 deployment.
# Override NODE_IPS on all four nodes when using a different cluster.
NODE_IPS="${NODE_IPS:-80.5.17.37,80.5.17.38,80.5.17.33,80.5.17.35}"
IFS=',' read -r -a NODE_IP_ARRAY <<< "${NODE_IPS}"
NNODES="${NNODES:-4}"
if (( ${#NODE_IP_ARRAY[@]} != NNODES )); then
    echo "NODE_IPS must contain exactly NNODES=${NNODES} comma-separated addresses" \
        "(got ${#NODE_IP_ARRAY[@]}: ${NODE_IPS})." >&2
    exit 2
fi
if (( NNODES != 4 )); then
    echo "This validation profile requires NNODES=4 (got ${NNODES})." >&2
    exit 2
fi

NODE_RANK="${NODE_RANK:-}"
if [[ -z "${NODE_RANK}" ]]; then
    LOCAL_ADDRESSES=" $(hostname -I 2>/dev/null || true) "
    for i in "${!NODE_IP_ARRAY[@]}"; do
        if [[ "${LOCAL_ADDRESSES}" == *" ${NODE_IP_ARRAY[$i]} "* ]]; then
            NODE_RANK="${i}"
            break
        fi
    done
fi
if [[ ! "${NODE_RANK}" =~ ^[0-3]$ ]]; then
    echo "Cannot infer NODE_RANK from hostname -I and NODE_IPS=${NODE_IPS}." >&2
    echo "Set NODE_RANK=0, 1, 2, or 3 explicitly on this node." >&2
    exit 2
fi

TP_SIZE="${TP_SIZE:-64}"
DP_SIZE="${DP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-4}"
if (( TP_SIZE % NNODES != 0 )); then
    echo "TP_SIZE must be divisible by NNODES (TP=${TP_SIZE}, NNODES=${NNODES})." >&2
    exit 2
fi
LOCAL_WORLD_SIZE=$((TP_SIZE / NNODES))

CP_ARGS=()
case "${PROFILE}" in
    off)
        ENABLE_PCP=0
        ACTIVE_CP_SIZE=1
        KDA_CP_BACKEND=a2a
        MLA_CP_BACKEND=allgather
        export SGLANG_ENABLE_CP_V2=0
        ;;
    a2a)
        ENABLE_PCP=1
        ACTIVE_CP_SIZE="${CP_SIZE}"
        KDA_CP_BACKEND=a2a
        MLA_CP_BACKEND=allgather
        export SGLANG_ENABLE_CP_V2=1
        ;;
    fla)
        ENABLE_PCP=1
        ACTIVE_CP_SIZE="${CP_SIZE}"
        KDA_CP_BACKEND=fla
        MLA_CP_BACKEND=ring
        export SGLANG_ENABLE_CP_V2=1
        ;;
esac

if (( TP_SIZE % (DP_SIZE * ACTIVE_CP_SIZE) != 0 )); then
    echo "TP_SIZE must be divisible by DP_SIZE * CP_SIZE" \
        "(TP=${TP_SIZE}, DP=${DP_SIZE}, CP=${ACTIVE_CP_SIZE})." >&2
    exit 2
fi
ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE / ACTIVE_CP_SIZE))
if (( ENABLE_PCP == 1 && ATTN_TP_SIZE < 2 )); then
    echo "PCP attention TP must be at least 2; got TP${TP_SIZE}/DP${DP_SIZE}/" \
        "CP${ACTIVE_CP_SIZE} => attention-TP${ATTN_TP_SIZE}." >&2
    exit 2
fi

DIST_PORT="${DIST_PORT:-15100}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${NODE_IP_ARRAY[0]}:${DIST_PORT}}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-15000}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
DEEPEP_MODE="${DEEPEP_MODE:-normal}"
case "${DEEPEP_MODE}" in
    auto|normal|low_latency) ;;
    *)
        echo "DEEPEP_MODE must be auto, normal, or low_latency; got ${DEEPEP_MODE}." >&2
        exit 2
        ;;
esac
if (( ENABLE_PCP == 1 )); then
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
else
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
fi
RUN_TAG="${RUN_TAG:-full_4node_${PROFILE}_cp${ACTIVE_CP_SIZE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_full}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_rank${NODE_RANK}_$(date '+%Y-%m-%d_%H-%M-%S').log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${NODE_IPS}"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
# Do not inherit a stale single-node "lo" setting: all four nodes must use the
# same routable interface. NET_IFACE is the explicit override for this script.
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"

if (( ENABLE_PCP == 1 )); then
    CP_ARGS=(
        --attn-cp-size "${ACTIVE_CP_SIZE}"
        --enable-prefill-cp
        --cp-strategy zigzag
        --kda-cp-backend "${KDA_CP_BACKEND}"
        --mla-cp-backend "${MLA_CP_BACKEND}"
    )
fi

SERVER_ARGS=(
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --dist-init-addr "${DIST_INIT_ADDR}"
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
    --dp-size "${DP_SIZE}"
    --enable-dp-lm-head
    --page-size "${PAGE_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    --moe-a2a-backend deepep
    --deepep-mode "${DEEPEP_MODE}"
    --disable-radix-cache
    --disable-cuda-graph
    --watchdog-timeout 9000
    --host "${HOST}"
    --port "${PORT}"
    "${CP_ARGS[@]}"
)

echo "Kimi-K3 four-node FULL checkpoint profile (no layer override):"
echo "  profile=${PROFILE}, rank=${NODE_RANK}/${NNODES}, local-world=${LOCAL_WORLD_SIZE}"
echo "  TP=${TP_SIZE}, DP=${DP_SIZE}, CP=${ACTIVE_CP_SIZE}, attention-TP=${ATTN_TP_SIZE}"
echo "  PCP=${ENABLE_PCP}, KDA=${KDA_CP_BACKEND}, MLA=${MLA_CP_BACKEND}"
echo "  model=${MODEL_PATH}, checkpoint-layers=${CHECKPOINT_LAYERS} (used as-is)"
echo "  dist=${DIST_INIT_ADDR}, interface=${NET_IFACE}"
echo "  chunk=${CHUNKED_PREFILL_SIZE}, max-tokens=${MAX_TOTAL_TOKENS}, mem=${MEM_FRACTION_STATIC}, DeepEP=${DEEPEP_MODE}"
echo "  run-tag=${RUN_TAG}, log=${LOG_FILE}"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"
    printf '\n'
    exit 0
fi

for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ -f "${env_script}" ]]; then
        set +u
        # shellcheck disable=SC1090
        source "${env_script}"
        set -u
    fi
done

mkdir -p "${LOG_DIR}"
"${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
