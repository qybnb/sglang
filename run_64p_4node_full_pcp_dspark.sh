#!/usr/bin/env bash
#
# Four-node Kimi-K3 PCP prefill + CP-off DSpark decode validation profile.
# This is intentionally separate from run_64p_4node_full_pcp.sh so enabling
# speculative decoding cannot change the established PCP-only workflow.
#
# Run the same profile on every node (normally via the companion controller):
#   ./run_64p_4node_full_pcp_dspark.sh a2a
#   ./run_64p_4node_full_pcp_dspark.sh fla

set -euo pipefail

PROFILE="${1:-}"
case "${PROFILE}" in
    a2a|fla) ;;
    *)
        echo "Usage: MODEL_PATH=/path/to/Kimi-K3 DRAFT_MODEL_PATH=/path/to/DSpark $0 {a2a|fla}" >&2
        exit 2
        ;;
esac

CONFIG_ONLY="${CONFIG_ONLY:-0}"
case "${CONFIG_ONLY}" in
    0|1) ;;
    *) echo "CONFIG_ONLY must be 0 or 1 (got ${CONFIG_ONLY})." >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
if [[ "${CONFIG_ONLY}" != "1" ]]; then
    for checkpoint in "${MODEL_PATH}" "${DRAFT_MODEL_PATH}"; do
        if [[ ! -f "${checkpoint}/config.json" ]]; then
            echo "Invalid checkpoint; config.json not found: ${checkpoint}" >&2
            exit 2
        fi
    done
fi

NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
IFS=',' read -r -a NODE_IP_ARRAY <<< "${NODE_IPS}"
NNODES="${NNODES:-4}"
if (( ${#NODE_IP_ARRAY[@]} != NNODES || NNODES != 4 )); then
    echo "This profile requires exactly four NODE_IPS (got ${NODE_IPS})." >&2
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
    echo "Cannot infer NODE_RANK; set NODE_RANK=0,1,2,3 explicitly." >&2
    exit 2
fi

TP_SIZE="${TP_SIZE:-64}"
DP_SIZE="${DP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-4}"
if (( TP_SIZE % NNODES != 0 )); then
    echo "TP_SIZE must be divisible by NNODES." >&2
    exit 2
fi
if (( TP_SIZE % (DP_SIZE * CP_SIZE) != 0 )); then
    echo "TP_SIZE must be divisible by DP_SIZE * CP_SIZE." >&2
    exit 2
fi
LOCAL_WORLD_SIZE=$((TP_SIZE / NNODES))
ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE / CP_SIZE))
if (( ATTN_TP_SIZE < 2 )); then
    echo "PCP DSpark requires attention-TP >= 2; got ${ATTN_TP_SIZE}." >&2
    exit 2
fi

case "${PROFILE}" in
    a2a)
        KDA_CP_BACKEND=a2a
        MLA_CP_BACKEND=allgather
        ;;
    fla)
        KDA_CP_BACKEND=fla
        MLA_CP_BACKEND=ring
        ;;
esac

# Dedicated defaults keep this service separate from the PCP-only service.
DIST_PORT="${DIST_PORT:-15110}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${NODE_IP_ARRAY[0]}:${DIST_PORT}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-15010}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
DEEPEP_MODE="${DEEPEP_MODE:-normal}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
DSPARK_DRAFT_QUANTIZATION="${DSPARK_DRAFT_QUANTIZATION:-unquant}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
case "${DEEPEP_MODE}" in auto|normal|low_latency) ;; *) exit 2 ;; esac
case "${DISABLE_CUDA_GRAPH}" in 0|1) ;; *) exit 2 ;; esac

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--disable-cuda-graph)
else
    read -r -a CUDA_GRAPH_BS_ARRAY <<< "${CUDA_GRAPH_BS:-1 4 16}"
    CUDA_GRAPH_ARGS=(--cuda-graph-bs "${CUDA_GRAPH_BS_ARRAY[@]}")
fi

RUN_TAG="${RUN_TAG:-full_4node_dspark_${PROFILE}_cp${CP_SIZE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_full_pcp_dspark}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_rank${NODE_RANK}_$(date '+%Y-%m-%d_%H-%M-%S').log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${NODE_IPS}"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

export SGLANG_ENABLE_CP_V2=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1200}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"

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
    --attn-cp-size "${CP_SIZE}"
    --enable-prefill-cp
    --cp-strategy zigzag
    --kda-cp-backend "${KDA_CP_BACKEND}"
    --mla-cp-backend "${MLA_CP_BACKEND}"
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
    "${CUDA_GRAPH_ARGS[@]}"
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    --speculative-draft-attention-backend "${DSPARK_DRAFT_ATTENTION_BACKEND}"
    --speculative-eagle-topk 1
    --speculative-draft-model-quantization "${DSPARK_DRAFT_QUANTIZATION}"
    --watchdog-timeout 9000
    --host "${HOST}"
    --port "${PORT}"
)

echo "Kimi-K3 four-node PCP + DSpark profile:"
echo "  profile=${PROFILE}, rank=${NODE_RANK}/${NNODES}, local-world=${LOCAL_WORLD_SIZE}"
echo "  TP=${TP_SIZE}, DP=${DP_SIZE}, CP=${CP_SIZE}, attention-TP=${ATTN_TP_SIZE}"
echo "  target=${MODEL_PATH}, draft=${DRAFT_MODEL_PATH}, block=${DSPARK_BLOCK_SIZE}"
echo "  KDA=${KDA_CP_BACKEND}, MLA=${MLA_CP_BACKEND}, ragged=${SGLANG_RAGGED_VERIFY_MODE}"
echo "  dist=${DIST_INIT_ADDR}, port=${PORT}, interface=${NET_IFACE}"
echo "  chunk=${CHUNKED_PREFILL_SIZE}, max-tokens=${MAX_TOTAL_TOKENS}, mem=${MEM_FRACTION_STATIC}"
echo "  HCCL_BUFFSIZE=${HCCL_BUFFSIZE}, DeepEP=${DEEPEP_MODE}, log=${LOG_FILE}"

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
