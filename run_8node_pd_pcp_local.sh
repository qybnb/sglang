#!/usr/bin/env bash
#
# Node-local Kimi-K3 PD launcher. The default is eight 16-NPU nodes (4P +
# 4D), while wrappers may select asymmetric P/D node counts and TP sizes.
#
# This script never uses SSH and never starts or stops a process on another
# host. Run it manually inside the same container on every node:
#
#   P0: NODE_RANK=0 LOCAL_IP=<P0> NET_IFACE=<if0> MODEL_PATH=<path0> \
#         ./run_8node_pd_pcp_local.sh prefill
#   P1: NODE_RANK=1 LOCAL_IP=<P1> NET_IFACE=<if1> MODEL_PATH=<path1> \
#         ./run_8node_pd_pcp_local.sh prefill
#   P2: NODE_RANK=2 LOCAL_IP=<P2> NET_IFACE=<if2> MODEL_PATH=<path2> \
#         ./run_8node_pd_pcp_local.sh prefill
#   P3: NODE_RANK=3 LOCAL_IP=<P3> NET_IFACE=<if3> MODEL_PATH=<path3> \
#         ./run_8node_pd_pcp_local.sh prefill
#
#   D0-D3 use the same NODE_RANK=0-3 convention with the decode role. When
#   ENABLE_DSPARK=1, each decode node must also set its local
#   DRAFT_MODEL_PATH. Start the router in a second shell on P0 after both
#   rank-0 /health endpoints are ready.
#
# Shared on all eight nodes:
#   PREFILL_IPS=<P0>,<P1>,<P2>,<P3>
#   DECODE_IPS=<D0>,<D1>,<D2>,<D3>

set -euo pipefail

ROLE="${1:-}"
case "${ROLE}" in
    prefill|decode|router) ;;
    *)
        echo "Usage: $0 {prefill|decode|router}" >&2
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

# A node may use a different kernel checkout. When omitted, use the kernel
# package already installed in the container.
if [[ -n "${KERNEL_CODE_ROOT:-}" ]]; then
    KERNEL_PYTHON_ROOT="${KERNEL_CODE_ROOT}/python/sgl_kernel_npu"
    if [[ ! -f "${KERNEL_PYTHON_ROOT}/sgl_kernel_npu/__init__.py" ]]; then
        echo "Invalid KERNEL_CODE_ROOT: ${KERNEL_CODE_ROOT}" >&2
        echo "Expected: ${KERNEL_PYTHON_ROOT}/sgl_kernel_npu/__init__.py" >&2
        exit 2
    fi
    if [[ "${CONFIG_ONLY}" != "1" \
        && ! -f "${KERNEL_PYTHON_ROOT}/sgl_kernel_npu/lib/libsgl_kernel_npu.so" ]]; then
        echo "KERNEL_CODE_ROOT has no built libsgl_kernel_npu.so: ${KERNEL_CODE_ROOT}" >&2
        echo "Build the kernel checkout first, or omit KERNEL_CODE_ROOT to use the container package." >&2
        exit 2
    fi
    export PYTHONPATH="${KERNEL_PYTHON_ROOT}:${PYTHONPATH}"
fi

PREFILL_IPS="${PREFILL_IPS:-}"
DECODE_IPS="${DECODE_IPS:-}"
IFS=',' read -r -a PREFILL_IP_ARRAY <<< "${PREFILL_IPS}"
IFS=',' read -r -a DECODE_IP_ARRAY <<< "${DECODE_IPS}"
GROUP_NNODES="${GROUP_NNODES:-4}"
PREFILL_NNODES="${PREFILL_NNODES:-${GROUP_NNODES}}"
DECODE_NNODES="${DECODE_NNODES:-${GROUP_NNODES}}"
if [[ ! "${PREFILL_NNODES}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${DECODE_NNODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREFILL_NNODES and DECODE_NNODES must be positive integers." >&2
    exit 2
fi
if (( ${#PREFILL_IP_ARRAY[@]} != PREFILL_NNODES )); then
    echo "PREFILL_IPS must contain exactly ${PREFILL_NNODES} comma-separated IPs." >&2
    exit 2
fi
if (( ${#DECODE_IP_ARRAY[@]} != DECODE_NNODES )); then
    echo "DECODE_IPS must contain exactly ${DECODE_NNODES} comma-separated IPs." >&2
    exit 2
fi

PREFILL_RANK0_IP="${PREFILL_IP_ARRAY[0]}"
DECODE_RANK0_IP="${DECODE_IP_ARRAY[0]}"

# P and D are independent distributed groups and may use different node counts
# and TP sizes. TP_SIZE remains as a backward-compatible default for wrappers
# that use a symmetric topology.
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-eight-node 4P4D}"
TP_SIZE_DEFAULT="${TP_SIZE:-64}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-${TP_SIZE_DEFAULT}}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-${TP_SIZE_DEFAULT}}"
PP_SIZE="${PP_SIZE:-1}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-2}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-4}"

if [[ ! "${PREFILL_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${DECODE_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${PREFILL_DP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${PREFILL_CP_SIZE}" =~ ^[1-9][0-9]*$ ]] \
    || [[ ! "${DECODE_DP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TP/DP/CP sizes must be positive integers." >&2
    exit 2
fi
if (( PREFILL_TP_SIZE % PREFILL_NNODES != 0 )); then
    echo "PREFILL_TP_SIZE=${PREFILL_TP_SIZE} must be divisible by PREFILL_NNODES=${PREFILL_NNODES}." >&2
    exit 2
fi
if (( DECODE_TP_SIZE % DECODE_NNODES != 0 )); then
    echo "DECODE_TP_SIZE=${DECODE_TP_SIZE} must be divisible by DECODE_NNODES=${DECODE_NNODES}." >&2
    exit 2
fi
if (( PREFILL_TP_SIZE % (PREFILL_DP_SIZE * PREFILL_CP_SIZE) != 0 )); then
    echo "PREFILL_TP_SIZE must be divisible by PREFILL_DP_SIZE * PREFILL_CP_SIZE." >&2
    exit 2
fi
if (( DECODE_TP_SIZE % DECODE_DP_SIZE != 0 )); then
    echo "DECODE_TP_SIZE must be divisible by DECODE_DP_SIZE." >&2
    exit 2
fi
if [[ "${PP_SIZE}" != "1" ]]; then
    echo "This PD profile keeps pipeline parallelism disabled; use PP_SIZE=1." >&2
    exit 2
fi

PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-15101}"
DECODE_DIST_PORT="${DECODE_DIST_PORT:-15102}"
PREFILL_PORT="${PREFILL_PORT:-31001}"
DECODE_PORT="${DECODE_PORT:-31002}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-18998}"
ROUTER_PORT="${ROUTER_PORT:-18077}"
MF_STORE_PORT="${MF_STORE_PORT:-34670}"
HOST="${HOST:-0.0.0.0}"

PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-4}"
DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-4}"
PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.82}"
DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.75}"
PREFILL_DEEPEP_MODE="${PREFILL_DEEPEP_MODE:-normal}"
DECODE_DEEPEP_MODE="${DECODE_DEEPEP_MODE:-auto}"
PREFILL_ENABLE_DEEPEP="${PREFILL_ENABLE_DEEPEP:-1}"
DECODE_ENABLE_DEEPEP="${DECODE_ENABLE_DEEPEP:-1}"
PREFILL_ENABLE_DP_ATTENTION="${PREFILL_ENABLE_DP_ATTENTION:-1}"
DECODE_ENABLE_DP_ATTENTION="${DECODE_ENABLE_DP_ATTENTION:-1}"
PREFILL_HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-2000}"
DECODE_HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-2000}"
PREFILL_DEEPEP_MAX_DISPATCH="${PREFILL_DEEPEP_MAX_DISPATCH:-128}"
DECODE_DEEPEP_MAX_DISPATCH="${DECODE_DEEPEP_MAX_DISPATCH:-128}"

KDA_CP_BACKEND="${KDA_CP_BACKEND:-fla}"
MLA_CP_BACKEND="${MLA_CP_BACKEND:-ring}"
case "${KDA_CP_BACKEND}" in a2a|fla) ;; *) echo "KDA_CP_BACKEND must be a2a or fla." >&2; exit 2 ;; esac
case "${MLA_CP_BACKEND}" in allgather|ring) ;; *) echo "MLA_CP_BACKEND must be allgather or ring." >&2; exit 2 ;; esac

# The full 4P4D profile enables dSparK and Decode NPU Graph by default.
# Wrappers for smaller validation topologies may override both defaults.
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-0}"
case "${ENABLE_DSPARK}" in 0|1) ;; *) echo "ENABLE_DSPARK must be 0 or 1." >&2; exit 2 ;; esac
case "${DISABLE_CUDA_GRAPH}" in 0|1) ;; *) echo "DISABLE_CUDA_GRAPH must be 0 or 1." >&2; exit 2 ;; esac
case "${ENABLE_PREFIX_CACHE}" in 0|1) ;; *) echo "ENABLE_PREFIX_CACHE must be 0 or 1." >&2; exit 2 ;; esac
case "${PREFILL_ENABLE_DEEPEP}" in 0|1) ;; *) echo "PREFILL_ENABLE_DEEPEP must be 0 or 1." >&2; exit 2 ;; esac
case "${DECODE_ENABLE_DEEPEP}" in 0|1) ;; *) echo "DECODE_ENABLE_DEEPEP must be 0 or 1." >&2; exit 2 ;; esac
case "${PREFILL_ENABLE_DP_ATTENTION}" in 0|1) ;; *) echo "PREFILL_ENABLE_DP_ATTENTION must be 0 or 1." >&2; exit 2 ;; esac
case "${DECODE_ENABLE_DP_ATTENTION}" in 0|1) ;; *) echo "DECODE_ENABLE_DP_ATTENTION must be 0 or 1." >&2; exit 2 ;; esac

DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
DSPARK_DRAFT_QUANTIZATION="${DSPARK_DRAFT_QUANTIZATION:-unquant}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_8node_pd_pcp}"
RUN_TAG="${RUN_TAG:-pd8_pcp_fla_ring}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${PREFILL_IPS},${DECODE_IPS}"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

if [[ "${ROLE}" == "router" ]]; then
    ROUTER_ARGS=(
        --pd-disaggregation
        --prefill "http://${PREFILL_RANK0_IP}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}"
        --decode "http://${DECODE_RANK0_IP}:${DECODE_PORT}"
        --host "${HOST}"
        --port "${ROUTER_PORT}"
        --mini-lb
    )
    echo "Kimi-K3 PD router: ${HOST}:${ROUTER_PORT}"
    echo "  prefill=http://${PREFILL_RANK0_IP}:${PREFILL_PORT}, bootstrap=${BOOTSTRAP_PORT}"
    echo "  decode=http://${DECODE_RANK0_IP}:${DECODE_PORT}"
    if [[ "${CONFIG_ONLY}" == "1" ]]; then
        printf 'command:'; printf ' %q' "${PYTHON_BIN}" -m sglang_router.launch_router "${ROUTER_ARGS[@]}"; printf '\n'
        exit 0
    fi
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${RUN_TAG}_router_$(date '+%Y-%m-%d_%H-%M-%S').log"
    "${PYTHON_BIN}" -m sglang_router.launch_router "${ROUTER_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    exit "${PIPESTATUS[0]}"
fi

if [[ "${ROLE}" == "prefill" ]]; then
    NNODES="${PREFILL_NNODES}"
    TP_SIZE="${PREFILL_TP_SIZE}"
else
    NNODES="${DECODE_NNODES}"
    TP_SIZE="${DECODE_TP_SIZE}"
fi

NODE_RANK="${NODE_RANK:-}"
LOCAL_IP="${LOCAL_IP:-}"
NET_IFACE="${NET_IFACE:-}"
MODEL_PATH="${MODEL_PATH:-}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]] \
    || (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "Set NODE_RANK to an integer in [0, $((NNODES - 1))] on every ${ROLE} node." >&2
    exit 2
fi
if [[ -z "${LOCAL_IP}" || -z "${NET_IFACE}" ]]; then
    echo "Set LOCAL_IP and NET_IFACE explicitly on every ${ROLE} node." >&2
    exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Set MODEL_PATH to this node's complete Kimi-K3 checkpoint directory." >&2
    exit 2
fi
if [[ "${ROLE}" == "decode" && "${ENABLE_DSPARK}" == "1" \
    && -z "${DRAFT_MODEL_PATH}" ]]; then
    echo "ENABLE_DSPARK=1 requires this node's DRAFT_MODEL_PATH." >&2
    exit 2
fi

if [[ "${ROLE}" == "prefill" ]]; then
    EXPECTED_LOCAL_IP="${PREFILL_IP_ARRAY[$NODE_RANK]}"
    DIST_INIT_ADDR="${PREFILL_RANK0_IP}:${PREFILL_DIST_PORT}"
    PORT="${PREFILL_PORT}"
    DP_SIZE="${PREFILL_DP_SIZE}"
    MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS}"
    MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC}"
    DEEPEP_MODE="${PREFILL_DEEPEP_MODE}"
    ENABLE_DEEPEP="${PREFILL_ENABLE_DEEPEP}"
    ENABLE_DP_ATTENTION="${PREFILL_ENABLE_DP_ATTENTION}"
    HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE}"
    DEEPEP_MAX_DISPATCH="${PREFILL_DEEPEP_MAX_DISPATCH}"
else
    EXPECTED_LOCAL_IP="${DECODE_IP_ARRAY[$NODE_RANK]}"
    DIST_INIT_ADDR="${DECODE_RANK0_IP}:${DECODE_DIST_PORT}"
    PORT="${DECODE_PORT}"
    DP_SIZE="${DECODE_DP_SIZE}"
    MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS}"
    MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC}"
    DEEPEP_MODE="${DECODE_DEEPEP_MODE}"
    ENABLE_DEEPEP="${DECODE_ENABLE_DEEPEP}"
    ENABLE_DP_ATTENTION="${DECODE_ENABLE_DP_ATTENTION}"
    HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE}"
    DEEPEP_MAX_DISPATCH="${DECODE_DEEPEP_MAX_DISPATCH}"
fi
if [[ "${LOCAL_IP}" != "${EXPECTED_LOCAL_IP}" ]]; then
    echo "LOCAL_IP/NODE_RANK mismatch for ${ROLE}:" >&2
    echo "  rank ${NODE_RANK} expects ${EXPECTED_LOCAL_IP}, got ${LOCAL_IP}." >&2
    exit 2
fi

if [[ "${CONFIG_ONLY}" != "1" ]]; then
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
        echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
        exit 2
    fi
    for tokenizer_file in tokenizer_config.json tokenization_kimi.py tiktoken.model; do
        if [[ ! -f "${MODEL_PATH}/${tokenizer_file}" ]]; then
            echo "MODEL_PATH is missing tokenizer file: ${MODEL_PATH}/${tokenizer_file}" >&2
            exit 2
        fi
    done
    if [[ "${ROLE}" == "decode" && "${ENABLE_DSPARK}" == "1" \
        && ! -f "${DRAFT_MODEL_PATH:-}/config.json" ]]; then
        echo "ENABLE_DSPARK=1 requires a local DRAFT_MODEL_PATH containing config.json." >&2
        exit 2
    fi
fi

export SGLANG_HOST_IP="${LOCAL_IP}"
export SGLANG_LOCAL_IP_NIC="${NET_IFACE}"
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"
export ASCEND_MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://${PREFILL_RANK0_IP}:${MF_STORE_PORT}}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-3600}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20000}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-20000-20199}"
export HCCL_BUFFSIZE
export DEEPEP_HCCL_BUFFSIZE="${DEEPEP_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE}}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${DEEPEP_MAX_DISPATCH}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE="${SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"
export ATB_CXX_ABI="${ATB_CXX_ABI:-1}"

CACHE_ARGS=()
if [[ "${ENABLE_PREFIX_CACHE}" == "0" ]]; then
    CACHE_ARGS=(--disable-radix-cache)
fi

MAMBA_CACHE_ARGS=()
if [[ -n "${MAX_MAMBA_CACHE_SIZE}" ]]; then
    if [[ ! "${MAX_MAMBA_CACHE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "MAX_MAMBA_CACHE_SIZE must be a positive integer." >&2
        exit 2
    fi
    MAMBA_CACHE_ARGS=(--max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}")
fi

DP_ARGS=(--dp-size "${DP_SIZE}")
if [[ "${ENABLE_DP_ATTENTION}" == "1" ]]; then
    DP_ARGS=(--enable-dp-attention --dp-size "${DP_SIZE}" --enable-dp-lm-head)
fi

MOE_ARGS=()
if [[ "${ENABLE_DEEPEP}" == "1" ]]; then
    MOE_ARGS=(--moe-a2a-backend deepep --deepep-mode "${DEEPEP_MODE}")
fi

COMMON_ARGS=(
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
    --pp-size "${PP_SIZE}"
    "${DP_ARGS[@]}"
    --base-gpu-id 0
    --page-size "${PAGE_SIZE}"
    --max-total-tokens "${MAX_TOTAL_TOKENS}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    "${MAMBA_CACHE_ARGS[@]}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    "${MOE_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    --watchdog-timeout 9000
    --host "${HOST}"
    --port "${PORT}"
)

if [[ "${ROLE}" == "prefill" ]]; then
    export SGLANG_ENABLE_CP_V2=1
    export SGLANG_KDA_CP_ASYNC_GATHER="${SGLANG_KDA_CP_ASYNC_GATHER:-0}"
    ROLE_ARGS=(
        --attn-cp-size "${PREFILL_CP_SIZE}"
        --enable-prefill-cp
        --cp-strategy zigzag
        --kda-cp-backend "${KDA_CP_BACKEND}"
        --mla-cp-backend "${MLA_CP_BACKEND}"
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --disable-cuda-graph
        --disaggregation-mode prefill
        --disaggregation-transfer-backend ascend
        --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
    )
else
    export SGLANG_ENABLE_CP_V2=0
    ROLE_ARGS=(
        --chunked-prefill-size -1
        --mem-fraction-static "${MEM_FRACTION_STATIC}"
        --disaggregation-mode decode
        --disaggregation-transfer-backend ascend
        --disaggregation-decode-extra-slots "${DECODE_EXTRA_SLOTS:-8}"
    )
    if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
        ROLE_ARGS+=(--disable-cuda-graph)
    else
        read -r -a CUDA_GRAPH_BS_ARRAY <<< "${CUDA_GRAPH_BS:-1 4}"
        ROLE_ARGS+=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_ARRAY[@]}")
    fi
    if [[ "${ENABLE_DSPARK}" == "1" ]]; then
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
        ROLE_ARGS+=(
            --speculative-algorithm DSPARK
            --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
            --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
            --speculative-draft-attention-backend "${DSPARK_DRAFT_ATTENTION_BACKEND}"
            --speculative-eagle-topk 1
            --speculative-draft-model-quantization "${DSPARK_DRAFT_QUANTIZATION}"
        )
    fi
fi

if [[ "${ROLE}" == "prefill" ]]; then
    ATTN_CP_FACTOR="${PREFILL_CP_SIZE}"
else
    ATTN_CP_FACTOR=1
fi
ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE / ATTN_CP_FACTOR))
echo "Kimi-K3 ${DEPLOYMENT_NAME} ${ROLE}:"
echo "  node-rank=${NODE_RANK}/${NNODES}, local-ip=${LOCAL_IP}, interface=${NET_IFACE}"
echo "  dist=${DIST_INIT_ADDR}, http=${HOST}:${PORT}, model=${MODEL_PATH}"
echo "  TP=${TP_SIZE}, PP=${PP_SIZE}, DP=${DP_SIZE}, attention-TP=${ATTN_TP_SIZE}"
if [[ "${ROLE}" == "prefill" ]]; then
    echo "  PCP=1, CP=${PREFILL_CP_SIZE}, KDA=${KDA_CP_BACKEND}, MLA=${MLA_CP_BACKEND}"
else
    echo "  PCP=0, CP=1, dSparK=${ENABLE_DSPARK}, decode-graph=$((1 - DISABLE_CUDA_GRAPH))"
    if [[ "${ENABLE_DSPARK}" == "1" ]]; then
        echo "  draft=${DRAFT_MODEL_PATH}"
    fi
fi
echo "  DP-attention=${ENABLE_DP_ATTENTION}, DeepEP=${ENABLE_DEEPEP}"
echo "  MF-store=${ASCEND_MF_STORE_URL}, HCCL-range=${HCCL_NPU_SOCKET_PORT_RANGE}"
echo "  max-tokens=${MAX_TOTAL_TOKENS}, max-running=${MAX_RUNNING_REQUESTS}, mem=${MEM_FRACTION_STATIC}"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf 'command:'; printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}"; printf '\n'
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
LOG_FILE="${LOG_DIR}/${RUN_TAG}_${ROLE}_rank${NODE_RANK}_$(date '+%Y-%m-%d_%H-%M-%S').log"
"${PYTHON_BIN}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
