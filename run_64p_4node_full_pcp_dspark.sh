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

# Prefer a portable sibling checkout on new hosts while retaining the legacy
# shared checkout used by the established cluster. Callers can always set
# KERNEL_CODE_ROOT explicitly.
if [[ -z "${KERNEL_CODE_ROOT:-}" ]]; then
    sibling_kernel_root="$(dirname "${REPO_ROOT}")/sgl-kernel-npu"
    legacy_kernel_root="/home/hanwlax/test-codes/sgl-kernel-npu"
    if [[ -f "${sibling_kernel_root}/python/sgl_kernel_npu/sgl_kernel_npu/__init__.py" ]]; then
        KERNEL_CODE_ROOT="${sibling_kernel_root}"
    else
        KERNEL_CODE_ROOT="${legacy_kernel_root}"
    fi
fi
KERNEL_PYTHON_ROOT="${KERNEL_CODE_ROOT}/python/sgl_kernel_npu"
if [[ ! -f "${KERNEL_PYTHON_ROOT}/sgl_kernel_npu/__init__.py" ]]; then
    echo "Invalid KERNEL_CODE_ROOT; sgl_kernel_npu package not found under: ${KERNEL_CODE_ROOT}" >&2
    echo "Clone sgl-kernel-npu beside this repository or set KERNEL_CODE_ROOT explicitly." >&2
    exit 2
fi
KERNEL_SHARED_LIBRARY="${KERNEL_PYTHON_ROOT}/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
if [[ ! -f "${KERNEL_SHARED_LIBRARY}" ]]; then
    # Git intentionally excludes generated .so files. Reuse the ABI-matched
    # library shipped in the active container while loading the updated Python
    # kernels from KERNEL_CODE_ROOT. No rebuild is needed when this exists.
    INSTALLED_KERNEL_SHARED_LIBRARY="$("${PYTHON_BIN}" - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    candidate = Path(root) / "sgl_kernel_npu/lib/libsgl_kernel_npu.so"
    if candidate.is_file():
        print(candidate)
        break
PY
)"
    if [[ -z "${INSTALLED_KERNEL_SHARED_LIBRARY}" ]]; then
        echo "libsgl_kernel_npu.so is absent from both the source checkout and active Python installation." >&2
        echo "Build it once with: cd ${KERNEL_CODE_ROOT} && bash build.sh -a kernels" >&2
        exit 2
    fi
    mkdir -p "$(dirname "${KERNEL_SHARED_LIBRARY}")"
    ln -sfn "${INSTALLED_KERNEL_SHARED_LIBRARY}" "${KERNEL_SHARED_LIBRARY}"
fi
export PYTHONPATH="${REPO_ROOT}/python:${KERNEL_PYTHON_ROOT}:${PYTHONPATH:-}"
KERNEL_GIT_REVISION="$(git -C "${KERNEL_CODE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"

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
DP_SIZE="${DP_SIZE:-2}"
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
if [[ -n "${MLA_CP_BACKEND_OVERRIDE:-}" ]]; then
    MLA_CP_BACKEND="${MLA_CP_BACKEND_OVERRIDE}"
fi
export SGLANG_KDA_CP_INTER_BLOCK_SIZE="${SGLANG_KDA_CP_INTER_BLOCK_SIZE:-32}"
export SGLANG_KDA_CP_FUSED_MERGE="${SGLANG_KDA_CP_FUSED_MERGE:-1}"
export SGLANG_KDA_CP_DIRECT_CONV_PLAN="${SGLANG_KDA_CP_DIRECT_CONV_PLAN:-1}"
export SGLANG_KDA_CP_FUSED_FULL_CHUNK="${SGLANG_KDA_CP_FUSED_FULL_CHUNK:-1}"
# Async HCCL gather is experimental: it can serialize the following DeepEP
# collective on multi-node NPU deployments, so keep it opt-in.
export SGLANG_KDA_CP_ASYNC_GATHER="${SGLANG_KDA_CP_ASYNC_GATHER:-0}"
export SGLANG_NPU_MLA_CP_RING_BATCH_CAUSAL_TILES="${SGLANG_NPU_MLA_CP_RING_BATCH_CAUSAL_TILES:-1}"
export SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_TILES="${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_TILES:-1}"
export SGLANG_NPU_MLA_CP_RING_BATCH_VISIBLE_BLOCKS="${SGLANG_NPU_MLA_CP_RING_BATCH_VISIBLE_BLOCKS:-1}"
export SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_MAX_TOKENS="${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_MAX_TOKENS:-16384}"
case "${MLA_CP_BACKEND}" in
    allgather|ring) ;;
    *)
        echo "MLA_CP_BACKEND_OVERRIDE must be allgather or ring (got ${MLA_CP_BACKEND})." >&2
        exit 2
        ;;
esac

# Dedicated defaults keep this service separate from the PCP-only service.
DIST_PORT="${DIST_PORT:-15110}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${NODE_IP_ARRAY[0]}:${DIST_PORT}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-15010}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
# Avoid importing torch from ATB's set_env.sh during every distributed launch.
# The current Kimi-K3 image uses the CXX11 ABI; callers using another image can
# override this with ATB_CXX_ABI=0.
ATB_CXX_ABI="${ATB_CXX_ABI:-1}"
# The Kimi-K3 EP64 low-latency combine kernel requires 1709 MB when
# SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128.  Keep a small,
# log-proven safety margin and fail before loading weights if a caller
# overrides the window with an undersized value.
KIMI_K3_DEEPEP_GRAPH_MIN_HCCL_BUFFSIZE="${KIMI_K3_DEEPEP_GRAPH_MIN_HCCL_BUFFSIZE:-1800}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
DSPARK_DRAFT_QUANTIZATION="${DSPARK_DRAFT_QUANTIZATION:-unquant}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-1}"
case "${DEEPEP_MODE}" in auto|normal|low_latency) ;; *) exit 2 ;; esac
case "${DISABLE_CUDA_GRAPH}" in 0|1) ;; *) exit 2 ;; esac
case "${ENABLE_PREFIX_CACHE}" in 0|1) ;; *) exit 2 ;; esac
case "${ATB_CXX_ABI}" in
    0|1) ;;
    *) echo "ATB_CXX_ABI must be 0 or 1 (got ${ATB_CXX_ABI})." >&2; exit 2 ;;
esac
if [[ "${DISABLE_CUDA_GRAPH}" == "0" && "${DEEPEP_MODE}" == "normal" ]]; then
    echo "Decode graph capture requires DEEPEP_MODE=auto or low_latency; got normal." >&2
    exit 2
fi
if [[ ! "${HCCL_BUFFSIZE}" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "${KIMI_K3_DEEPEP_GRAPH_MIN_HCCL_BUFFSIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "HCCL buffer sizes must be positive integers in MB." >&2
    exit 2
fi
if [[ "${DISABLE_CUDA_GRAPH}" == "0" && "${DEEPEP_MODE}" != "normal" ]] && \
   (( HCCL_BUFFSIZE < KIMI_K3_DEEPEP_GRAPH_MIN_HCCL_BUFFSIZE )); then
    echo "HCCL_BUFFSIZE=${HCCL_BUFFSIZE}MB is too small for Kimi-K3 EP64 DeepEP low-latency graph capture." >&2
    echo "The combine operator reports a 1709MB minimum at maxBs=128; use at least ${KIMI_K3_DEEPEP_GRAPH_MIN_HCCL_BUFFSIZE}MB (default: 2000MB)." >&2
    exit 2
fi

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--disable-cuda-graph)
else
    read -r -a CUDA_GRAPH_BS_ARRAY <<< "${CUDA_GRAPH_BS:-2}"
    CUDA_GRAPH_ARGS=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_ARRAY[@]}")
fi

CACHE_ARGS=()
if [[ "${ENABLE_PREFIX_CACHE}" == "0" ]]; then
    CACHE_ARGS=(--disable-radix-cache)
fi

RUN_TAG="${RUN_TAG:-full_4node_dspark_${PROFILE}_cp${CP_SIZE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_full_pcp_dspark}"
LOG_STAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_rank${NODE_RANK}_${LOG_STAMP}.log"
STARTUP_LOG_FILE="${LOG_DIR}/${RUN_TAG}_rank${NODE_RANK}_${LOG_STAMP}.startup.log"

# Keep startup diagnostics independent from the controller's launcher log.  In
# particular, this records failures before launch_server and leaves the last
# completed phase behind even when the process is killed by SIGKILL.
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${STARTUP_LOG_FILE}") 2>&1
STARTUP_PHASE="configuration-complete"
STARTUP_STARTED_AT="$(date --iso-8601=seconds)"

startup_log() {
    printf '[KIMI_K3_STARTUP] ts=%s pid=%s rank=%s phase=%s %s\n' \
        "$(date --iso-8601=seconds)" "$$" "${NODE_RANK}" \
        "${STARTUP_PHASE}" "$*"
}

startup_error() {
    local rc=$?
    local line="${BASH_LINENO[0]:-unknown}"
    local command="${BASH_COMMAND:-unknown}"
    trap - ERR
    startup_log "event=error rc=${rc} line=${line} command=$(printf '%q' "${command}")"
    exit "${rc}"
}

startup_exit() {
    local rc=$?
    startup_log "event=exit rc=${rc} started_at=${STARTUP_STARTED_AT}"
}

startup_signal() {
    local signal="$1"
    startup_log "event=signal signal=${signal}"
    exit 128
}

trap startup_error ERR
trap startup_exit EXIT
trap 'startup_signal HUP' HUP
trap 'startup_signal INT' INT
trap 'startup_signal TERM' TERM
startup_log "event=start startup_log=${STARTUP_LOG_FILE} runtime_log=${LOG_FILE}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${NODE_IPS}"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

export SGLANG_ENABLE_CP_V2=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export SGLANG_LOG_DECODE_GRAPH_KEY="${SGLANG_LOG_DECODE_GRAPH_KEY:-0}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"
# Avoid the auto allocator wrapping to invalid port 65536 after stale or
# parallel launches. Keep a deterministic, valid host-side HCCL port range.
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-18000}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-18000-18100}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export HCCL_BUFFSIZE
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
    "${CACHE_ARGS[@]}"
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
echo "  KDA-inter-BC=${SGLANG_KDA_CP_INTER_BLOCK_SIZE}," \
     "fused-merge=${SGLANG_KDA_CP_FUSED_MERGE}," \
     "direct-conv=${SGLANG_KDA_CP_DIRECT_CONV_PLAN}," \
     "fused-chunk=${SGLANG_KDA_CP_FUSED_FULL_CHUNK}," \
     "async-gather=${SGLANG_KDA_CP_ASYNC_GATHER}"
echo "  MLA-ring-batched-causal=${SGLANG_NPU_MLA_CP_RING_BATCH_CAUSAL_TILES}," \
     "batched-prefix=${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_TILES}," \
     "batched-visible=${SGLANG_NPU_MLA_CP_RING_BATCH_VISIBLE_BLOCKS}," \
     "prefix-pack-cap=${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_MAX_TOKENS}"
echo "  dist=${DIST_INIT_ADDR}, port=${PORT}, interface=${NET_IFACE}"
echo "  chunk=${CHUNKED_PREFILL_SIZE}, max-tokens=${MAX_TOTAL_TOKENS}, mem=${MEM_FRACTION_STATIC}"
echo "  HCCL_BUFFSIZE=${HCCL_BUFFSIZE}, DeepEP=${DEEPEP_MODE}, decode-graph=$((1 - DISABLE_CUDA_GRAPH))"
echo "  ATB_CXX_ABI=${ATB_CXX_ABI}"
echo "  kernel-root=${KERNEL_CODE_ROOT}, kernel-revision=${KERNEL_GIT_REVISION}"
echo "  kernel-so=$(readlink -f "${KERNEL_SHARED_LIBRARY}")"
echo "  prefix-cache=${ENABLE_PREFIX_CACHE}"
echo "  HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE}"
echo "  HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT}"
echo "  graph-replay-log=${SGLANG_LOG_DECODE_GRAPH_KEY}, log=${LOG_FILE}"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf '  command:'
    printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}"
    printf '\n'
    exit 0
fi

STARTUP_PHASE="source-ascend-environment"
for env_script in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [[ -f "${env_script}" ]]; then
        startup_log "event=source-begin file=${env_script}"
        set +u
        # shellcheck disable=SC1090
        if [[ "${env_script}" == */nnal/atb/set_env.sh ]]; then
            source "${env_script}" "--cxx_abi=${ATB_CXX_ABI}"
        else
            source "${env_script}"
        fi
        set -u
        startup_log "event=source-end file=${env_script} rc=0"
    else
        startup_log "event=source-skip file=${env_script} reason=not-found"
    fi
done

STARTUP_PHASE="launch-server"
startup_log "event=exec-begin python=${PYTHON_BIN} runtime_log=${LOG_FILE}"
# launch_server owns the long-running process.  Capture its status explicitly
# so the ERR trap does not report the final `tee` process as the failed command.
trap - ERR
set +e
"${PYTHON_BIN}" -m sglang.launch_server "${SERVER_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
server_rc="${PIPESTATUS[0]}"
set -e
trap startup_error ERR
startup_log "event=exec-end rc=${server_rc}"
exit "${server_rc}"
