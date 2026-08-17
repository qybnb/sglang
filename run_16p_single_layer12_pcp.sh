#!/usr/bin/env bash
#
# Single-node, 16-NPU Kimi-K3 reduced-layer PCP performance launcher.
#
# This is a unified (non-PD) service. PCP is active for prefill while decode
# keeps CP off. DSpark, decode graph, DeepEP auto mode, and Radix Cache are on
# by default so the runtime profile matches the optimized mixed deployment.
# The target checkpoint itself is not modified; only
# text_config.num_hidden_layers is overridden while loading it.
#
# MLA-only comparison (KDA stays on FLA in both runs):
#   MODEL_PATH=/path/to/Kimi-K3 ./run_16p_single_layer12_pcp.sh allgather
#   MODEL_PATH=/path/to/Kimi-K3 ./run_16p_single_layer12_pcp.sh ring
#
# Optional CP-off resource baseline:
#   MODEL_PATH=/path/to/Kimi-K3 ./run_16p_single_layer12_pcp.sh off
#
# Useful overrides:
#   NUM_HIDDEN_LAYERS=12       # reduced target-model layer count
#   CP_SIZE=4                  # PCP profiles only
#   CHUNKED_PREFILL_SIZE=4096  # 4096 / (2 * CP4) = 512-token ring block
#   DRAFT_MODEL_PATH=/path/to/reduced-layer/DSpark
#   DISABLE_CUDA_GRAPH=0 CUDA_GRAPH_BS="1 4"  # defaults
#   CONFIG_ONLY=1              # print the resolved command without launching
#
set -euo pipefail

PROFILE="${1:-}"
case "${PROFILE}" in
    off|allgather|ring) ;;
    *)
        echo "Usage: MODEL_PATH=/path/to/Kimi-K3 $0 {off|allgather|ring}" >&2
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
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/hanwlax/workspace/checkpoints/DSpark-Kimi-K3-layer6-smoke}"
if [[ "${CONFIG_ONLY}" != "1" ]]; then
    for checkpoint in "${MODEL_PATH}" "${DRAFT_MODEL_PATH}"; do
        if [[ ! -f "${checkpoint}/config.json" ]]; then
            echo "Invalid checkpoint; config.json not found: ${checkpoint}" >&2
            exit 2
        fi
    done
fi

NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-12}"
if [[ ! "${NUM_HIDDEN_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_HIDDEN_LAYERS must be a positive integer." >&2
    exit 2
fi

CHECKPOINT_LAYERS=unknown
if [[ -f "${MODEL_PATH}/config.json" ]]; then
    CHECKPOINT_LAYERS="$("${PYTHON_BIN}" -c \
        'import json,sys; c=json.load(open(sys.argv[1])); print(c.get("text_config", c).get("num_hidden_layers", "unknown"))' \
        "${MODEL_PATH}/config.json")"
    if [[ "${CHECKPOINT_LAYERS}" =~ ^[1-9][0-9]*$ ]] \
        && (( NUM_HIDDEN_LAYERS > CHECKPOINT_LAYERS )); then
        echo "NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS} exceeds checkpoint layers=${CHECKPOINT_LAYERS}." >&2
        exit 2
    fi
fi

# The production DSpark checkpoint taps target layers [7, 23, 51, 67, 83]
# and cannot initialize against a 12-layer target. Require a reduced-layer
# smoke config whose tap ids are all inside the selected target layer range.
# This validates configuration only; no model/runtime Python code is changed.
DSPARK_TARGET_LAYER_IDS=unknown
if [[ -f "${DRAFT_MODEL_PATH}/config.json" ]]; then
    if ! DSPARK_TARGET_LAYER_IDS="$("${PYTHON_BIN}" - "${DRAFT_MODEL_PATH}/config.json" "${NUM_HIDDEN_LAYERS}" <<'PY'
import json
import sys

path, target_layers = sys.argv[1], int(sys.argv[2])
config = json.load(open(path))
layer_ids = config.get("dspark_target_layer_ids")
if layer_ids is None:
    layer_ids = (config.get("dspark_config") or {}).get("target_layer_ids")
if layer_ids is None:
    layer_ids = (config.get("dflash_config") or {}).get("target_layer_ids")
if not layer_ids:
    raise SystemExit(f"DSpark config has no target_layer_ids: {path}")
invalid = [int(layer_id) for layer_id in layer_ids if not 0 <= int(layer_id) < target_layers]
if invalid:
    raise SystemExit(
        f"DSpark target_layer_ids={layer_ids} are incompatible with "
        f"NUM_HIDDEN_LAYERS={target_layers}; invalid={invalid}. "
        "Use the reduced-layer DSpark smoke checkpoint."
    )
print(",".join(str(int(layer_id)) for layer_id in layer_ids))
PY
    )"
    then
        exit 2
    fi
fi

TP_SIZE="${TP_SIZE:-16}"
if [[ "${TP_SIZE}" != "16" ]]; then
    echo "This launcher is fixed to one 16-NPU node; TP_SIZE must be 16." >&2
    exit 2
fi

CP_ARGS=()
case "${PROFILE}" in
    off)
        ENABLE_PCP=0
        ACTIVE_CP_SIZE=1
        # DP4 gives attention-TP4, matching the default PCP topology below.
        ACTIVE_DP_SIZE="${DP_SIZE:-4}"
        KDA_CP_BACKEND=disabled
        MLA_CP_BACKEND=disabled
        export SGLANG_ENABLE_CP_V2=0
        ;;
    allgather|ring)
        ENABLE_PCP=1
        ACTIVE_CP_SIZE="${CP_SIZE:-4}"
        ACTIVE_DP_SIZE="${DP_SIZE:-1}"
        # Keep KDA fixed so the two profiles isolate only MLA Ring vs AllGather.
        KDA_CP_BACKEND="${KDA_CP_BACKEND:-fla}"
        MLA_CP_BACKEND="${PROFILE}"
        export SGLANG_ENABLE_CP_V2=1
        ;;
esac

for value_name in ACTIVE_DP_SIZE ACTIVE_CP_SIZE; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer (got ${value})." >&2
        exit 2
    fi
done
if (( TP_SIZE % (ACTIVE_DP_SIZE * ACTIVE_CP_SIZE) != 0 )); then
    echo "TP_SIZE must be divisible by DP_SIZE * CP_SIZE" \
        "(TP=${TP_SIZE}, DP=${ACTIVE_DP_SIZE}, CP=${ACTIVE_CP_SIZE})." >&2
    exit 2
fi
ATTN_TP_SIZE=$((TP_SIZE / ACTIVE_DP_SIZE / ACTIVE_CP_SIZE))
if (( ENABLE_PCP == 1 && ATTN_TP_SIZE < 2 )); then
    echo "PCP requires attention-TP >= 2; got TP${TP_SIZE}/DP${ACTIVE_DP_SIZE}/CP${ACTIVE_CP_SIZE}." >&2
    exit 2
fi

case "${KDA_CP_BACKEND}" in
    disabled|a2a|fla) ;;
    *) echo "KDA_CP_BACKEND must be a2a or fla (got ${KDA_CP_BACKEND})." >&2; exit 2 ;;
esac

DIST_PORT="${DIST_PORT:-15120}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-127.0.0.1:${DIST_PORT}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-15020}"
BASE_GPU_ID="${BASE_GPU_ID:-0}"
PAGE_SIZE="${PAGE_SIZE:-128}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-65536}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 4}"
ENABLE_PREFIX_CACHE="${ENABLE_PREFIX_CACHE:-1}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
DSPARK_DRAFT_QUANTIZATION="${DSPARK_DRAFT_QUANTIZATION:-unquant}"

case "${DEEPEP_MODE}" in auto|normal|low_latency) ;; *) echo "Invalid DEEPEP_MODE=${DEEPEP_MODE}." >&2; exit 2 ;; esac
case "${DISABLE_CUDA_GRAPH}" in 0|1) ;; *) echo "DISABLE_CUDA_GRAPH must be 0 or 1." >&2; exit 2 ;; esac
case "${ENABLE_PREFIX_CACHE}" in 0|1) ;; *) echo "ENABLE_PREFIX_CACHE must be 0 or 1." >&2; exit 2 ;; esac
if (( CHUNKED_PREFILL_SIZE <= 0 || CHUNKED_PREFILL_SIZE % PAGE_SIZE != 0 )); then
    echo "CHUNKED_PREFILL_SIZE must be positive and divisible by PAGE_SIZE" \
        "(chunk=${CHUNKED_PREFILL_SIZE}, page=${PAGE_SIZE})." >&2
    exit 2
fi

if (( ENABLE_PCP == 1 )); then
    CP_ARGS=(
        --attn-cp-size "${ACTIVE_CP_SIZE}"
        --enable-prefill-cp
        --cp-strategy zigzag
        --kda-cp-backend "${KDA_CP_BACKEND}"
        --mla-cp-backend "${MLA_CP_BACKEND}"
    )
fi

if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    CUDA_GRAPH_ARGS=(--disable-cuda-graph)
else
    read -r -a CUDA_GRAPH_BS_ARRAY <<< "${CUDA_GRAPH_BS}"
    CUDA_GRAPH_ARGS=(--cuda-graph-bs-decode "${CUDA_GRAPH_BS_ARRAY[@]}")
fi

CACHE_ARGS=()
if [[ "${ENABLE_PREFIX_CACHE}" == "0" ]]; then
    CACHE_ARGS=(--disable-radix-cache)
fi

RUN_TAG="${RUN_TAG:-single_16p_${NUM_HIDDEN_LAYERS}l_${PROFILE}_dp${ACTIVE_DP_SIZE}_cp${ACTIVE_CP_SIZE}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_single_layer_reduced}"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_$(date '+%Y-%m-%d_%H-%M-%S').log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2000}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"

SERVER_ARGS=(
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --dist-init-addr "${DIST_INIT_ADDR}"
    --nnodes 1
    --node-rank 0
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${MODEL_PATH}"
    --json-model-override-args "{\"text_config\":{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS}}}"
    --trust-remote-code
    --attention-backend ascend
    --device npu
    --quantization modelslim
    --dtype bfloat16
    --base-gpu-id "${BASE_GPU_ID}"
    --tp-size "${TP_SIZE}"
    --enable-dp-attention
    --dp-size "${ACTIVE_DP_SIZE}"
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
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    --speculative-draft-attention-backend "${DSPARK_DRAFT_ATTENTION_BACKEND}"
    --speculative-eagle-topk 1
    --speculative-draft-model-quantization "${DSPARK_DRAFT_QUANTIZATION}"
    "${CACHE_ARGS[@]}"
    "${CUDA_GRAPH_ARGS[@]}"
    --watchdog-timeout 9000
    --host "${HOST}"
    --port "${PORT}"
    "${CP_ARGS[@]}"
)

echo "Kimi-K3 single-node reduced-layer PCP profile:"
echo "  profile=${PROFILE}, TP=${TP_SIZE}, DP=${ACTIVE_DP_SIZE}, CP=${ACTIVE_CP_SIZE}, attention-TP=${ATTN_TP_SIZE}"
echo "  PCP=${ENABLE_PCP}, KDA=${KDA_CP_BACKEND}, MLA=${MLA_CP_BACKEND}"
echo "  model=${MODEL_PATH}, checkpoint-layers=${CHECKPOINT_LAYERS}, loaded-layers=${NUM_HIDDEN_LAYERS}"
echo "  DSpark=${DRAFT_MODEL_PATH}, target-layer-ids=${DSPARK_TARGET_LAYER_IDS}, block-size=${DSPARK_BLOCK_SIZE}, spec-v2=${SGLANG_ENABLE_SPEC_V2}"
echo "  chunk=${CHUNKED_PREFILL_SIZE}, page=${PAGE_SIZE}, max-tokens=${MAX_TOTAL_TOKENS}, mem=${MEM_FRACTION_STATIC}"
echo "  prefix-cache=${ENABLE_PREFIX_CACHE}, cuda-graph-disabled=${DISABLE_CUDA_GRAPH}, cuda-graph-bs=${CUDA_GRAPH_BS}"
echo "  endpoint=http://127.0.0.1:${PORT}, log=${LOG_FILE}"

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
