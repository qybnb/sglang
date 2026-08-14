#!/usr/bin/env bash
#
# Full-model Kimi-K3 2P2D deployment on four 16-NPU nodes.
#
# Topology:
#   192.168.25.209 / 192.168.25.212: Prefill, TP32 / PP1 / DP1 / CP4
#   192.168.25.216 / 192.168.25.217: Decode,  TP32 / PP1 / DP1 / CP1
#
# Run inside the sglang-zkk-k3 container on every node:
#   # 209
#   bash run_64p_4node_pd_pcp.sh router
#   NODE_RANK=0 bash run_64p_4node_pd_pcp.sh prefill
#   # 212
#   NODE_RANK=1 bash run_64p_4node_pd_pcp.sh prefill
#   # 216
#   NODE_RANK=0 bash run_64p_4node_pd_pcp.sh decode
#   # 217
#   NODE_RANK=1 bash run_64p_4node_pd_pcp.sh decode
#
# Keep PP disabled so this profile validates PD+PCP without introducing a Kimi
# pipeline-parallel weight-loading path. DSpark remains disabled by default so
# the first validation changes only the P/D topology.

set -euo pipefail

ROLE="${1:-}"
case "${ROLE}" in
    router|prefill|decode) ;;
    *)
        echo "Usage: $0 {router|prefill|decode}" >&2
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
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

PREFILL_IPS_CSV="${PREFILL_IPS:-192.168.25.209,192.168.25.212}"
DECODE_IPS_CSV="${DECODE_IPS:-192.168.25.216,192.168.25.217}"
IFS=',' read -r -a PREFILL_IP_ARRAY <<< "${PREFILL_IPS_CSV}"
IFS=',' read -r -a DECODE_IP_ARRAY <<< "${DECODE_IPS_CSV}"
if (( ${#PREFILL_IP_ARRAY[@]} != 2 || ${#DECODE_IP_ARRAY[@]} != 2 )); then
    echo "PREFILL_IPS and DECODE_IPS must each contain two comma-separated IPs." >&2
    exit 2
fi

PREFILL_RANK0_IP="${PREFILL_IP_ARRAY[0]}"
DECODE_RANK0_IP="${DECODE_IP_ARRAY[0]}"
PREFILL_PORT="${PREFILL_PORT:-31001}"
DECODE_PORT="${DECODE_PORT:-31002}"
ROUTER_PORT="${ROUTER_PORT:-18077}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-18998}"
PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-15001}"
DECODE_DIST_PORT="${DECODE_DIST_PORT:-15002}"
MF_STORE_PORT="${MF_STORE_PORT:-34670}"

TP_SIZE="${TP_SIZE:-32}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-4}"
ENABLE_PCP="${ENABLE_PCP:-1}"
if [[ "${TP_SIZE}" != "32" || "${PP_SIZE}" != "1" || "${DP_SIZE}" != "1" ]]; then
    echo "This no-PP profile requires TP_SIZE=32, PP_SIZE=1, DP_SIZE=1." >&2
    exit 2
fi
case "${ENABLE_PCP}" in
    0) PREFILL_CP_SIZE=1 ;;
    1)
        if [[ ! "${PREFILL_CP_SIZE}" =~ ^[2-9][0-9]*$ ]] \
            || (( TP_SIZE % (DP_SIZE * PREFILL_CP_SIZE) != 0 )); then
            echo "PREFILL_CP_SIZE must be greater than 1 and divide TP_SIZE / DP_SIZE." >&2
            exit 2
        fi
        ;;
    *) echo "ENABLE_PCP must be 0 or 1 (got ${ENABLE_PCP})." >&2; exit 2 ;;
esac

KDA_CP_BACKEND="${KDA_CP_BACKEND:-fla}"
MLA_CP_BACKEND="${MLA_CP_BACKEND:-ring}"
case "${KDA_CP_BACKEND}" in a2a|fla) ;; *) echo "KDA_CP_BACKEND must be a2a or fla." >&2; exit 2 ;; esac
case "${MLA_CP_BACKEND}" in allgather|ring) ;; *) echo "MLA_CP_BACKEND must be allgather or ring." >&2; exit 2 ;; esac

PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.92}"
DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.90}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
PAGE_SIZE="${PAGE_SIZE:-128}"
# TP32/PP1 leaves little headroom while materializing all 93 layers. Keep the
# DeepEP windows above their configured max-BS requirements without retaining
# the larger PP2-era buffers (prefill maxBs=64 needs about 462 MiB; decode
# maxBs=16 needs substantially less).
PREFILL_HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-550}"
DECODE_HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-400}"
PREFILL_DEEPEP_MAX_DISPATCH="${PREFILL_DEEPEP_MAX_DISPATCH:-64}"
DECODE_DEEPEP_MAX_DISPATCH="${DECODE_DEEPEP_MAX_DISPATCH:-16}"
PREFILL_DEEPEP_MODE="${PREFILL_DEEPEP_MODE:-normal}"
DECODE_DEEPEP_MODE="${DECODE_DEEPEP_MODE:-auto}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_pd_pcp}"
RUN_TAG="${RUN_TAG:-full_pd_pcp${ENABLE_PCP}_${KDA_CP_BACKEND}_${MLA_CP_BACKEND}}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash." >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${PREFILL_IPS_CSV},${DECODE_IPS_CSV}"
export no_proxy="${NO_PROXY}"
unset ASCEND_LAUNCH_BLOCKING

detect_node_rank() {
    local expected_csv="$1"
    local -a expected
    local addresses=" $(hostname -I 2>/dev/null || true) "
    IFS=',' read -r -a expected <<< "${expected_csv}"
    for i in "${!expected[@]}"; do
        if [[ "${addresses}" == *" ${expected[$i]} "* ]]; then
            echo "${i}"
            return 0
        fi
    done
    return 1
}

if [[ "${ROLE}" != "router" ]]; then
    if [[ -z "${NODE_RANK:-}" ]]; then
        if [[ "${ROLE}" == "prefill" ]]; then
            NODE_RANK="$(detect_node_rank "${PREFILL_IPS_CSV}" || true)"
        else
            NODE_RANK="$(detect_node_rank "${DECODE_IPS_CSV}" || true)"
        fi
    fi
    if [[ ! "${NODE_RANK:-}" =~ ^[01]$ ]]; then
        echo "Cannot infer NODE_RANK for ${ROLE}; set NODE_RANK=0 or 1." >&2
        exit 2
    fi
fi

if [[ "${ROLE}" != "router" && "${CONFIG_ONLY}" != "1" \
    && ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Invalid MODEL_PATH; config.json not found: ${MODEL_PATH}" >&2
    exit 2
fi

if [[ "${CONFIG_ONLY}" != "1" ]]; then
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
fi

export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"
export SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE="${SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE:-1}"
export SGLANG_NPU_USE_MULTI_STREAM="${SGLANG_NPU_USE_MULTI_STREAM:-0}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-bfloat16}"
export ASCEND_MF_STORE_URL="${ASCEND_MF_STORE_URL:-tcp://${PREFILL_RANK0_IP}:${MF_STORE_PORT}}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-3600}"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-3600}"
unset SGLANG_PP_LAYER_PARTITION

find_iface_by_ip() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import fcntl
import os
import socket
import struct
import sys

target = sys.argv[1]
for name in sorted(os.listdir("/sys/class/net")):
    if name == "lo":
        continue
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", name[:15].encode())
        address = socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])
    except OSError:
        continue
    finally:
        sock.close()
    if address == target:
        print(name)
        break
PY
}

COMMON_ARGS=(
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --nnodes 2
    --model-path "${MODEL_PATH}"
    --tokenizer-path "${MODEL_PATH}"
    --trust-remote-code
    --attention-backend ascend
    --device npu
    --quantization modelslim
    --dtype bfloat16
    --tp-size "${TP_SIZE}"
    --pp-size "${PP_SIZE}"
    --dp-size "${DP_SIZE}"
    --base-gpu-id 0
    --disable-radix-cache
    --page-size "${PAGE_SIZE}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --mamba-ssm-dtype bfloat16
    --reasoning-parser kimi_k3
    --moe-a2a-backend deepep
    --disable-cuda-graph
    --watchdog-timeout 9000
    --host 0.0.0.0
)

case "${ROLE}" in
    router)
        ROUTER_ARGS=(
            --pd-disaggregation
            --prefill "http://${PREFILL_RANK0_IP}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}"
            --decode "http://${DECODE_RANK0_IP}:${DECODE_PORT}"
            --host 0.0.0.0
            --port "${ROUTER_PORT}"
            --mini-lb
        )
        echo "Kimi-K3 PD router: :${ROUTER_PORT} -> prefill=${PREFILL_RANK0_IP}:${PREFILL_PORT}, decode=${DECODE_RANK0_IP}:${DECODE_PORT}"
        if [[ "${CONFIG_ONLY}" == "1" ]]; then
            printf 'command:'; printf ' %q' "${PYTHON_BIN}" -m sglang_router.launch_router "${ROUTER_ARGS[@]}"; printf '\n'
            exit 0
        fi
        LOG_FILE="${LOG_DIR}/${RUN_TAG}_router_$(date '+%Y-%m-%d_%H-%M-%S').log"
        "${PYTHON_BIN}" -m sglang_router.launch_router "${ROUTER_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
        exit "${PIPESTATUS[0]}"
        ;;
    prefill)
        LOCAL_IP="${PREFILL_IP_ARRAY[$NODE_RANK]}"
        export HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE}"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${PREFILL_DEEPEP_MAX_DISPATCH}"
        if [[ "${ENABLE_PCP}" == "1" ]]; then
            export SGLANG_ENABLE_CP_V2=1
            CP_ARGS=(
                --attn-cp-size "${PREFILL_CP_SIZE}"
                --enable-prefill-cp
                --cp-strategy zigzag
                --kda-cp-backend "${KDA_CP_BACKEND}"
                --mla-cp-backend "${MLA_CP_BACKEND}"
            )
        else
            export SGLANG_ENABLE_CP_V2=0
            CP_ARGS=()
        fi
        ROLE_ARGS=(
            --dist-init-addr "${PREFILL_RANK0_IP}:${PREFILL_DIST_PORT}"
            --node-rank "${NODE_RANK}"
            --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC}"
            --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
            --enable-dp-attention
            --enable-dp-lm-head
            "${CP_ARGS[@]}"
            --deepep-mode "${PREFILL_DEEPEP_MODE}"
            --disaggregation-mode prefill
            --disaggregation-transfer-backend ascend
            --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
            --port "${PREFILL_PORT}"
        )
        ;;
    decode)
        LOCAL_IP="${DECODE_IP_ARRAY[$NODE_RANK]}"
        export HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE}"
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${DECODE_DEEPEP_MAX_DISPATCH}"
        export SGLANG_ENABLE_CP_V2=0
        ROLE_ARGS=(
            --dist-init-addr "${DECODE_RANK0_IP}:${DECODE_DIST_PORT}"
            --node-rank "${NODE_RANK}"
            --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC}"
            --chunked-prefill-size -1
            --deepep-mode "${DECODE_DEEPEP_MODE}"
            --disaggregation-mode decode
            --disaggregation-transfer-backend ascend
            --disaggregation-decode-extra-slots 8
            --port "${DECODE_PORT}"
        )
        ;;
esac

NET_IFACE="${NET_IFACE:-$(find_iface_by_ip "${LOCAL_IP}")}"
if [[ -z "${NET_IFACE}" ]]; then
    echo "Cannot find a network interface for ${LOCAL_IP}; set NET_IFACE explicitly." >&2
    exit 2
fi
export HCCL_SOCKET_IFNAME="${NET_IFACE}"
export GLOO_SOCKET_IFNAME="${NET_IFACE}"

ATTN_TP_SIZE=$((TP_SIZE / DP_SIZE / PREFILL_CP_SIZE))
echo "Kimi-K3 PD ${ROLE}: rank=${NODE_RANK}/2 ip=${LOCAL_IP} iface=${NET_IFACE}"
echo "  model=${MODEL_PATH}, TP=${TP_SIZE}, PP=${PP_SIZE}, DP=${DP_SIZE}"
if [[ "${ROLE}" == "prefill" ]]; then
    echo "  PCP=${ENABLE_PCP}, CP=${PREFILL_CP_SIZE}, attention-TP=${ATTN_TP_SIZE}, KDA=${KDA_CP_BACKEND}, MLA=${MLA_CP_BACKEND}"
else
    echo "  PCP=0, CP=1, attention-TP=${TP_SIZE}"
fi
echo "  HCCL=${HCCL_BUFFSIZE}MB, DeepEP-max-dispatch=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}, MF-store=${ASCEND_MF_STORE_URL}"

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    printf 'command:'; printf ' %q' "${PYTHON_BIN}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}"; printf '\n'
    exit 0
fi

LOG_FILE="${LOG_DIR}/${RUN_TAG}_${ROLE}_rank${NODE_RANK}_$(date '+%Y-%m-%d_%H-%M-%S').log"
"${PYTHON_BIN}" -m sglang.launch_server "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
