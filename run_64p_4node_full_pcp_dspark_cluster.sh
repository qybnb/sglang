#!/usr/bin/env bash
# Minimal four-node launcher for Kimi-K3 PCP + DSpark.
set -euo pipefail

ACTION="${1:-}"
PROFILE="${2:-}"
case "${ACTION}" in
    start)
        [[ "${PROFILE}" == "a2a" || "${PROFILE}" == "fla" ]] || {
            echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
            exit 2
        }
        ;;
    status|stop)
        [[ -z "${PROFILE}" ]] || {
            echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
            exit 2
        }
        ;;
    *)
        echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_REPO_ROOT}/logs/kimi_k3_4node_full_pcp_dspark/control}"
NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
IFS=',' read -r -a NODES <<< "${NODE_IPS}"
(( ${#NODES[@]} == 4 )) || {
    echo "NODE_IPS must contain exactly four comma-separated addresses." >&2
    exit 2
}

CONTAINER_NAME="${CONTAINER_NAME:-}"
SSH_USER="${SSH_USER:-root}"
SSH_BIN="${SSH_BIN:-ssh}"
LOCAL_IPS=" $(hostname -I 2>/dev/null || true) "

run_on() {
    local host="$1"
    shift
    local command="$*"

    if [[ -n "${CONTAINER_NAME}" ]]; then
        printf -v command '%q ' docker exec "${CONTAINER_NAME}" bash -c "${command}"
    fi

    if [[ "${LOCAL_IPS}" == *" ${host} "* ]]; then
        bash -c "${command}"
    else
        printf '%s\n' "${command}" | \
            "${SSH_BIN}" \
                -o BatchMode=yes \
                -o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-10}" \
                -o StrictHostKeyChecking=accept-new \
                "${SSH_USER}@${host}" bash -s
    fi
}

status_command() {
    local pid_file="${REMOTE_LOG_DIR}/server.pid"
    local exit_file="${REMOTE_LOG_DIR}/server.exit"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); exit_file=$(printf '%q' "${exit_file}"); " \
        'pid=$(cat "${pid_file}" 2>/dev/null || true); ' \
        'if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then echo "RUNNING pid=${pid}"; ' \
        'else rc=$(cat "${exit_file}" 2>/dev/null || true); if [[ "${rc}" =~ ^[0-9]+$ ]]; then echo "STOPPED rc=${rc}"; else echo "STOPPED rc=unknown (possible SIGKILL)"; fi; fi'
}

cleanup_command() {
    local pid_file="${REMOTE_LOG_DIR}/server.pid"
    local pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    local exit_file="${REMOTE_LOG_DIR}/server.exit"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); pgid_file=$(printf '%q' "${pgid_file}"); exit_file=$(printf '%q' "${exit_file}"); " \
        'pkill -9 -f "[r]un_64p_4node_full_pcp_dspark.sh" 2>/dev/null || true; ' \
        'pkill -9 -f "[s]glang.launch_server" 2>/dev/null || true; ' \
        'pkill -9 -f "[s]glang::scheduler" 2>/dev/null || true; ' \
        'rm -f "${pid_file}" "${pgid_file}" "${exit_file}"; echo STOPPED'
}

if [[ "${ACTION}" == "status" || "${ACTION}" == "stop" ]]; then
    for rank in 0 1 2 3; do
        host="${NODES[$rank]}"
        printf 'rank=%s host=%s: ' "${rank}" "${host}"
        if [[ "${ACTION}" == "status" ]]; then
            run_on "${host}" "$(status_command)"
        else
            run_on "${host}" "$(cleanup_command)"
        fi
    done
    exit 0
fi

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"

COMMON_ENV=(
    "NODE_IPS=${NODE_IPS}"
    "TP_SIZE=${TP_SIZE:-64}"
    "DP_SIZE=${DP_SIZE:-2}"
    "CP_SIZE=${CP_SIZE:-4}"
    "DIST_PORT=${DIST_PORT:-15110}"
    "PORT=${PORT:-15010}"
    "PAGE_SIZE=${PAGE_SIZE:-128}"
    "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}"
    "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-131072}"
    "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-2}"
    "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.82}"
    "HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2000}"
    "DEEPEP_HCCL_BUFFSIZE=${DEEPEP_HCCL_BUFFSIZE:-${HCCL_BUFFSIZE:-2000}}"
    "ATB_CXX_ABI=${ATB_CXX_ABI:-1}"
    "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-18000}"
    "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-18000-18100}"
    "DEEPEP_MODE=${DEEPEP_MODE:-auto}"
    "DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}"
    "CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-2}"
    "ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-1}"
    "SGLANG_KDA_CP_INTER_BLOCK_SIZE=${SGLANG_KDA_CP_INTER_BLOCK_SIZE:-32}"
    "SGLANG_KDA_CP_FUSED_MERGE=${SGLANG_KDA_CP_FUSED_MERGE:-1}"
    "SGLANG_KDA_CP_DIRECT_CONV_PLAN=${SGLANG_KDA_CP_DIRECT_CONV_PLAN:-1}"
    "SGLANG_KDA_CP_FUSED_FULL_CHUNK=${SGLANG_KDA_CP_FUSED_FULL_CHUNK:-1}"
    "SGLANG_KDA_CP_ASYNC_GATHER=${SGLANG_KDA_CP_ASYNC_GATHER:-0}"
    "SGLANG_NPU_MLA_CP_RING_BATCH_CAUSAL_TILES=${SGLANG_NPU_MLA_CP_RING_BATCH_CAUSAL_TILES:-1}"
    "SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_TILES=${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_TILES:-1}"
    "SGLANG_NPU_MLA_CP_RING_BATCH_VISIBLE_BLOCKS=${SGLANG_NPU_MLA_CP_RING_BATCH_VISIBLE_BLOCKS:-1}"
    "SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_MAX_TOKENS=${SGLANG_NPU_MLA_CP_RING_BATCH_PREFIX_MAX_TOKENS:-16384}"
)
[[ -z "${DIST_INIT_ADDR:-}" ]] || COMMON_ENV+=("DIST_INIT_ADDR=${DIST_INIT_ADDR}")
[[ -z "${MLA_CP_BACKEND_OVERRIDE:-}" ]] || \
    COMMON_ENV+=("MLA_CP_BACKEND_OVERRIDE=${MLA_CP_BACKEND_OVERRIDE}")

echo "Launching four ranks directly: profile=${PROFILE}, nodes=${NODE_IPS}"
for rank in 1 2 3 0; do
    host="${NODES[$rank]}"
    log_file="${REMOTE_LOG_DIR}/${PROFILE}_rank${rank}_launcher.log"
    pid_file="${REMOTE_LOG_DIR}/server.pid"
    pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    exit_file="${REMOTE_LOG_DIR}/server.exit"
    model_var="MODEL_PATH_RANK${rank}"
    draft_var="DRAFT_MODEL_PATH_RANK${rank}"
    net_iface_var="NET_IFACE_RANK${rank}"
    kernel_var="KERNEL_CODE_ROOT_RANK${rank}"
    rank_model_path="${MODEL_PATH}"
    rank_draft_model_path="${DRAFT_MODEL_PATH}"
    rank_net_iface="${NET_IFACE}"
    rank_kernel_code_root="${KERNEL_CODE_ROOT:-}"
    [[ -z "${!model_var:-}" ]] || rank_model_path="${!model_var}"
    [[ -z "${!draft_var:-}" ]] || rank_draft_model_path="${!draft_var}"
    [[ -z "${!net_iface_var:-}" ]] || rank_net_iface="${!net_iface_var}"
    [[ -z "${!kernel_var:-}" ]] || rank_kernel_code_root="${!kernel_var}"
    env_args=(
        "${COMMON_ENV[@]}"
        "MODEL_PATH=${rank_model_path}"
        "DRAFT_MODEL_PATH=${rank_draft_model_path}"
        "NET_IFACE=${rank_net_iface}"
        "NODE_RANK=${rank}"
    )
    [[ -z "${rank_kernel_code_root}" ]] || \
        env_args+=("KERNEL_CODE_ROOT=${rank_kernel_code_root}")

    run_on "${host}" "$(cleanup_command)" >/dev/null

    printf -v server_command '%q ' \
        env "${env_args[@]}" \
        ./run_64p_4node_full_pcp_dspark.sh "${PROFILE}"
    printf -v exit_file_q '%q' "${exit_file}"
    wrapper_command="printf '[KIMI_K3_CONTROLLER] ts=%s event=child-start pid=%s\\n' \"\$(date --iso-8601=seconds)\" \"\$\$\"; "
    wrapper_command+="${server_command}; rc=\$?; "
    wrapper_command+="printf '[KIMI_K3_CONTROLLER] ts=%s event=child-exit rc=%s\\n' \"\$(date --iso-8601=seconds)\" \"\${rc}\"; "
    wrapper_command+="printf '%s\\n' \"\${rc}\" >${exit_file_q}; exit \"\${rc}\""
    printf -v launch '%q ' nohup setsid bash -c "${wrapper_command}"

    command="cd $(printf '%q' "${REMOTE_REPO_ROOT}") || exit 1; "
    command+="mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}"); "
    command+="rm -f $(printf '%q' "${pid_file}") $(printf '%q' "${pgid_file}") $(printf '%q' "${exit_file}"); "
    command+="${launch}>$(printf '%q' "${log_file}") 2>&1 < /dev/null & "
    command+='pid=$!; echo "${pid}" >'"$(printf '%q' "${pid_file}")"'; '
    command+='pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d " " || true); echo "${pgid}" >'"$(printf '%q' "${pgid_file}")"

    run_on "${host}" "${command}"
    echo "rank=${rank} host=${host} STARTED log=${log_file}"
done

echo "All four launch commands were sent."
echo "Rank-0 log: ${REMOTE_LOG_DIR}/${PROFILE}_rank0_launcher.log"
echo "Service URL after model loading: http://${NODES[0]}:${PORT:-15010}"
