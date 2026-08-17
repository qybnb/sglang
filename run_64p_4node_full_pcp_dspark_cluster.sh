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
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); " \
        'pid=$(cat "${pid_file}" 2>/dev/null || true); ' \
        'if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then echo "RUNNING pid=${pid}"; else echo STOPPED; fi'
}

cleanup_command() {
    local pid_file="${REMOTE_LOG_DIR}/server.pid"
    local pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); pgid_file=$(printf '%q' "${pgid_file}"); " \
        'pkill -9 -f "[r]un_64p_4node_full_pcp_dspark.sh" 2>/dev/null || true; ' \
        'pkill -9 -f "[s]glang.launch_server" 2>/dev/null || true; ' \
        'pkill -9 -f "[s]glang::scheduler" 2>/dev/null || true; ' \
        'rm -f "${pid_file}" "${pgid_file}"; echo STOPPED'
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
    "PAGE_SIZE=${PAGE_SIZE:-64}"
    "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}"
    "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-212992}"
    "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-4}"
    "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}"
    "HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2000}"
    "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
    "DEEPEP_MODE=${DEEPEP_MODE:-auto}"
    "DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}"
    "CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-1 4}"
    "ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-1}"
)
[[ -z "${DIST_INIT_ADDR:-}" ]] || COMMON_ENV+=("DIST_INIT_ADDR=${DIST_INIT_ADDR}")

echo "Launching four ranks directly: profile=${PROFILE}, nodes=${NODE_IPS}"
for rank in 1 2 3 0; do
    host="${NODES[$rank]}"
    log_file="${REMOTE_LOG_DIR}/${PROFILE}_rank${rank}_launcher.log"
    pid_file="${REMOTE_LOG_DIR}/server.pid"
    pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    model_var="MODEL_PATH_RANK${rank}"
    draft_var="DRAFT_MODEL_PATH_RANK${rank}"
    net_iface_var="NET_IFACE_RANK${rank}"
    rank_model_path="${MODEL_PATH}"
    rank_draft_model_path="${DRAFT_MODEL_PATH}"
    rank_net_iface="${NET_IFACE}"
    [[ -z "${!model_var:-}" ]] || rank_model_path="${!model_var}"
    [[ -z "${!draft_var:-}" ]] || rank_draft_model_path="${!draft_var}"
    [[ -z "${!net_iface_var:-}" ]] || rank_net_iface="${!net_iface_var}"
    env_args=(
        "${COMMON_ENV[@]}"
        "MODEL_PATH=${rank_model_path}"
        "DRAFT_MODEL_PATH=${rank_draft_model_path}"
        "NET_IFACE=${rank_net_iface}"
        "NODE_RANK=${rank}"
    )

    run_on "${host}" "$(cleanup_command)" >/dev/null

    printf -v launch '%q ' \
        nohup setsid env "${env_args[@]}" \
        ./run_64p_4node_full_pcp_dspark.sh "${PROFILE}"

    command="cd $(printf '%q' "${REMOTE_REPO_ROOT}") || exit 1; "
    command+="mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}"); "
    command+="rm -f $(printf '%q' "${pid_file}") $(printf '%q' "${pgid_file}"); "
    command+="${launch}>$(printf '%q' "${log_file}") 2>&1 < /dev/null & "
    command+='pid=$!; echo "${pid}" >'"$(printf '%q' "${pid_file}")"'; '
    command+='pgid=$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d " " || true); echo "${pgid}" >'"$(printf '%q' "${pgid_file}")"

    run_on "${host}" "${command}"
    echo "rank=${rank} host=${host} STARTED log=${log_file}"
done

echo "All four launch commands were sent."
echo "Rank-0 log: ${REMOTE_LOG_DIR}/${PROFILE}_rank0_launcher.log"
echo "Service URL after model loading: http://${NODES[0]}:${PORT:-15010}"
