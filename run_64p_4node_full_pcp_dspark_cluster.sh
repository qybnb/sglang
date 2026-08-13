#!/usr/bin/env bash
# Start, stop, or inspect the isolated four-node PCP + DSpark validation job.
set -euo pipefail

ACTION="${1:-}"
PROFILE="${2:-}"
case "${ACTION}" in
    start)
        case "${PROFILE}" in
            a2a|fla) ;;
            *)
                echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
                exit 2
                ;;
        esac
        ;;
    status|stop)
        if [[ -n "${PROFILE}" ]]; then
            echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
            exit 2
        fi
        ;;
    *)
        echo "Usage: $0 start {a2a|fla} | $0 {status|stop}" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${REPO_ROOT}}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
IFS=',' read -r -a NODE_IP_ARRAY <<< "${NODE_IPS}"
if (( ${#NODE_IP_ARRAY[@]} != 4 )); then
    echo "NODE_IPS must contain four comma-separated addresses." >&2
    exit 2
fi

CONTROL_HOST="${CONTROL_HOST:-${NODE_IP_ARRAY[0]}}"
LOCAL_ADDRESSES=" $(hostname -I 2>/dev/null || true) "
if [[ "${FORCE_SSH_ALL:-0}" != "1" && "${LOCAL_ADDRESSES}" != *" ${CONTROL_HOST} "* ]]; then
    echo "Run this controller on ${CONTROL_HOST}; local addresses:${LOCAL_ADDRESSES}" >&2
    exit 2
fi

SSH_USER="${SSH_USER:-root}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
SSH_BIN="${SSH_BIN:-ssh}"
FORCE_SSH_ALL="${FORCE_SSH_ALL:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
SSH_OPTS=(
    -o BatchMode=yes
    -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
    -o StrictHostKeyChecking=accept-new
)
if [[ -n "${SSH_IDENTITY_FILE:-}" ]]; then
    SSH_OPTS+=(
        -i "${SSH_IDENTITY_FILE}"
        -o IdentitiesOnly=yes
    )
fi

# This directory is deliberately different from the PCP-only controller.
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_REPO_ROOT}/logs/kimi_k3_4node_full_pcp_dspark/control}"

remote_shell() {
    local host="$1"
    shift
    local command="$*"
    if [[ -n "${CONTAINER_NAME}" ]]; then
        printf -v command '%q ' docker exec "${CONTAINER_NAME}" bash -c "${command}"
    fi
    if [[ "${host}" == "${CONTROL_HOST}" && "${FORCE_SSH_ALL}" != "1" ]]; then
        bash -c "${command}"
    else
        printf '%s\n' "${command}" | \
            "${SSH_BIN}" "${SSH_OPTS[@]}" "${SSH_USER}@${host}" bash -s
    fi
}

process_status_command() {
    local pid_file="${REMOTE_LOG_DIR}/server.pid"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); " \
        'if [[ ! -s "${pid_file}" ]]; then echo STOPPED; exit 1; fi; ' \
        'pid=$(cat "${pid_file}"); ' \
        'if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${pid}" 2>/dev/null; then echo STOPPED; exit 1; fi; ' \
        'cmdline=$(tr "\0" " " <"/proc/${pid}/cmdline" 2>/dev/null || true); ' \
        'if [[ "${cmdline}" != *run_64p_4node_full_pcp_dspark.sh* ]]; then echo "STALE_PID pid=${pid}"; exit 2; fi; ' \
        'echo "RUNNING pid=${pid}"'
}

process_stop_command() {
    local pid_file="${REMOTE_LOG_DIR}/server.pid"
    local pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); pgid_file=$(printf '%q' "${pgid_file}"); " \
        'if [[ ! -s "${pid_file}" ]]; then echo ALREADY_STOPPED; exit 0; fi; ' \
        'pid=$(cat "${pid_file}"); ' \
        'if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${pid}" 2>/dev/null; then rm -f "${pid_file}" "${pgid_file}"; echo ALREADY_STOPPED; exit 0; fi; ' \
        'cmdline=$(tr "\0" " " <"/proc/${pid}/cmdline" 2>/dev/null || true); ' \
        'if [[ "${cmdline}" != *run_64p_4node_full_pcp_dspark.sh* ]]; then echo "Refusing stale PID ${pid}: ${cmdline}" >&2; exit 2; fi; ' \
        'pgid=$(cat "${pgid_file}" 2>/dev/null || true); ' \
        'if [[ ! "${pgid}" =~ ^[1-9][0-9]*$ ]]; then echo "Invalid process group for PID ${pid}" >&2; exit 2; fi; ' \
        'kill -TERM -- "-${pgid}"; rm -f "${pid_file}" "${pgid_file}"; echo "STOPPED pid=${pid} pgid=${pgid}"'
}

for rank in "${!NODE_IP_ARRAY[@]}"; do
    host="${NODE_IP_ARRAY[$rank]}"
    case "${ACTION}" in
        status)
            printf 'rank=%s host=%s: ' "${rank}" "${host}"
            remote_shell "${host}" "$(process_status_command)" || true
            ;;
        stop)
            echo "Stopping DSpark rank=${rank} host=${host}"
            remote_shell "${host}" "$(process_stop_command)"
            ;;
    esac
done
if [[ "${ACTION}" != "start" ]]; then
    exit 0
fi

echo "Preflight: PCP+DSpark profile=${PROFILE}, target=${MODEL_PATH}, draft=${DRAFT_MODEL_PATH}"
if [[ -n "${CONTAINER_NAME}" ]]; then
    echo "Container mode: ${CONTAINER_NAME} on all four hosts"
fi
for rank in "${!NODE_IP_ARRAY[@]}"; do
    host="${NODE_IP_ARRAY[$rank]}"
    echo "  checking rank=${rank} host=${host}"
    remote_shell "${host}" \
        "test -x $(printf '%q' "${REMOTE_REPO_ROOT}/run_64p_4node_full_pcp_dspark.sh") && test -f $(printf '%q' "${MODEL_PATH}/config.json") && test -f $(printf '%q' "${DRAFT_MODEL_PATH}/config.json") && command -v setsid >/dev/null && mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}")"

    set +e
    current_status="$(remote_shell "${host}" "$(process_status_command)" 2>&1)"
    status_code=$?
    set -e
    case "${status_code}" in
        0)
            echo "Refusing to start: rank=${rank} host=${host} already ${current_status}." >&2
            exit 1
            ;;
        1) ;;
        *) echo "Refusing to start: rank=${rank} host=${host}: ${current_status}" >&2; exit 1 ;;
    esac

    # Never kill or overlap an existing inference service. The override is
    # intentionally explicit for controlled environments with partitioned NPUs.
    if [[ "${ALLOW_EXISTING_SGLANG_SERVICE:-0}" != "1" ]]; then
        set +e
        conflicts="$(remote_shell "${host}" \
            "ps -eo pid=,args= | grep -E '[s]glang\\.launch_server|[r]un_64p_4node_full_pcp(\\.sh|_cluster\\.sh)' | head -8" 2>&1)"
        conflict_status=$?
        set -e
        if [[ "${conflict_status}" == "0" && -n "${conflicts}" ]]; then
            echo "Refusing to overlap an existing SGLang service on ${host}:" >&2
            echo "${conflicts}" >&2
            exit 1
        fi
    fi
done

NET_IFACE="${NET_IFACE:-enp196s0f0}"
FORWARDED_ENV=(
    "MODEL_PATH=${MODEL_PATH}"
    "DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH}"
    "NODE_IPS=${NODE_IPS}"
    "NET_IFACE=${NET_IFACE}"
)
for var_name in \
    TP_SIZE DP_SIZE CP_SIZE DIST_PORT DIST_INIT_ADDR PORT PAGE_SIZE \
    CHUNKED_PREFILL_SIZE MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS \
    MEM_FRACTION_STATIC HCCL_BUFFSIZE DEEPEP_MODE DISABLE_CUDA_GRAPH CUDA_GRAPH_BS \
    DSPARK_BLOCK_SIZE DSPARK_DRAFT_ATTENTION_BACKEND DSPARK_DRAFT_QUANTIZATION \
    SGLANG_RAGGED_VERIFY_MODE SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK \
    DEEPEP_NORMAL_LONG_SEQ_ROUND DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS; do
    if [[ -v "${var_name}" ]]; then
        FORWARDED_ENV+=("${var_name}=${!var_name}")
    fi
done

for rank in 1 2 3 0; do
    host="${NODE_IP_ARRAY[$rank]}"
    remote_log="${REMOTE_LOG_DIR}/${PROFILE}_rank${rank}_launcher.log"
    pid_file="${REMOTE_LOG_DIR}/server.pid"
    pgid_file="${REMOTE_LOG_DIR}/server.pgid"
    env_args=("${FORWARDED_ENV[@]}" "NODE_RANK=${rank}")
    printf -v launch_command '%q ' \
        nohup setsid env "${env_args[@]}" \
        ./run_64p_4node_full_pcp_dspark.sh "${PROFILE}"
    command="cd $(printf '%q' "${REMOTE_REPO_ROOT}") || exit 1; "
    command+="rm -f $(printf '%q' "${pid_file}") $(printf '%q' "${pgid_file}"); "
    command+="${launch_command}>$(printf '%q' "${remote_log}") 2>&1 < /dev/null & "
    command+='pid=$!; '
    command+="echo \"\${pid}\" >$(printf '%q' "${pid_file}"); "
    command+='pgid=$(ps -o pgid= -p "${pid}" | tr -d " "); '
    command+="echo \"\${pgid}\" >$(printf '%q' "${pgid_file}")"
    echo "Starting DSpark rank=${rank} host=${host}; launcher-log=${remote_log}"
    remote_shell "${host}" "${command}"
done

STARTUP_CHECK_SECONDS="${STARTUP_CHECK_SECONDS:-3}"
sleep "${STARTUP_CHECK_SECONDS}"
startup_failed=0
for rank in "${!NODE_IP_ARRAY[@]}"; do
    host="${NODE_IP_ARRAY[$rank]}"
    printf 'rank=%s host=%s: ' "${rank}" "${host}"
    if ! remote_shell "${host}" "$(process_status_command)"; then
        echo "Inspect ${host}:${REMOTE_LOG_DIR}/${PROFILE}_rank${rank}_launcher.log" >&2
        startup_failed=1
    fi
done
if (( startup_failed != 0 )); then
    echo "One or more DSpark ranks failed during startup; stopping tracked ranks." >&2
    exec "$0" stop
fi

echo "All four PCP+DSpark launchers are running; model loading continues in background."
echo "Service URL after readiness: http://${NODE_IP_ARRAY[0]}:${PORT:-15010}"
echo "Status: $0 status"
echo "Stop:   $0 stop"
echo "Rank-0 log: ${REMOTE_LOG_DIR}/${PROFILE}_rank0_launcher.log"
