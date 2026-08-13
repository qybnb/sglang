#!/usr/bin/env bash
# Portable one-command entry point for four-node full-model PCP validation.
#
# Defaults target the 192.168.25.209/212/216/217 cluster. To run on another
# four-node cluster, override NODE_IPS without editing this file:
#
#   NODE_IPS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4 \
#     ./run_64p_4node_full_pcp_portable.sh start a2a
#
# The first address is always rank 0, the API host, and the controller host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="${SCRIPT_DIR}/run_64p_4node_full_pcp_cluster.sh"

if [[ ! -x "${CONTROLLER}" ]]; then
    echo "Cluster controller is missing or not executable: ${CONTROLLER}" >&2
    exit 2
fi

export NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
export CONTAINER_NAME="${CONTAINER_NAME:-sglang-zkk-k3}"
export REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${SCRIPT_DIR}}"
export MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
export NET_IFACE="${NET_IFACE:-enp196s0f0}"
export PORT="${PORT:-15000}"
export DIST_PORT="${DIST_PORT:-15100}"
export STARTUP_CHECK_SECONDS="${STARTUP_CHECK_SECONDS:-10}"

if [[ "${1:-}" == "config" ]]; then
    cat <<EOF
Four-node PCP portable profile:
  NODE_IPS=${NODE_IPS}
  rank0/API=${NODE_IPS%%,*}
  CONTAINER_NAME=${CONTAINER_NAME}
  REMOTE_REPO_ROOT=${REMOTE_REPO_ROOT}
  MODEL_PATH=${MODEL_PATH}
  NET_IFACE=${NET_IFACE}
  PORT=${PORT}
  DIST_PORT=${DIST_PORT}
  DEEPEP_MODE=${DEEPEP_MODE:-normal}
EOF
    exit 0
fi

exec "${CONTROLLER}" "$@"
