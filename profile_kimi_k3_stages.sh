#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-start}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-$(pwd)/profiling/kimi_k3_${RUN_TAG}}"

if ! [[ "${PROFILE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_STEPS must be a positive integer, got: ${PROFILE_STEPS}" >&2
    exit 2
fi

case "${ACTION}" in
    start)
        mkdir -p "${PROFILE_OUTPUT_DIR}"
        curl --fail --silent --show-error \
            --request POST \
            --header "Content-Type: application/json" \
            --data-binary @- \
            "${SERVER_URL%/}/start_profile" <<JSON
{
  "output_dir": "${PROFILE_OUTPUT_DIR}",
  "num_steps": ${PROFILE_STEPS},
  "activities": ["CPU", "GPU"],
  "profile_by_stage": true,
  "profile_stages": ["prefill", "decode"],
  "with_stack": false,
  "record_shapes": true,
  "merge_profiles": false,
  "profile_prefix": "kimi-k3",
  "profile_id": "${RUN_TAG}"
}
JSON
        echo
        echo "Profiling armed: prefill=${PROFILE_STEPS} steps, decode=${PROFILE_STEPS} steps"
        echo "Trace directory: ${PROFILE_OUTPUT_DIR}"
        echo "Send the workload now. Use at least ${PROFILE_STEPS} prefill batches and ${PROFILE_STEPS} decode batches."
        ;;
    stop)
        curl --fail --silent --show-error \
            --request POST \
            "${SERVER_URL%/}/stop_profile"
        echo
        echo "Profiling stopped."
        ;;
    *)
        echo "Usage: SERVER_URL=http://host:port $0 {start|stop}" >&2
        exit 2
        ;;
esac
