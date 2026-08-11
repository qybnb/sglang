#!/usr/bin/env bash
# Two-node (8 NPU + 8 NPU) 12-layer PCP A2A/allgather validation profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-int4}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/kimi_k3_layer12_pd}"

export NUM_HIDDEN_LAYERS=12
export TP_SIZE=8
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=2
export PREFILL_CP_SIZE=4
export ENABLE_PCP=1
export SGLANG_ENABLE_CP_V2=1
export KDA_CP_BACKEND=a2a
export MLA_CP_BACKEND=allgather
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=0
export MAX_TOTAL_TOKENS=32768
export MAX_RUNNING_REQUESTS=8
export CHUNKED_PREFILL_SIZE=4096
export PAGE_SIZE=128
export RUN_TAG=B_2node_12l_pcp_a2a_allgather

exec "${SCRIPT_DIR}/run_16p_pd_layer24.sh" "$@"
