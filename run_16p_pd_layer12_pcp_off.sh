#!/usr/bin/env bash
# Two-node (8 NPU + 8 NPU) 12-layer PCP-off validation baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep model/network overrides available for a server whose mount or IP differs
# from the validated 80.5.17.37/38 defaults in run_16p_pd_layer24.sh.
export MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-int4}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/kimi_k3_layer12_pd}"

export NUM_HIDDEN_LAYERS=12
export TP_SIZE=8
export PREFILL_DP_SIZE=2
export DECODE_DP_SIZE=2
export PREFILL_CP_SIZE=1
export ENABLE_PCP=0
export SGLANG_ENABLE_CP_V2=0
export KDA_CP_BACKEND=a2a
export MLA_CP_BACKEND=allgather
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=0
export MAX_TOTAL_TOKENS=32768
export MAX_RUNNING_REQUESTS=8
export CHUNKED_PREFILL_SIZE=4096
export PAGE_SIZE=128
export RUN_TAG=A_2node_12l_pcp_off

exec "${SCRIPT_DIR}/run_16p_pd_layer24.sh" "$@"
