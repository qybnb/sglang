#!/usr/bin/env bash
#
# Node-local Kimi-K3 four-node 2P2D launcher.
#
# Topology on four 16-NPU nodes:
#   P0/P1: Prefill, TP32 / PP1 / DP1 / CP4
#   D0/D1: Decode,  TP32 / PP1 / DP1 / CP1
#
# This wrapper reuses run_8node_pd_pcp_local.sh so the PD transport, network,
# per-node checkpoint, optional dSparK, and logging behavior stay identical.
# It never uses SSH; run one command manually inside the container on each
# physical node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GROUP_NNODES=2
export DEPLOYMENT_NAME="four-node 2P2D"
export TP_SIZE="${TP_SIZE:-32}"
export PP_SIZE="${PP_SIZE:-1}"
export PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-1}"
export PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-4}"
export DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"

# Keep the established 2P2D memory defaults. The checkpoint may be reduced to
# any layer count by the caller; this launcher does not enforce a model size.
export PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.92}"
export DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.90}"

# Decode always uses dSparK and NPU Graph in this 2P2D profile. The shared
# launcher ignores these settings for the prefill role, so prefill does not
# need DRAFT_MODEL_PATH and keeps decode graph disabled.
export ENABLE_DSPARK=1
export DISABLE_CUDA_GRAPH=0
export CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 4}"

export PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-15201}"
export DECODE_DIST_PORT="${DECODE_DIST_PORT:-15202}"
export PREFILL_PORT="${PREFILL_PORT:-31101}"
export DECODE_PORT="${DECODE_PORT:-31102}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-19098}"
export ROUTER_PORT="${ROUTER_PORT:-18177}"
export MF_STORE_PORT="${MF_STORE_PORT:-34671}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20200}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-20200-20399}"

export RUN_TAG="${RUN_TAG:-pd4_2p2d_pcp_fla_ring}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/kimi_k3_4node_2p2d_pd_pcp}"

exec "${SCRIPT_DIR}/run_8node_pd_pcp_local.sh" "$@"
