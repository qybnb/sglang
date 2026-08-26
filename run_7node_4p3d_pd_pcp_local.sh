#!/usr/bin/env bash
#
# Node-local Kimi-K3 seven-node 4P3D launcher.
#
# Topology on seven 16-NPU nodes:
#   P0-P3: Prefill, TP64 / PP1 / DP2 / CP2
#   D0-D2: Decode,  TP48 / PP1 / DP1 / CP1
#
# Run this script manually inside the same container on every node. It never
# uses SSH. All nodes must receive the same PREFILL_IPS and DECODE_IPS; each
# node supplies its own NODE_RANK, LOCAL_IP, NET_IFACE and checkpoint path.
#
# Decode enables DeepEP, dSparK and Decode NPU Graph by default. Every exported
# default remains externally overridable.
#
# Examples:
#   P0: NODE_RANK=0 LOCAL_IP=<P0> NET_IFACE=<nic0> MODEL_PATH=<model0> \
#         ./run_7node_4p3d_pd_pcp_local.sh prefill
#   P3: NODE_RANK=3 LOCAL_IP=<P3> NET_IFACE=<nic3> MODEL_PATH=<model3> \
#         ./run_7node_4p3d_pd_pcp_local.sh prefill
#   D0: NODE_RANK=0 LOCAL_IP=<D0> NET_IFACE=<nic4> MODEL_PATH=<model4> \
#         ./run_7node_4p3d_pd_pcp_local.sh decode
#   D2: NODE_RANK=2 LOCAL_IP=<D2> NET_IFACE=<nic6> MODEL_PATH=<model6> \
#         ./run_7node_4p3d_pd_pcp_local.sh decode
#   P0: ./run_7node_4p3d_pd_pcp_local.sh router

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PREFILL_NNODES="${PREFILL_NNODES:-4}"
export DECODE_NNODES="${DECODE_NNODES:-3}"
export DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-seven-node 4P3D}"
export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-64}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-48}"
export PP_SIZE="${PP_SIZE:-1}"
export PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
export PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-2}"
export DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"

export PREFILL_ENABLE_DP_ATTENTION="${PREFILL_ENABLE_DP_ATTENTION:-1}"
export PREFILL_ENABLE_DEEPEP="${PREFILL_ENABLE_DEEPEP:-1}"
export DECODE_ENABLE_DP_ATTENTION="${DECODE_ENABLE_DP_ATTENTION:-0}"
export DECODE_ENABLE_DEEPEP="${DECODE_ENABLE_DEEPEP:-1}"

export PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.82}"
export DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.75}"
export PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-4}"
export DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-4}"
export PREFILL_HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-2000}"
export DECODE_HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-2000}"

export ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
export CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 4}"
export ALLOW_DSPARK_HETERO_ATTN_TP="${ALLOW_DSPARK_HETERO_ATTN_TP:-1}"

export PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-15401}"
export DECODE_DIST_PORT="${DECODE_DIST_PORT:-15402}"
export PREFILL_PORT="${PREFILL_PORT:-31301}"
export DECODE_PORT="${DECODE_PORT:-31302}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-19298}"
export ROUTER_PORT="${ROUTER_PORT:-18377}"
export MF_STORE_PORT="${MF_STORE_PORT:-34673}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20600}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-20600-20799}"

export RUN_TAG="${RUN_TAG:-pd7_4p3d_pcp_fla_ring}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/kimi_k3_7node_4p3d_pd_pcp}"

exec "${SCRIPT_DIR}/run_8node_pd_pcp_local.sh" "$@"
