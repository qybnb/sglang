#!/usr/bin/env bash
#
# Node-local Kimi-K3 six-node 4P2D launcher.
#
# Topology on six 16-NPU nodes:
#   P0-P3: Prefill, TP64 / PP1 / DP2 / CP2
#   D0-D1: Decode,  TP32 / PP1 / DP1 / CP1
#
# Run this script manually inside the same container on every node. It never
# uses SSH. All nodes must receive the same PREFILL_IPS and DECODE_IPS; each
# node supplies its own NODE_RANK, LOCAL_IP, NET_IFACE and checkpoint path.
#
# Decode uses the minimal validation profile: DeepEP, dSparK and decode graph
# are all disabled. The full 93-layer checkpoint does not fit TP32/PP1 on the
# currently tested 64-GiB NPUs, so P and D must use the same reduced-layer
# checkpoint unless the target hardware has enough memory.
#
# Examples:
#   P0: NODE_RANK=0 LOCAL_IP=<P0> NET_IFACE=<nic0> MODEL_PATH=<model0> \
#         ./run_6node_4p2d_pd_pcp_local.sh prefill
#   P3: NODE_RANK=3 LOCAL_IP=<P3> NET_IFACE=<nic3> MODEL_PATH=<model3> \
#         ./run_6node_4p2d_pd_pcp_local.sh prefill
#   D0: NODE_RANK=0 LOCAL_IP=<D0> NET_IFACE=<nic4> MODEL_PATH=<model4> \
#         ./run_6node_4p2d_pd_pcp_local.sh decode
#   D1: NODE_RANK=1 LOCAL_IP=<D1> NET_IFACE=<nic5> MODEL_PATH=<model5> \
#         ./run_6node_4p2d_pd_pcp_local.sh decode
#   P0: ./run_6node_4p2d_pd_pcp_local.sh router

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PREFILL_NNODES=4
export DECODE_NNODES=2
export DEPLOYMENT_NAME="six-node 4P2D"
export PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-64}"
export DECODE_TP_SIZE="${DECODE_TP_SIZE:-32}"
export PP_SIZE="${PP_SIZE:-1}"
export PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
export PREFILL_CP_SIZE="${PREFILL_CP_SIZE:-2}"
export DECODE_DP_SIZE="${DECODE_DP_SIZE:-1}"

export PREFILL_ENABLE_DP_ATTENTION=1
export PREFILL_ENABLE_DEEPEP=1
export DECODE_ENABLE_DP_ATTENTION=0
export DECODE_ENABLE_DEEPEP=0

export PREFILL_MEM_FRACTION_STATIC="${PREFILL_MEM_FRACTION_STATIC:-0.82}"
export DECODE_MEM_FRACTION_STATIC="${DECODE_MEM_FRACTION_STATIC:-0.85}"
export PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-4}"
export DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-4}"
export PREFILL_HCCL_BUFFSIZE="${PREFILL_HCCL_BUFFSIZE:-2000}"
export DECODE_HCCL_BUFFSIZE="${DECODE_HCCL_BUFFSIZE:-512}"

# Keep the first 4P2D transfer/accuracy validation independent of speculative
# decoding and graph capture. They can be enabled later as separate variables.
export ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"

export PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-15301}"
export DECODE_DIST_PORT="${DECODE_DIST_PORT:-15302}"
export PREFILL_PORT="${PREFILL_PORT:-31201}"
export DECODE_PORT="${DECODE_PORT:-31202}"
export BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-19198}"
export ROUTER_PORT="${ROUTER_PORT:-18277}"
export MF_STORE_PORT="${MF_STORE_PORT:-34672}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-20400}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-20400-20599}"

export RUN_TAG="${RUN_TAG:-pd6_4p2d_pcp_fla_ring}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/kimi_k3_6node_4p2d_pd_pcp}"

exec "${SCRIPT_DIR}/run_8node_pd_pcp_local.sh" "$@"
