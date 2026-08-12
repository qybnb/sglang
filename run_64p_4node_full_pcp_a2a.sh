#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_TAG="${RUN_TAG:-B_full_pcp_a2a_allgather}"
exec "${SCRIPT_DIR}/run_64p_4node_full_pcp.sh" a2a "$@"
