#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_TAG="${RUN_TAG:-A_full_pcp_off}"
exec "${SCRIPT_DIR}/run_64p_4node_full_pcp.sh" off "$@"
