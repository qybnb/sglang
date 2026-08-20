#!/usr/bin/env bash

# Run all five Qwen3.6-27B W8A8 GPQA A/B commits sequentially on one NPU pair.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_qwen36_gpqa_ab.sh" all
