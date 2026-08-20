#!/usr/bin/env bash

# Run the five Qwen3.6 GPQA candidate commits concurrently on NPU 0-9.
# Device assignment: 0,1 / 2,3 / 4,5 / 6,7 / 8,9.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)}"
RUNS="${RUNS:-1}"
MODEL_PATH="${MODEL_PATH:-/data/weights/Qwen3.6-27B-w8a8}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
RUN_ROOT="${ARTIFACT_ROOT:-$(dirname "$REPO")/qwen36_gpqa_parallel_$RUN_ID}"
WTROOT="$RUN_ROOT/worktrees"
RUNNER="$SCRIPT_DIR/run_qwen36_gpqa_ab.sh"
OUTPUT_PATCH="$SCRIPT_DIR/qwen36_gpqa_parallel_output.patch"
MODEL_LINK="/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.6-27B-w8a8"

COMMITS=(
  4d0c5a89af7061178e4b7ab11d84c4dd2bc92482
  0f7aaceda5e60a2df56472f87b2b366665215daa
  e03c53fc13abb7b7441c07250bb75cd7ecbe9899
  b83d507cd711ee8726c1505da77160474f9e0b19
  711bdacb825231d01d8d0da2431d2ccdc0b28c1f
)

DEVICE_PAIRS=("0,1" "2,3" "4,5" "6,7" "8,9")

die() {
  echo "ERROR: $*" >&2
  exit 1
}

preflight() {
  local commit short wt accuracy_utils

  [[ "$(id -u)" -eq 0 ]] || die "Run as root inside the NPU container"
  [[ -d "$REPO/.git" ]] || die "Not a Git repository: $REPO"
  [[ -d "$MODEL_PATH" ]] || die "Model directory is missing: $MODEL_PATH"
  [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
  [[ -x "$RUNNER" || -f "$RUNNER" ]] || die "Missing runner: $RUNNER"
  [[ -f "$OUTPUT_PATCH" ]] || die "Missing output isolation patch: $OUTPUT_PATCH"
  command -v npu-smi >/dev/null 2>&1 || die "npu-smi is unavailable"

  mkdir -p "$WTROOT" "$RUN_ROOT/jobs"

  if [[ ! -e /root/sglang && ! -L /root/sglang ]]; then
    ln -s "$REPO" /root/sglang
  fi
  [[ -f /root/sglang/python/sglang/test/ascend/e2e/run_evalscope.sh ]] || \
    die "/root/sglang does not provide run_evalscope.sh"

  mkdir -p "$(dirname "$MODEL_LINK")"
  if [[ ! -e "$MODEL_LINK" && ! -L "$MODEL_LINK" ]]; then
    ln -s "$MODEL_PATH" "$MODEL_LINK"
  fi
  [[ "$(readlink -f "$MODEL_LINK")" == "$(readlink -f "$MODEL_PATH")" ]] || \
    die "$MODEL_LINK points to a different model"

  # Install the shared evalscope environment once before parallel workers start.
  if [[ ! -x "$REPO/test_env_evalscope/bin/python" ]]; then
    echo "Preparing shared evalscope environment..."
    (cd "$REPO" && bash /root/sglang/python/sglang/test/ascend/e2e/run_evalscope.sh)
  fi

  # Create and instrument worktrees serially to avoid concurrent .git locks.
  for commit in "${COMMITS[@]}"; do
    git -C "$REPO" cat-file -e "${commit}^{commit}" || die "Missing commit: $commit"
    short="${commit:0:10}"
    wt="$WTROOT/$short"
    if [[ ! -e "$wt/.git" ]]; then
      git -C "$REPO" worktree add --detach "$wt" "$commit"
    fi
    [[ "$(git -C "$wt" rev-parse HEAD)" == "$commit" ]] || die "$wt is at the wrong commit"

    accuracy_utils="$wt/python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py"
    if ! grep -q 'SGLANG_TEST_OUTPUT_SUFFIX' "$accuracy_utils"; then
      git -C "$wt" apply "$OUTPUT_PATCH" || die "Cannot instrument $wt for parallel output"
    fi
  done

  echo "NPU inventory before launch:"
  npu-smi info
}

launch_all() {
  local index commit devices short job_root launch_log pid failed=0
  local -a pids=()

  for index in "${!COMMITS[@]}"; do
    commit="${COMMITS[$index]}"
    devices="${DEVICE_PAIRS[$index]}"
    short="${commit:0:10}"
    job_root="$RUN_ROOT/jobs/${index}_${short}_dev${devices//,/}"
    launch_log="$RUN_ROOT/launch_${index}_${short}_dev${devices//,/}.log"
    mkdir -p "$job_root"

    echo "Launching $short on NPU $devices -> $launch_log"
    env \
      REPO="$REPO" \
      WTROOT="$WTROOT" \
      ARTIFACT_ROOT="$job_root" \
      DEVICES="$devices" \
      MODEL_PATH="$MODEL_PATH" \
      RUNS="$RUNS" \
      bash "$RUNNER" "$commit" >"$launch_log" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "$pid $commit $devices $launch_log" >> "$RUN_ROOT/processes.txt"
  done

  echo
  echo "All five jobs launched. Process table: $RUN_ROOT/processes.txt"
  cat "$RUN_ROOT/processes.txt"
  echo

  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Launcher failed: ${COMMITS[$index]} on ${DEVICE_PAIRS[$index]}" >&2
      failed=1
    fi
  done

  echo
  echo "Parallel run finished. Artifacts: $RUN_ROOT"
  find "$RUN_ROOT/jobs" -name 'summary_*.tsv' -type f -print -exec cat {} \;
  return "$failed"
}

preflight
launch_all
