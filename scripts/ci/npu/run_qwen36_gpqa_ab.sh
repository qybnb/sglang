#!/usr/bin/env bash

# One-command Qwen3.6-27B W8A8 GPQA commit A/B runner for an NPU container.
#
# Quick start after cloning this branch:
#   DEVICES=0,1 MODEL_PATH=/data/weights/Qwen3.6-27B-w8a8 \
#     bash scripts/ci/npu/run_qwen36_gpqa_ab.sh onset
#
# Groups:
#   onset: PR #34916 parent/commit (08-16 onset)
#   npu:   PR #33676 parent/commit (08-17 NPU target-verify change)
#   war:   PR #35059 parent/commit (speculative shared-read change)
#   all:   all five unique commits

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(dirname "$REPO")/qwen36_gpqa_artifacts}"
WTROOT="${WTROOT:-$ARTIFACT_ROOT/worktrees}"
LOGROOT="${LOGROOT:-$ARTIFACT_ROOT/logs}"
DEVICES="${DEVICES:-0,1}"
RUNS="${RUNS:-1}"
MODEL_PATH="${MODEL_PATH:-/data/weights/Qwen3.6-27B-w8a8}"
MODEL_LINK="/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.6-27B-w8a8"

TEST_RELATIVE_PATH="test/registered/npu/accuracy/qwen3_6_27b/test_npu_qwen3_6_27b_w8a8_1p_in3k5_out1k5_50ms_gpqa.py"
PARALLEL_OUTPUT_PATCH="$SCRIPT_DIR/qwen36_gpqa_parallel_output.patch"

ONSET_BEFORE="4d0c5a89af7061178e4b7ab11d84c4dd2bc92482"
ONSET_AFTER="0f7aaceda5e60a2df56472f87b2b366665215daa"
NPU_BEFORE="e03c53fc13abb7b7441c07250bb75cd7ecbe9899"
NPU_AFTER="b83d507cd711ee8726c1505da77160474f9e0b19"
WAR_AFTER="711bdacb825231d01d8d0da2431d2ccdc0b28c1f"

usage() {
  cat <<'EOF'
Usage:
  run_qwen36_gpqa_ab.sh onset       PR #34916 parent/commit (default)
  run_qwen36_gpqa_ab.sh npu         PR #33676 parent/commit
  run_qwen36_gpqa_ab.sh war         PR #35059 parent/commit
  run_qwen36_gpqa_ab.sh all         All five unique commits
  run_qwen36_gpqa_ab.sh COMMIT...   Explicit commit IDs

Environment:
  DEVICES=0,1                 NPU pair (default: 0,1)
  MODEL_PATH=/data/weights/... Physical Qwen3.6-27B W8A8 model directory
  RUNS=2                      Full testcase runs per commit (default: 1)
  ARTIFACT_ROOT=/path         Worktrees and logs directory
  CANN_ENV_SCRIPT=/path       Override auto-detected CANN set_env.sh
  ATB_ENV_SCRIPT=/path        Override auto-detected ATB set_env.sh

Examples:
  DEVICES=0,1 MODEL_PATH=/data/weights/Qwen3.6-27B-w8a8 \
    bash scripts/ci/npu/run_qwen36_gpqa_ab.sh onset

  DEVICES=12,13 RUNS=2 bash scripts/ci/npu/run_qwen36_gpqa_ab.sh all
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

find_first_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_commits() {
  case "${1:-onset}" in
    onset) COMMITS=("$ONSET_BEFORE" "$ONSET_AFTER") ;;
    npu) COMMITS=("$NPU_BEFORE" "$NPU_AFTER") ;;
    war) COMMITS=("$NPU_AFTER" "$WAR_AFTER") ;;
    all)
      COMMITS=(
        "$ONSET_BEFORE" "$ONSET_AFTER" "$NPU_BEFORE" "$NPU_AFTER" "$WAR_AFTER"
      )
      ;;
    -h|--help|help) usage; exit 0 ;;
    *) COMMITS=("$@") ;;
  esac
}

validate_host() {
  local cmd
  for cmd in git python3 tee npu-smi; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
  done

  [[ "$(id -u)" -eq 0 ]] || die "Run this script as root inside the NPU container"
  [[ -d "$REPO/.git" ]] || die "Not a Git repository: $REPO"
  [[ -d "$MODEL_PATH" ]] || die "Model directory is missing: $MODEL_PATH"
  [[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"
  [[ -n "$DEVICES" ]] || die "DEVICES must not be empty"
  mkdir -p "$WTROOT" "$LOGROOT" "$LOGROOT/reports"
}

load_ascend_environment() {
  local cann_script atb_script cann_home

  cann_script="$(find_first_file \
    "${CANN_ENV_SCRIPT:-}" \
    "${ASCEND_HOME_PATH:-}/set_env.sh" \
    /usr/local/Ascend/cann/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh)" || \
    die "Cannot find CANN set_env.sh; set CANN_ENV_SCRIPT explicitly"

  # Vendor setup scripts read optional variables such as ZSH_VERSION without
  # guarding them. Temporarily disable nounset while sourcing, then restore it.
  set +u
  # shellcheck disable=SC1090
  source "$cann_script"
  set -u
  cann_home="$(cd -- "$(dirname -- "$cann_script")" && pwd)"

  atb_script="$(find_first_file \
    "${ATB_ENV_SCRIPT:-}" \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/nnal/atb/latest/set_env.sh || true)"
  if [[ -n "$atb_script" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$atb_script" --cxx_abi=1
    set -u
  fi

  BASE_PYTHONPATH="${PYTHONPATH:-}"
  [[ -f "$cann_home/lib64/libhccl.so" ]] || \
    die "libhccl.so is missing under $cann_home/lib64"
  [[ ":${LD_LIBRARY_PATH:-}:" == *":$cann_home/lib64:"* ]] || \
    die "$cann_home/lib64 is absent from LD_LIBRARY_PATH after sourcing $cann_script"

  echo "CANN_ENV_SCRIPT=$cann_script"
  echo "ATB_ENV_SCRIPT=${atb_script:-not-found}"
}

ensure_compatibility_links() {
  local launcher="/root/sglang/python/sglang/test/ascend/e2e/run_evalscope.sh"
  local resolved_model

  if [[ ! -e /root/sglang && ! -L /root/sglang ]]; then
    ln -s "$REPO" /root/sglang
  fi
  [[ -f "$launcher" ]] || \
    die "/root/sglang must point to this checkout; currently: $(readlink -f /root/sglang 2>/dev/null || echo unknown)"

  mkdir -p "$(dirname "$MODEL_LINK")"
  if [[ ! -e "$MODEL_LINK" && ! -L "$MODEL_LINK" ]]; then
    ln -s "$MODEL_PATH" "$MODEL_LINK"
  fi
  resolved_model="$(readlink -f "$MODEL_LINK" 2>/dev/null || true)"
  [[ "$resolved_model" == "$(readlink -f "$MODEL_PATH")" ]] || \
    die "$MODEL_LINK resolves to $resolved_model, expected $(readlink -f "$MODEL_PATH")"
}

prepare_worktree() {
  local requested="$1" commit short wt accuracy_utils

  commit="$(git -C "$REPO" rev-parse --verify "${requested}^{commit}")" || \
    die "Commit $requested is unavailable. Clone this branch without --depth, or run git fetch --unshallow."
  short="${commit:0:10}"
  wt="$WTROOT/$short"

  if [[ ! -e "$wt/.git" ]]; then
    [[ ! -e "$wt" ]] || die "$wt exists but is not a Git worktree"
    git -C "$REPO" worktree add --detach "$wt" "$commit" >&2 || \
      die "Failed to create worktree for $commit"
  fi

  [[ "$(git -C "$wt" rev-parse HEAD)" == "$commit" ]] || \
    die "$wt is not at $commit"
  [[ -f "$wt/$TEST_RELATIVE_PATH" ]] || die "Test file is missing at $commit"

  # The historical harness uses one fixed /tmp config file and one fixed
  # evalscope output directory. Apply an instrumentation-only patch so several
  # commits can run concurrently without overwriting one another's config or
  # reports. This does not modify inference code.
  accuracy_utils="$wt/python/sglang/test/ascend/e2e/test_npu_accuracy_utils.py"
  if ! grep -q 'SGLANG_TEST_OUTPUT_SUFFIX' "$accuracy_utils"; then
    [[ -f "$PARALLEL_OUTPUT_PATCH" ]] || die "Missing $PARALLEL_OUTPUT_PATCH"
    git -C "$wt" apply "$PARALLEL_OUTPUT_PATCH" || \
      die "Failed to apply parallel-output harness patch in $wt"
  fi
  printf '%s\n' "$commit"
}

archive_report() {
  local log="$1" short="$2" run_no="$3" timestamp="$4"
  local report report_abs destination

  report="$(sed -n 's/.*Dump report to:[[:space:]]*//p' "$log" | tail -n 1)"
  [[ -n "$report" ]] || return 0
  if [[ "$report" = /* ]]; then report_abs="$report"; else report_abs="$REPO/$report"; fi

  if [[ -f "$report_abs" ]]; then
    destination="$LOGROOT/reports/${short}_run${run_no}_${timestamp}_$(basename "$report_abs")"
    cp -a "$report_abs" "$destination"
    echo "COPIED_REPORT=$destination" | tee -a "$log"
  else
    echo "REPORT_NOT_FOUND=$report_abs" | tee -a "$log"
  fi
}

run_one() {
  local commit="$1" run_no="$2" short wt timestamp log rc
  short="${commit:0:10}"
  wt="$WTROOT/$short"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$LOGROOT/gpqa_${short}_run${run_no}_${timestamp}.log"

  echo
  echo "============================================================"
  echo "COMMIT=$commit"
  echo "WORKTREE=$wt"
  echo "DEVICES=$DEVICES"
  echo "RUN=$run_no/$RUNS"
  echo "LOG=$log"
  echo "============================================================"

  (
    set -e
    cd "$REPO"
    export PYTHONPATH="$wt/python${BASE_PYTHONPATH:+:$BASE_PYTHONPATH}"
    export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
    export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
    export SGLANG_TEST_OUTPUT_SUFFIX="${short}_run${run_no}_dev${DEVICES//,/}"
    export ATB_SHARE_MEMORY_NAME_SUFFIX="$SGLANG_TEST_OUTPUT_SUFFIX"
    unset SGLANG_ENABLE_WAR_BARRIER
    unset SGLANG_FORCE_COARSE_WAR_BARRIER

    echo "START_TIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "TEST_COMMIT=$(git -C "$wt" rev-parse HEAD)"
    echo "SGLANG_EXPECTED_PREFIX=$wt/python"
    echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    npu-smi info
    python3 -c "import sglang; print('SGLANG_IMPORT_PATH=' + sglang.__file__)"
    python3 -u "$wt/$TEST_RELATIVE_PATH"
  ) 2>&1 | tee "$log"

  rc=${PIPESTATUS[0]}
  echo "TEST_EXIT_CODE=$rc" | tee -a "$log"
  echo "END_TIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
  archive_report "$log" "$short" "$run_no" "$timestamp"
  printf '%s\t%s\t%s\t%s\n' "$commit" "$run_no" "$rc" "$log" >> "$SUMMARY_FILE"
}

main() {
  local requested commit run_no
  resolve_commits "$@"
  validate_host
  load_ascend_environment
  ensure_compatibility_links

  SUMMARY_FILE="$LOGROOT/summary_$(date -u +%Y%m%dT%H%M%SZ)_pid$$.tsv"
  printf 'commit\trun\texit_code\tlog_path\n' > "$SUMMARY_FILE"

  echo "REPO=$REPO"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "DEVICES=$DEVICES"
  echo "RUNS=$RUNS"
  echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  echo "Confirm the selected NPU pair is free:"
  npu-smi info || die "npu-smi info failed"

  for requested in "${COMMITS[@]}"; do
    commit="$(prepare_worktree "$requested")"
    for ((run_no = 1; run_no <= RUNS; run_no++)); do
      run_one "$commit" "$run_no"
    done
  done

  echo
  echo "All requested runs finished: $SUMMARY_FILE"
  column -t -s $'\t' "$SUMMARY_FILE" 2>/dev/null || cat "$SUMMARY_FILE"
}

main "$@"
