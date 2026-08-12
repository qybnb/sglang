#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VALIDATOR="${ROOT_DIR}/scripts/kimi_k3_pcp_validation.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:6688}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/logs/kimi_k3_pcp_validation_v2}"

ACCURACY_INPUT_LENS="${ACCURACY_INPUT_LENS:-2048,8192}"
ACCURACY_OUTPUT_LEN="${ACCURACY_OUTPUT_LEN:-32}"
ACCURACY_REPEATS="${ACCURACY_REPEATS:-3}"
PREFILL_LOGPROB_TOKENS="${PREFILL_LOGPROB_TOKENS:-64}"

PERF_INPUT_LENS="${PERF_INPUT_LENS:-1024,4096,8192}"
PERF_OUTPUT_LENS="${PERF_OUTPUT_LENS:-1,32}"
PERF_CONCURRENCIES="${PERF_CONCURRENCIES:-1,4}"
PERF_NUM_PROMPTS="${PERF_NUM_PROMPTS:-16}"
PERF_ROUNDS="${PERF_ROUNDS:-3}"
PERF_WARMUP_REQUESTS="${PERF_WARMUP_REQUESTS:-4}"

usage() {
  echo "Usage:"
  echo "  MODEL_PATH=/path/to/Kimi-K3 $0 collect A1|A2|B|C"
  echo "  $0 compare"
  echo
  echo "A1/A2 are two independently restarted PCP-off runs."
  echo "B is PCP+A2A/allgather; C is PCP+FLA/ring."
}

label_tag() {
  case "$1" in
    A1) echo "A1_pcp_off" ;;
    A2) echo "A2_pcp_off" ;;
    B) echo "B_pcp_a2a" ;;
    C) echo "C_pcp_fla_ring" ;;
    *) return 1 ;;
  esac
}

collect() {
  local label="$1"
  local tag
  tag=$(label_tag "${label}") || {
    echo "Unknown label: ${label}; expected A1, A2, B, or C" >&2
    return 2
  }
  : "${MODEL_PATH:?Set MODEL_PATH to the local Kimi-K3 checkpoint directory}"

  local smoke_file="${RESULT_DIR}/${label}_smoke.json"
  local accuracy_file="${RESULT_DIR}/${label}_accuracy.json"
  local perf_file="${RESULT_DIR}/${label}_perf.jsonl"
  for artifact in "${smoke_file}" "${accuracy_file}" "${perf_file}"; do
    if [[ -e "${artifact}" ]]; then
      echo "Refusing to overwrite existing artifact: ${artifact}" >&2
      echo "Use a new RESULT_DIR for another suite run." >&2
      return 2
    fi
  done
  mkdir -p "${RESULT_DIR}"

  echo "Collecting ${label} (${tag}) from ${BASE_URL}"
  echo "Commit: $(git -C "${ROOT_DIR}" rev-parse HEAD)"
  echo "Before continuing, confirm the running service matches label ${label}."

  "${PYTHON_BIN}" "${VALIDATOR}" smoke \
    --base-url "${BASE_URL}" \
    --tokenizer "${MODEL_PATH}" \
    --tag "${tag}" \
    --input-len 2048 \
    --output-len 32 \
    --num-requests 8 \
    --concurrency 4 \
    --output "${smoke_file}"

  "${PYTHON_BIN}" "${VALIDATOR}" accuracy \
    --base-url "${BASE_URL}" \
    --tokenizer "${MODEL_PATH}" \
    --tag "${tag}" \
    --input-lens "${ACCURACY_INPUT_LENS}" \
    --output-len "${ACCURACY_OUTPUT_LEN}" \
    --repeats "${ACCURACY_REPEATS}" \
    --prefill-logprob-tokens "${PREFILL_LOGPROB_TOKENS}" \
    --top-logprobs 5 \
    --output "${accuracy_file}"

  "${PYTHON_BIN}" "${VALIDATOR}" perf \
    --base-url "${BASE_URL}" \
    --model "${MODEL_PATH}" \
    --tokenizer "${MODEL_PATH}" \
    --tag "${tag}" \
    --input-lens "${PERF_INPUT_LENS}" \
    --output-lens "${PERF_OUTPUT_LENS}" \
    --concurrencies "${PERF_CONCURRENCIES}" \
    --num-prompts "${PERF_NUM_PROMPTS}" \
    --rounds "${PERF_ROUNDS}" \
    --warmup-requests "${PERF_WARMUP_REQUESTS}" \
    --output-file "${perf_file}"

  echo "Collection complete: ${RESULT_DIR}/${label}_*"
}

compare_one_accuracy() {
  local left="$1"
  local right="$2"
  local output="${RESULT_DIR}/comparisons/${left}_vs_${right}_accuracy.json"
  if ! "${PYTHON_BIN}" "${VALIDATOR}" compare-accuracy \
    "${RESULT_DIR}/${left}_accuracy.json" \
    "${RESULT_DIR}/${right}_accuracy.json" \
    --output "${output}"; then
    COMPARE_FAILURES=1
  fi
}

check_accuracy_stability() {
  local label="$1"
  local output="${RESULT_DIR}/comparisons/${label}_accuracy_stability.json"
  if ! "${PYTHON_BIN}" "${VALIDATOR}" accuracy-stability \
    "${RESULT_DIR}/${label}_accuracy.json" \
    --min-repeats "${ACCURACY_REPEATS}" \
    --output "${output}"; then
    COMPARE_FAILURES=1
  fi
}

compare_one_perf() {
  local left="$1"
  local right="$2"
  local output="${RESULT_DIR}/comparisons/${left}_vs_${right}_perf.json"
  if ! "${PYTHON_BIN}" "${VALIDATOR}" compare-perf \
    "${RESULT_DIR}/${left}_perf.jsonl" \
    "${RESULT_DIR}/${right}_perf.jsonl" \
    --min-rounds "${PERF_ROUNDS}" \
    --output "${output}"; then
    COMPARE_FAILURES=1
  fi
}

compare_all() {
  local required
  for required in A1 A2 B C; do
    if [[ ! -f "${RESULT_DIR}/${required}_accuracy.json" ]]; then
      echo "Missing ${RESULT_DIR}/${required}_accuracy.json" >&2
      return 2
    fi
    if [[ ! -f "${RESULT_DIR}/${required}_perf.jsonl" ]]; then
      echo "Missing ${RESULT_DIR}/${required}_perf.jsonl" >&2
      return 2
    fi
  done
  mkdir -p "${RESULT_DIR}/comparisons"
  COMPARE_FAILURES=0

  echo "=== Accuracy: within-service stability of identical prompts ==="
  check_accuracy_stability A1
  check_accuracy_stability A2
  check_accuracy_stability B
  check_accuracy_stability C

  echo "=== Accuracy: A1 vs A2 is the PCP-off natural-variation gate ==="
  compare_one_accuracy A1 A2
  compare_one_accuracy A1 B
  compare_one_accuracy A2 B
  compare_one_accuracy A1 C
  compare_one_accuracy A2 C
  compare_one_accuracy B C

  echo "=== Performance: every row is the median of fixed-length rounds ==="
  compare_one_perf A1 A2
  compare_one_perf A1 B
  compare_one_perf A2 B
  compare_one_perf A1 C
  compare_one_perf A2 C
  compare_one_perf B C

  echo "Comparison artifacts: ${RESULT_DIR}/comparisons"
  if [[ "${COMPARE_FAILURES}" -ne 0 ]]; then
    echo "At least one strict comparison failed; retain all artifacts for analysis." >&2
    return 1
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

case "$1" in
  collect)
    if [[ $# -ne 2 ]]; then
      usage
      exit 2
    fi
    collect "$2"
    ;;
  compare)
    if [[ $# -ne 1 ]]; then
      usage
      exit 2
    fi
    compare_all
    ;;
  *)
    usage
    exit 2
    ;;
esac
