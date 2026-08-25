#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}
SERVICE_PORT=${SERVICE_PORT:-15000}
BASE_URL=http://${SERVICE_HOST}:${SERVICE_PORT}
MODEL_PATH=${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}
RESULT_ROOT=${RESULT_ROOT:-${ROOT_DIR}/logs/kimi_k3_mixed_accuracy}
RUN_LABEL=${RUN_LABEL:-mixed}

# Accuracy requests must bypass the proxy used by some test containers.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost},${SERVICE_HOST}"
export no_proxy="${no_proxy:-127.0.0.1,localhost},${SERVICE_HOST}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_kimi_k3_mixed_accuracy.sh [quick|standard|full]

The script evaluates an already-running co-located (non-PD) service. It never
starts or stops the service.

Required when the defaults do not exist:
  GSM8K_DATA_PATH=/path/to/test.jsonl
  GPQA_DATA_PATH=/path/to/gpqa_diamond.csv

Common overrides:
  SERVICE_HOST=127.0.0.1 SERVICE_PORT=15000
  MODEL_PATH=/home/weights/Kimi-K3-w4a8-int-moe
  RUN_LABEL=pcp_off|a2a|fla_ring
  RESULT_ROOT=/path/to/results
  GSM8K_PARALLEL=4 GPQA_PARALLEL=8
  GSM8K_MAX_TOKENS=512 GPQA_MAX_TOKENS=32768

Profiles:
  quick:     10 GSM8K + 10 GPQA questions
  standard: 200 GSM8K + all 198 GPQA Diamond questions (default)
  full:     all 1314 held-out GSM8K + all GPQA Diamond questions

Explicit GSM8K_NUM_QUESTIONS or GPQA_NUM_EXAMPLES overrides the profile.
Set CONFIG_ONLY=1 to validate inputs and print the resolved configuration
without sending requests.
EOF
}

PROFILE=${1:-standard}
case "${PROFILE}" in
  quick)
    DEFAULT_GSM8K_QUESTIONS=10
    DEFAULT_GPQA_EXAMPLES=10
    ;;
  standard)
    DEFAULT_GSM8K_QUESTIONS=200
    DEFAULT_GPQA_EXAMPLES=198
    ;;
  full)
    DEFAULT_GSM8K_QUESTIONS=1314
    DEFAULT_GPQA_EXAMPLES=198
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    usage >&2
    exit 2
    ;;
esac

GSM8K_NUM_QUESTIONS=${GSM8K_NUM_QUESTIONS:-${DEFAULT_GSM8K_QUESTIONS}}
GPQA_NUM_EXAMPLES=${GPQA_NUM_EXAMPLES:-${DEFAULT_GPQA_EXAMPLES}}
GSM8K_NUM_SHOTS=${GSM8K_NUM_SHOTS:-5}
GSM8K_PARALLEL=${GSM8K_PARALLEL:-4}
GPQA_PARALLEL=${GPQA_PARALLEL:-8}
GSM8K_MAX_TOKENS=${GSM8K_MAX_TOKENS:-512}
GPQA_MAX_TOKENS=${GPQA_MAX_TOKENS:-32768}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
GPQA_REASONING_EFFORT=${GPQA_REASONING_EFFORT:-max}
GPQA_THINKING_MODE=${GPQA_THINKING_MODE:-}

resolve_dataset() {
  local explicit_path=$1
  shift
  if [[ -n "${explicit_path}" ]]; then
    if [[ ! -s "${explicit_path}" ]]; then
      echo "Dataset does not exist or is empty: ${explicit_path}" >&2
      return 1
    fi
    printf '%s\n' "${explicit_path}"
    return 0
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

GSM8K_DATA_PATH=$(resolve_dataset "${GSM8K_DATA_PATH:-}" \
  /tmp/test.jsonl \
  "${ROOT_DIR}/test.jsonl" \
  /home/q00886407/datasets/gsm8k/test.jsonl) || {
  echo "Cannot find GSM8K test.jsonl; set GSM8K_DATA_PATH explicitly." >&2
  exit 2
}

GPQA_DATA_PATH=$(resolve_dataset "${GPQA_DATA_PATH:-}" \
  /tmp/gpqa_diamond.csv \
  "${ROOT_DIR}/gpqa_diamond.csv" \
  /home/q00886407/datasets/gpqa/gpqa_diamond.csv \
  /home/hanwlax/datasets/gpqa/gpqa_diamond.csv) || {
  echo "Cannot find GPQA Diamond CSV; set GPQA_DATA_PATH explicitly." >&2
  exit 2
}

if ! [[ "${GSM8K_NUM_QUESTIONS}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${GPQA_NUM_EXAMPLES}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${GSM8K_PARALLEL}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${GPQA_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Question counts and parallelism must be positive integers." >&2
  exit 2
fi

"${PYTHON_BIN}" - "${GSM8K_DATA_PATH}" "${GSM8K_NUM_SHOTS}" \
  "${GSM8K_NUM_QUESTIONS}" "${GPQA_DATA_PATH}" \
  "${GPQA_NUM_EXAMPLES}" <<'PY'
import csv
import json
import sys

gsm8k_path, num_shots, num_questions, gpqa_path, num_gpqa = sys.argv[1:]
num_shots = int(num_shots)
num_questions = int(num_questions)
num_gpqa = int(num_gpqa)

with open(gsm8k_path, encoding="utf-8") as f:
    gsm8k_rows = [json.loads(line) for line in f if line.strip()]
if len(gsm8k_rows) < num_shots + num_questions:
    raise SystemExit(
        f"GSM8K has {len(gsm8k_rows)} rows, but {num_shots} shots + "
        f"{num_questions} evaluated questions were requested"
    )
for column in ("question", "answer"):
    if column not in gsm8k_rows[0]:
        raise SystemExit(f"GSM8K dataset is missing column: {column}")

with open(gpqa_path, encoding="utf-8", newline="") as f:
    gpqa_reader = csv.DictReader(f)
    gpqa_rows = list(gpqa_reader)
required_gpqa_columns = {
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
}
missing = required_gpqa_columns.difference(gpqa_reader.fieldnames or [])
if missing:
    raise SystemExit(f"GPQA dataset is missing columns: {sorted(missing)}")
if len(gpqa_rows) < num_gpqa:
    raise SystemExit(
        f"GPQA has {len(gpqa_rows)} rows, but {num_gpqa} examples were requested"
    )
PY

SAFE_LABEL=${RUN_LABEL//[^a-zA-Z0-9_.-]/_}
RUN_TAG=${RUN_TAG:-${SAFE_LABEL}_${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RESULT_ROOT}/${RUN_TAG}
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/gsm8k" "${RUN_DIR}/gpqa"
STATUS_FILE=${RUN_DIR}/STATUS
printf 'RUNNING\n' >"${STATUS_FILE}"
trap 'rc=$?; if (( rc != 0 )); then printf "FAILED rc=%s\n" "${rc}" >"${STATUS_FILE}"; fi' EXIT

COMMIT=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)
GSM8K_SHA256=$(sha256sum "${GSM8K_DATA_PATH}" | awk '{print $1}')
GPQA_SHA256=$(sha256sum "${GPQA_DATA_PATH}" | awk '{print $1}')

export PROFILE RUN_LABEL RUN_TAG RUN_DIR ROOT_DIR COMMIT BASE_URL MODEL_PATH
export GSM8K_DATA_PATH GPQA_DATA_PATH GSM8K_SHA256 GPQA_SHA256
export GSM8K_NUM_QUESTIONS GPQA_NUM_EXAMPLES GSM8K_NUM_SHOTS
export GSM8K_PARALLEL GPQA_PARALLEL GSM8K_MAX_TOKENS GPQA_MAX_TOKENS
export TEMPERATURE TOP_P GPQA_REASONING_EFFORT GPQA_THINKING_MODE

"${PYTHON_BIN}" - <<'PY' >"${RUN_DIR}/config.json"
import json
import os

keys = [
    "PROFILE", "RUN_LABEL", "RUN_TAG", "RUN_DIR", "ROOT_DIR", "COMMIT",
    "BASE_URL", "MODEL_PATH", "GSM8K_DATA_PATH", "GPQA_DATA_PATH",
    "GSM8K_SHA256", "GPQA_SHA256", "GSM8K_NUM_QUESTIONS",
    "GPQA_NUM_EXAMPLES", "GSM8K_NUM_SHOTS", "GSM8K_PARALLEL",
    "GPQA_PARALLEL", "GSM8K_MAX_TOKENS", "GPQA_MAX_TOKENS",
    "TEMPERATURE", "TOP_P", "GPQA_REASONING_EFFORT",
    "GPQA_THINKING_MODE",
]
print(json.dumps({key: os.environ.get(key) for key in keys}, indent=2))
PY

echo "Run directory: ${RUN_DIR}"
echo "Endpoint:      ${BASE_URL}"
echo "Model:         ${MODEL_PATH}"
echo "Profile:       ${PROFILE}"
echo "GSM8K:        ${GSM8K_NUM_QUESTIONS} questions, parallel=${GSM8K_PARALLEL}"
echo "GPQA:         ${GPQA_NUM_EXAMPLES} questions, parallel=${GPQA_PARALLEL}"
echo "Config:        ${RUN_DIR}/config.json"

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  printf 'CONFIG_ONLY\n' >"${STATUS_FILE}"
  exit 0
fi

curl --noproxy '*' --fail --silent --show-error --max-time 15 \
  "${BASE_URL%/}/health" >"${RUN_DIR}/health.txt"
curl --noproxy '*' --fail --silent --show-error --max-time 15 \
  "${BASE_URL%/}/v1/models" >"${RUN_DIR}/models.json"

echo "[1/2] Running GSM8K..."
PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${ROOT_DIR}/benchmark/gsm8k/bench_sglang.py" \
    --data-path "${GSM8K_DATA_PATH}" \
    --num-shots "${GSM8K_NUM_SHOTS}" \
    --num-questions "${GSM8K_NUM_QUESTIONS}" \
    --max-new-tokens "${GSM8K_MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --parallel "${GSM8K_PARALLEL}" \
    --host "${SERVICE_HOST}" \
    --port "${SERVICE_PORT}" \
    --backend srt \
    --result-file "${RUN_DIR}/gsm8k/summary.jsonl" \
    --raw-result-file "${RUN_DIR}/gsm8k/raw_results.jsonl" \
    2>&1 | tee "${RUN_DIR}/gsm8k/run.log"

echo "[2/2] Running GPQA Diamond..."
GPQA_ARGS=(
  --base-url "${BASE_URL%/}"
  --model "${MODEL_PATH}"
  --eval-name gpqa
  --gpqa-data-path "${GPQA_DATA_PATH}"
  --num-examples "${GPQA_NUM_EXAMPLES}"
  --num-threads "${GPQA_PARALLEL}"
  --max-tokens "${GPQA_MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --output-dir "${RUN_DIR}/gpqa"
  --raw-result-file "${RUN_DIR}/gpqa/raw_results.jsonl"
)
if [[ -n "${GPQA_REASONING_EFFORT}" ]]; then
  GPQA_ARGS+=(--reasoning-effort "${GPQA_REASONING_EFFORT}")
fi
if [[ -n "${GPQA_THINKING_MODE}" ]]; then
  GPQA_ARGS+=(--thinking-mode "${GPQA_THINKING_MODE}")
fi
PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m sglang.test.run_eval "${GPQA_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/gpqa/run.log"

GSM8K_SUMMARY=${RUN_DIR}/gsm8k/summary.jsonl
GPQA_SUMMARY=$(find "${RUN_DIR}/gpqa" -maxdepth 1 -type f -name 'gpqa_*.json' -print -quit)
export GSM8K_SUMMARY GPQA_SUMMARY
"${PYTHON_BIN}" - <<'PY' >"${RUN_DIR}/summary.json"
import json
import os

with open(os.environ["GSM8K_SUMMARY"], encoding="utf-8") as f:
    gsm8k = json.loads(list(f)[-1])
with open(os.environ["GPQA_SUMMARY"], encoding="utf-8") as f:
    gpqa = json.load(f)

summary = {
    "run_label": os.environ["RUN_LABEL"],
    "profile": os.environ["PROFILE"],
    "commit": os.environ["COMMIT"],
    "gsm8k": gsm8k,
    "gpqa": gpqa,
}
print(json.dumps(summary, indent=2))
PY

printf 'SUCCESS\n' >"${STATUS_FILE}"
echo "Accuracy run complete: ${RUN_DIR}"
cat "${RUN_DIR}/summary.json"
