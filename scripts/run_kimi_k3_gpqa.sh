#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}
SERVICE_PORT=${SERVICE_PORT:-15010}
BASE_URL=${BASE_URL:-http://${SERVICE_HOST}:${SERVICE_PORT}}
MODEL_PATH=${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}
GPQA_NUM_EXAMPLES=${GPQA_NUM_EXAMPLES:-198}
GPQA_PARALLEL=${GPQA_PARALLEL:-8}
GPQA_MAX_TOKENS=${GPQA_MAX_TOKENS:-32768}
GPQA_REASONING_EFFORT=${GPQA_REASONING_EFFORT:-max}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
RESULT_ROOT=${RESULT_ROOT:-${ROOT_DIR}/logs/kimi_k3_gpqa}
RUN_LABEL=${RUN_LABEL:-fla_ring_dspark_graph}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost},${SERVICE_HOST}"
export no_proxy="${no_proxy:-127.0.0.1,localhost},${SERVICE_HOST}"

resolve_dataset() {
    local candidate
    if [[ -n "${GPQA_DATA_PATH:-}" ]]; then
        candidate=${GPQA_DATA_PATH}
        [[ -s "${candidate}" ]] || {
            echo "GPQA_DATA_PATH does not exist or is empty: ${candidate}" >&2
            return 1
        }
        printf '%s\n' "${candidate}"
        return 0
    fi
    for candidate in \
        /tmp/gpqa_diamond.csv \
        "${ROOT_DIR}/gpqa_diamond.csv" \
        /home/q00886407/datasets/gpqa/gpqa_diamond.csv \
        /home/hanwlax/datasets/gpqa/gpqa_diamond.csv; do
        if [[ -s "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    echo "Cannot find GPQA Diamond CSV; set GPQA_DATA_PATH." >&2
    return 1
}

GPQA_DATA_PATH=$(resolve_dataset)
for value in "${GPQA_NUM_EXAMPLES}" "${GPQA_PARALLEL}" "${GPQA_MAX_TOKENS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Question count, parallelism, and max tokens must be positive integers." >&2
        exit 2
    }
done

"${PYTHON_BIN}" - "${GPQA_DATA_PATH}" "${GPQA_NUM_EXAMPLES}" <<'PY'
import csv
import sys

path, requested = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
required = {
    "Question", "Correct Answer", "Incorrect Answer 1",
    "Incorrect Answer 2", "Incorrect Answer 3",
}
missing = required.difference(reader.fieldnames or [])
if missing:
    raise SystemExit(f"GPQA dataset is missing columns: {sorted(missing)}")
if len(rows) < requested:
    raise SystemExit(f"GPQA has {len(rows)} rows, requested {requested}")
PY

curl --noproxy '*' --fail --silent --show-error --max-time 60 \
    "${BASE_URL%/}/health" >/dev/null

SAFE_LABEL=${RUN_LABEL//[^a-zA-Z0-9_.-]/_}
RUN_TAG=${RUN_TAG:-${SAFE_LABEL}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${RESULT_ROOT}/${RUN_TAG}
[[ ! -e "${RUN_DIR}" ]] || {
    echo "Refusing to overwrite existing run directory: ${RUN_DIR}" >&2
    exit 2
}
mkdir -p "${RUN_DIR}"

cat >"${RUN_DIR}/config.json" <<EOF
{
  "base_url": "${BASE_URL%/}",
  "model": "${MODEL_PATH}",
  "dataset": "${GPQA_DATA_PATH}",
  "num_examples": ${GPQA_NUM_EXAMPLES},
  "parallel": ${GPQA_PARALLEL},
  "max_tokens": ${GPQA_MAX_TOKENS},
  "temperature": ${TEMPERATURE},
  "top_p": ${TOP_P},
  "reasoning_effort": "${GPQA_REASONING_EFFORT}"
}
EOF

echo "Running GPQA Diamond: examples=${GPQA_NUM_EXAMPLES}, parallel=${GPQA_PARALLEL}"
echo "Result directory: ${RUN_DIR}"

ARGS=(
    --base-url "${BASE_URL%/}"
    --model "${MODEL_PATH}"
    --eval-name gpqa
    --gpqa-data-path "${GPQA_DATA_PATH}"
    --num-examples "${GPQA_NUM_EXAMPLES}"
    --num-threads "${GPQA_PARALLEL}"
    --max-tokens "${GPQA_MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --output-dir "${RUN_DIR}"
    --raw-result-file "${RUN_DIR}/raw_results.jsonl"
)
if [[ -n "${GPQA_REASONING_EFFORT}" ]]; then
    ARGS+=(--reasoning-effort "${GPQA_REASONING_EFFORT}")
fi

PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m sglang.test.run_eval "${ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/run.log"

echo "GPQA run completed: ${RUN_DIR}"
