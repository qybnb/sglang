#!/usr/bin/env bash

set -euo pipefail

EVALSCOPE_HOME="${EVALSCOPE_HOME:-/home/wzy/.venvs/evalscope}"
EVALSCOPE_CLI="${EVALSCOPE_HOME}/bin/evalscope"
EVALSCOPE_SITE_PACKAGES="${EVALSCOPE_HOME}/lib/python3.11/site-packages"

if [[ -n "${EVALSCOPE_PYTHON:-}" ]]; then
  PYTHON_BIN="${EVALSCOPE_PYTHON}"
elif [[ -x /usr/local/python3.11.15/bin/python3.11 ]]; then
  PYTHON_BIN=/usr/local/python3.11.15/bin/python3.11
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  echo "ERROR: Python 3.11 was not found in this container." >&2
  exit 1
fi

if [[ ! -f "${EVALSCOPE_CLI}" || ! -d "${EVALSCOPE_SITE_PACKAGES}" ]]; then
  echo "ERROR: EvalScope environment not found under ${EVALSCOPE_HOME}." >&2
  exit 1
fi

API_URL="${API_URL:-http://127.0.0.1:15010/v1}"
MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
GPQA_DATASET_PATH="${GPQA_DATASET_PATH:-/home/hanwlax/datasets/gpqa}"
WORK_ROOT="${WORK_ROOT:-/home/hanwlax/workspace/progress/kimi_k3/gpqa}"
RUN_TAG="${RUN_TAG:-gpqa_$(date +%Y-%m-%d_%H-%M-%S)}"
WORK_DIR="${WORK_DIR:-${WORK_ROOT}/${RUN_TAG}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"

if [[ -n "${LIMIT}" && ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: LIMIT must be a positive integer when set, got: ${LIMIT}" >&2
  exit 2
fi

if [[ ! -d "${GPQA_DATASET_PATH}" ]]; then
  echo "ERROR: GPQA dataset directory does not exist: ${GPQA_DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

# Requests target a local OpenAI-compatible endpoint and must bypass any proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${EVALSCOPE_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(
  "${PYTHON_BIN}" "${EVALSCOPE_CLI}" eval
  --model "${MODEL_PATH}"
  --api-url "${API_URL}"
  --api-key EMPTY
  --work-dir "${WORK_DIR}"
  --no-timestamp
  --eval-type openai_api
  --datasets gpqa_diamond
  --dataset-args "{\"gpqa_diamond\":{\"local_path\":\"${GPQA_DATASET_PATH}\",\"subset_list\":[\"gpqa_diamond\"],\"default_subset\":\"gpqa_diamond\"}}"
  --generation-config '{"max_tokens":131072,"timeout":10000,"temperature":1.0,"top_p":0.95,"extra_body":{"reasoning_effort":"max"}}'
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --seed "${SEED}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

if [[ -n "${USE_CACHE:-}" ]]; then
  cmd+=(--use-cache "${USE_CACHE}")
fi

echo "EvalScope GPQA starting"
echo "  API:       ${API_URL}"
echo "  work dir:  ${WORK_DIR}"
echo "  batch:     ${EVAL_BATCH_SIZE}"
echo "  limit:     ${LIMIT:-all}"
echo "  seed:      ${SEED}"
echo "  live log:  ${WORK_DIR}/evalscope.log"

set +e
"${cmd[@]}" 2>&1 | tee "${WORK_DIR}/evalscope.log"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
