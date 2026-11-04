#!/usr/bin/env bash

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
GLM52_RUNTIME_DIR=${GLM52_RUNTIME_DIR:-"${XDG_CACHE_HOME:-$HOME/.cache}/sglang/glm52-ascend-dcp"}
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/home/weights/GLM-5.2-w4a8}
P_PORT=${P_PORT:-8003}
P_BOOTSTRAP_PORT=${P_BOOTSTRAP_PORT:-8998}
D_PORT=${D_PORT:-8004}
D_DIST_PORT=${D_DIST_PORT:-5000}
MF_PORT=${MF_PORT:-24667}

load_ascend_env() {
    source "${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
    source "${ATB_ENV_SCRIPT:-/usr/local/Ascend/nnal/atb/set_env.sh}"
}

prepare_server_runtime() {
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "Missing model config at $MODEL_PATH/config.json" >&2
        exit 1
    fi
    if [[ ! -f "$GLM52_RUNTIME_DIR/kernel-runtime/sgl_kernel_npu/norm/add_rmsnorm_bias.py" ]]; then
        echo "Run setup-kernel.sh to install the pinned kernel into $GLM52_RUNTIME_DIR" >&2
        exit 1
    fi

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING PYTHONPATH
    load_ascend_env
    export PYTHONPATH="$GLM52_RUNTIME_DIR/kernel-runtime:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$GLM52_RUNTIME_DIR/triton-cache}"
    export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error
    export SGLANG_SET_CPU_AFFINITY=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export STREAMS_PER_DEVICE=32
}

prepare_server() {
    : "${P_IP:?Set P_IP to the Prefill node address}"
    prepare_server_runtime
    export ASCEND_MF_TRANSFER_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-sdma}"
    export ASCEND_MF_STORE_URL="tcp://$P_IP:$MF_PORT"
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
    export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
}

run_server() {
    local mode=$1
    shift
    case "$mode" in
        serve)
            exec "$PYTHON_BIN" -m sglang.launch_server "$@"
            ;;
        check)
            "$PYTHON_BIN" - "$@" <<'PY'
import sys
from importlib.util import find_spec

from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import prepare_server_args

for package in ("sglang", "sgl_kernel_npu", "deep_ep"):
    spec = find_spec(package)
    if spec is None:
        sys.exit(f"Missing package: {package}")
    print(f"{package}: {spec.origin}")
print(f"add_rmsnorm_bias: {add_rmsnorm_bias.__module__}")

load_plugins()
args = prepare_server_args(sys.argv[1:])
args.resolve_once()
cfg = resolving_view(args)
for key in (
    "device", "disaggregation_mode", "disaggregation_transfer_backend",
    "tp_size", "pp_size", "attn_dp_size", "ep_size", "dcp_size", "port",
):
    print(f"{key}={getattr(cfg, key)}")
PY
            ;;
        *)
            echo "Usage: bash $(basename "$0") [serve|check]" >&2
            exit 2
            ;;
    esac
}
