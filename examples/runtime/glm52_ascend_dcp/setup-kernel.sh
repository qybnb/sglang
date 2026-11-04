#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if (( $# > 1 )); then
    echo "Usage: bash setup-kernel.sh [path-to-sgl-kernel-npu-wheel]" >&2
    exit 2
fi
if [[ -e "$GLM52_RUNTIME_DIR/kernel-runtime" ]]; then
    echo "Kernel target already exists. Use a new GLM52_RUNTIME_DIR to install again." >&2
    exit 1
fi

load_ascend_env
mkdir -p "$GLM52_RUNTIME_DIR"
stage=$(mktemp -d "$GLM52_RUNTIME_DIR/.kernel-build.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT

if (( $# == 1 )); then
    wheel=$1
else
    kernel_commit=bebbbc732974cf955bd4dc9e54c9e4b8210f5e5f
    kernel_src="$stage/sgl-kernel-npu"
    git init -q "$kernel_src"
    git -C "$kernel_src" fetch --depth=1 \
        https://github.com/sgl-project/sgl-kernel-npu.git "$kernel_commit"
    git -C "$kernel_src" checkout -q --detach FETCH_HEAD
    git -C "$kernel_src" submodule update --init --recursive
    (
        cd "$kernel_src"
        VERSION=2026.9.0 bash build.sh -a kernels Ascend910_9382
    )
    shopt -s nullglob
    wheels=("$kernel_src"/output/sgl_kernel_npu-*.whl)
    if (( ${#wheels[@]} != 1 )); then
        echo "Expected one sgl_kernel_npu wheel from the build" >&2
        exit 1
    fi
    wheel=${wheels[0]}
fi

"$PYTHON_BIN" -m pip install --no-deps --target "$stage/runtime" "$wheel"
test -f "$stage/runtime/sgl_kernel_npu/norm/add_rmsnorm_bias.py"
mv -T "$stage/runtime" "$GLM52_RUNTIME_DIR/kernel-runtime"
echo "Kernel runtime ready at $GLM52_RUNTIME_DIR/kernel-runtime"
