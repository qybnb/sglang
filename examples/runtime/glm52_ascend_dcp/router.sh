#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${P_IP:?Set P_IP to the Prefill node address}"
: "${D_IP:?Set D_IP to the Decode node address}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

exec "$PYTHON_BIN" -m sglang_router.launch_router \
    --pd-disaggregation --policy round_robin \
    --prefill "http://$P_IP:$P_PORT" "$P_BOOTSTRAP_PORT" \
    --decode "http://$D_IP:$D_PORT" \
    --host "${ROUTER_HOST:-127.0.0.1}" --mini-lb --port "${ROUTER_PORT:-5599}"
