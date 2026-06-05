#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 RUN_ID [ROUTER_VARIANT]" >&2
  exit 2
fi

REPO_ROOT="${SWEPRO_REPO_ROOT:-/code/slime}"
RUN_CONFIG="${SWEPRO_RUN_CONFIG:-examples/swebench-pro/router_bench_trace_replay.yaml}"

export SWEPRO_RUN_ID="$1"
if [ "$#" -ge 2 ]; then
  export SWEPRO_ROUTER_VARIANT="$2"
fi

cd "${REPO_ROOT}"
python3 examples/swebench-pro/launch_swepro_rl.py \
  --repo-root "${REPO_ROOT}" \
  --run-config "${RUN_CONFIG}" \
  --print-env
