# Local SWE-bench Pro Runs

This is the short path for running SWE-bench Pro tooling from a local checkout or
from the `/code/slime` checkout mounted into the GCP pods.

## Prepare Prompt Data

Normalize the SWE-bench Pro rows into slime prompt-data JSONL:

```bash
uv run examples/swebench-pro/prepare_swebench_pro_data.py \
  --input ~/proj/SWE-bench_Pro-os/helper_code/sweap_eval_full_v2.jsonl \
  --source-root ~/proj/SWE-bench_Pro-os \
  --output /data/swebench-pro/swebench_pro_train.jsonl
```

Use `--limit` for quick local parser checks. The output rows contain the prompt,
`instance_id`, SWE-agent image metadata, test lists, and the raw source row.

## Smoke A Session Worker

When NATS and the SWE-agent session workers are up, verify one environment before
starting a rollout:

```bash
export SWEPRO_NATS_URL=nats://warnold-swepro-nats:4222
export DYNAMO_FRONTEND_URL=http://warnold-swepro-frontend:3000

python3 examples/swebench-pro/session_smoke.py \
  --instance-id <instance-id> \
  --image-name <swebench-pro-image> \
  --base-commit <base-commit>
```

This starts a session, runs a tiny `bash` tool call, submits, and closes the
session. If this fails, fix the environment plane before launching RL.

## Build A Trace-Replay Workload

For deterministic router benchmarks, derive the replay workload from a Dynamo
agent trace and render the HTML shape check:

```bash
uv run examples/swebench-pro/trace_replay_workload.py \
  --source /data/swebench-pro/traces/dynamo-agent-trace.<id>.jsonl.gz \
  --target-trajectories 400 \
  --max-generated-tokens-per-turn 8192 \
  --max-tool-duration-s 120 \
  --output-jsonl /data/swebench-pro/traces/trace-replay-sample-qwen35-v1-8192gen-tool120-$(date -u +%Y%m%d).jsonl \
  --output-html .context/traces/trace-replay-workload-$(date -u +%Y%m%d).html \
  --summary-json .context/traces/trace-replay-workload-$(date -u +%Y%m%d).json
```

Inspect the HTML before using the JSONL. The intended shape is one lane per
sample trajectory, with repeated model calls and tool sleeps in that same
trajectory.

## Launch With A YAML Config

Prefer the reproducible two-file interface for new experiments:

```bash
python3 examples/swebench-pro/run_experiment.py \
  --cluster examples/swebench-pro/reproducible/cluster.yaml \
  --experiment examples/swebench-pro/reproducible/experiment_config.yaml \
  --dry-run
```

Use `--mode perf-test` with the same two files for the trace-replay performance
test path. See `examples/swebench-pro/reproducible/README.md` for the full
workflow.

Use `examples/swebench-pro/router_bench_trace_replay.yaml` as the baseline
config for mocked-trainer/router runs. Explicit environment variables override
YAML values, so keep run identifiers and one-off router flags in the shell:

```bash
cd /code/slime

env \
  SWEPRO_RUN_ID=routerbench_example_$(date -u +%Y%m%dT%H%M%SZ) \
  SWEPRO_RUN_CONFIG=examples/swebench-pro/router_bench_trace_replay.yaml \
  SWEPRO_TRACE_REPLAY_PATH=/data/swebench-pro/traces/trace-replay-sample-qwen35-v1-8192gen-tool120-20260605.jsonl \
  SWEPRO_DYNAMO_ROUTER_KV_EVENTS=1 \
  SWEPRO_DYNAMO_ROUTER_PREDICTED_TTL_SECS=10 \
  python3 examples/swebench-pro/launch_swepro_rl.py --repo-root /code/slime -- \
    --recompute-loss-function
```

Router-mode overrides used for the June 2026 ablations:

```bash
# KV-event routing.
SWEPRO_DYNAMO_ROUTER_KV_EVENTS=1
SWEPRO_DYNAMO_ROUTER_PREDICTED_TTL_SECS=10
SWEPRO_ROUTER_PREFILL_LOAD_SCALE=16

# Approximate routing without KV events.
SWEPRO_DYNAMO_ROUTER_KV_EVENTS=0
SWEPRO_ROUTER_PREFILL_LOAD_SCALE=1

# Sticky-style approximate routing on the router-load-weight branch.
SWEPRO_DYNAMO_ROUTER_KV_EVENTS=0
SWEPRO_DYNAMO_ROUTER_LOAD_WEIGHT=0.02
```

The engine and stack manifests now pass `--stream-output` to SGLang. Keep that
flag enabled for Dynamo/SGLang v1 completions compatibility; without it, token
ID streaming semantics can drift and trace-replay requests may accumulate
incorrect output tokens.

For real SWE-agent rollouts, switch `rollout.function_path` back to the normal
SWE-Pro rollout function, disable `mock_trainer`, and use the real trainer shape
for the current experiment.

## Collect Artifacts

Every run should leave a packet under `/data/swebench-pro/runs/<run-id>` with:

- `env.json`
- `ray-driver.log`
- `ray-status.log`
- `metrics/` with Tachometer config and parquet output when enabled
- pod logs and Dynamo agent traces when the launcher copied them

For committed perf summaries, keep raw logs under the remote run directory or an
ignored `.context/` artifact folder, then add a compact README and
`metrics.generated.json` under `qwen3.5-122b-perf-results/<date-label>/`.
