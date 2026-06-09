# RouterBench Router Iteration: 2026-06-08

This packet records the 400-trajectory trace-replay router sweep on
`dynamo-gcp-dev-02`. The workload used the same replay trace shape for every
run:

```text
/data/swebench-pro/traces/trace-replay-sample-qwen35-v1-8192gen-tool120-20260605.jsonl
```

## Run Shape

- 400 completed trajectories.
- 7,656 model requests and responses.
- 2,474,472 generated/trainable tokens.
- Mock trainer at 15,000 tok/s.
- Mock weight update time: 20s.
- 4 Dynamo/SGLang engines, 2 GPUs per engine.
- `ignore_eos=true` with per-turn OSL from the replay trace.
- SGLang/Dynamo runtime image family:
  `aphoh/slime:swepro-sglang0512-main-b5ff7e2-*`.

## Primary Result

The best KV-event configuration found so far beats the sticky-session baseline
on all three user-facing metrics:

| Config | Traj/min | Mean e2e | P90 e2e | Result |
|---|---:|---:|---:|---|
| Approx/no KV events, load=0.02 | 21.0342 | 244.0675s | 447.2s | Sticky baseline |
| KV events, TTL=10, prefill scale=8, load=1 | 21.0711 | 255.255s | 461.3s | Throughput only |
| KV events, TTL=10, prefill scale=16, load=1 | 21.5633 | 245.880s | 442.0s | Misses mean |
| KV events, TTL=10, prefill scale=16, load=0.1 | 21.9579 | 244.935s | 444.0s | Misses mean by 0.8675s |
| KV events, TTL=10, prefill scale=16, load=0.05 | 21.6216 | 249.090s | 458.0s | Regressed |
| KV events, TTL=10, prefill scale=24, load=0.1 | 20.6186 | 249.675s | 466.0s | Regressed |
| KV events, TTL=5, prefill scale=16, load=0.1 | partial | partial | partial | Stopped at 288/400 |
| **KV events, TTL=10, prefill scale=16, load=0.15** | **21.6411** | **241.400s** | **439.0s** | **Best completed run** |

Relative to sticky load=0.02, the winning run improves:

- Trajectory throughput: +3.95%.
- Mean e2e latency: -1.09%.
- P90 e2e latency: -1.83%.

## Why The Winner Helped

The earlier KV-event baseline was not failing because events were missing:
published and frontend-applied event counts matched exactly, and invalid,
dropped, duplicate, and rejected event counters were zero. The problem was the
policy tradeoff: with load weight 1.0, the router often chose lower active load
over growing per-trajectory prompt locality.

The winning `load=0.15` run kept most of the locality benefit while improving
balance. Compared with `load=0.1`, it had nearly identical actual cache reuse,
processed more requests in the same early active window, lowered mean queue
depth, and tightened per-engine token-usage spread.

## Raw Artifact Locations

The raw Tachometer parquet, Ray logs, and helper scripts are intentionally kept
out of git because they are large. On this workstation they were organized under:

```text
.context/routerbench-kv-vs-sticky-tachometer-20260608/
.context/routerbench-router-iteration-20260608/
```

The compact ledgers are:

- `.context/routerbench-router-iteration-20260608/results.csv`
- `.context/routerbench-router-iteration-20260608/*.json`
- `.context/routerbench-kv-vs-sticky-tachometer-20260608/summaries/*.json`
- `.context/routerbench-kv-vs-sticky-tachometer-20260608/summaries/*.csv`

The top-level `.context` artifact directories are ignored; use this result
packet for committed inspection and the remote `/data/swebench-pro/runs/*`
directories for raw data.
