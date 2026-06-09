# RouterBench Trace Replay: KV Events vs Approx

This compares two successful full trace-replay runs on `dynamo-gcp-dev-02`, using the same 160 source trajectories from:

`/data/swebench-pro/traces/trace-replay-sample-qwen35-v1-8192gen-tool120-20260605.jsonl`

## Result

Approximate routing was faster on this trace workload.

| Metric | KV events | Approx | Delta |
|---|---:|---:|---:|
| Successful trajectories | 160 | 160 | same |
| Model requests | 3,085 | 3,085 | same |
| Trainable/generated tokens | 979,819 | 979,819 | same |
| Rollout response tokens | 4,185,489 | 4,185,489 | same |
| Event span | 686s | 655s | approx -31s |
| Mean trajectory e2e | 281.9s | 251.3s | approx -30.6s |
| p50 trajectory e2e | 258.0s | 223.5s | approx -34.5s |
| p90 trajectory e2e | 462.6s | 420.8s | approx -41.8s |
| p95 trajectory e2e | 503.5s | 461.6s | approx -41.9s |

Paired by `trace_replay_source_trajectory_id`, approx was faster for `158/160` trajectories, tied for `2/160`, and slower for `0/160`. Median approx/KV ratio was `0.893`, so the median trajectory was about 10.7% faster.

## Run Shape

Both runs used:

- `num_rollout=20`, `rollout_batch_size=8`, `over_sampling_batch_size=32`
- `max_started_groups=160`, `max_inflight=128`
- mock trainer at `15000 tok/s`, mock weight update at `20s`
- 4 Dynamo/SGLang engines, 2 GPUs per engine, `SGLANG_MAX_RUNNING_REQUESTS=128`
- TCP request plane
- fixed trace decode: `ignore_eos=true`, requested OSL from trace

Routing differed as follows:

- KV events: frontend used KV-event routing; engines published KV events.
- Approx: frontend used `--no-router-kv-events` with `router-prefill-load-scale=1`; engines were relaunched with `SWEPRO_DYNAMO_ROUTER_KV_EVENTS=0`.

## Supporting Signals

Tachometer supports the log-level result, but trajectory logs are the primary evidence because they cleanly pair the same source traces.

| Signal | KV events | Approx |
|---|---:|---:|
| SGLang prefill compute token delta | 4,463,616 | 3,060,992 |
| SGLang prefill cache token delta | 9,780,224 | 9,557,184 |
| Engine cache hit sample mean | 0.084 | 0.128 |
| Engine cache hit sample p90 | 0.303 | 0.812 |
| Engine queue req max | 6 | 28 |
| Engine retracted req max | 0 | 0 |
| Frontend TTFT p50 | 1.37s | 1.46s |
| Frontend TTFT p90 | 3.61s | 3.69s |
| SGLang queue time p90 | 0.70s | 1.33s |

Approx saw more transient queueing, but still completed faster overall. The likely explanation from these counters is that approximate routing produced less prefill compute and higher observed engine cache-hit samples on this workload.

## Artifacts

KV events:

- Remote run dir: `/data/swebench-pro/runs/routerbench_kvevents_trace_replay_streamfix2_20260606T0514Z`
- Tachometer parquet: `/data/swebench-pro/runs/routerbench_kvevents_trace_replay_streamfix2_20260606T0514Z/metrics/tachometer-local/final.parquet`

Approx:

- Remote run dir: `/data/swebench-pro/runs/routerbench_approx_trace_replay_streamfix_20260606T0555Z`
- Tachometer parquet: `/data/swebench-pro/runs/routerbench_approx_trace_replay_streamfix_20260606T0555Z/metrics/tachometer-local/final.parquet`

## Caveats

- The primary comparison uses `SWEPRO_AGENT_TRACE` timestamps from `ray-driver.log`, grouped by `trace_replay_source_trajectory_id`.
- The KV Tachometer scraper was manually stopped after the run because the old watcher missed Ray's `Job ... succeeded` wording. Do not use collector wall-clock based rates from that run as primary evidence.
- Frontend logs are cumulative across restarts. The relevant evidence is the startup tail: KV had a `kv-events` subscriber; approx logged `Skipping KV event subscription (use_kv_events=false, overlap_score_credit=1)`.
- Full raw logs were left in the remote run dirs rather than copied locally because each frontend log is hundreds of MB.
