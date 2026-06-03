# Qwen3.5 122B SWE-Pro RL Perf: 2026-05-25/26 First Result

## Run

- Run id: `qwen35_swepro_rl32_rollout8gpus_gbs8_inflight32_131k_unclipped_tp8cp4ep32_dyn86268c59_20260525T074115Z`
- Result source: `/data/swebench-pro/runs/qwen35_swepro_rl32_rollout8gpus_gbs8_inflight32_131k_unclipped_tp8cp4ep32_dyn86268c59_20260525T074115Z`
- Runtime: 50.91 min from run id timestamp to Ray success; 48.10 min from first session start to last rollout finish.
- Ray status: succeeded at `2026-05-25T08:32:09.879Z`.
- Trainer shape: 32 GPUs, `TP=8 CP=4 EP=32 PP=1`, microbatch 1, GBS 8.
- Rollout shape: 8 GPUs total, 4 Dynamo/SGLang engines, 2 GPUs per engine, max in-flight 32, return batch size 8.
- Length config: `seq_length=max_context=max_response=131072`; `SWEPRO_MAX_TOOL_CALLS=0` and `SWEPRO_TURN_MAX_TOKENS=0`.
- Optimizer config: precision-aware optimizer on, CPU optimizer offload on, CPU optimizer D2H/H2D overlap on.

## Throughput

- Finished rollouts: 187.
- Trained samples: 64 expected from 8 trainer batches x GBS 8.
- Rollouts/minute: 3.67 over total job runtime; 3.89 over active rollout window.
- Trained samples/minute: 1.26 over total job runtime.
- Model-generated tokens: 10,780,631 from `model_response.generated_tokens`.
- Model generation rate: 3,529 tok/s over total job runtime; 3,735 tok/s over active rollout window; 467 tok/s per rollout GPU over active rollout window.
- Rollout response tokens: 16,052,114 from `rollout_finished.response_tokens`.
- Rollout trainable tokens: 10,612,586 from `rollout_finished.trainable_tokens`.
- Response token distribution: min 32, median 93,053, mean 85,840, p90 129,335, max 129,619.
- Trainable token distribution: min 32, median 58,082, mean 56,752, p90 102,039, max 121,214.

## Trainer Timing

| step | step time s | train time s | wait time s | actor tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1002.3 | 466.4 | 535.9 | 1788.8 |
| 1 | 218.1 | 217.8 | 0.3 | 4475.4 |
| 2 | 298.1 | 278.9 | 19.2 | 3625.2 |
| 3 | 233.7 | 230.5 | 3.2 | 3485.2 |
| 4 | 306.0 | 283.2 | 22.8 | 2800.9 |
| 5 | 203.6 | 200.1 | 3.5 | 4312.6 |
| 6 | 398.9 | 381.1 | 17.8 | 1490.3 |
| 7 | 254.4 | 250.9 | 3.5 | 2480.8 |

- Mean train time: 288.6s; median train time: 264.9s.
- Mean step time: 364.4s including the first fill/wait-heavy step.
- Mean trainer wait: 75.8s including first fill; 10.1s excluding the first step.
- Mean actor train throughput: 3,057 tok/s across all steps; 3,239 tok/s excluding the first step.

## Weight Update

Five weight updates were logged: one initial connection/update, then four steady updates.

| update | total s | sync s | non-expert sync s | expert sync s | engines | new engines |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 33.787 | 21.567 | 3.338 | 18.229 | 4 | 4 |
| 1 | 15.758 | 13.662 | 1.975 | 11.687 | 4 | 0 |
| 2 | 19.362 | 13.276 | 1.723 | 11.553 | 4 | 0 |
| 3 | 14.271 | 13.211 | 1.682 | 11.529 | 4 | 0 |
| 4 | 15.454 | 14.382 | 2.883 | 11.499 | 4 | 0 |

- Steady-state update mean: 16.21s total, 13.63s sync.
- NCCL log evidence showed `NET/IB`, `NET/IBext_v11`, and GDRDMA/DMABUF paths, with `NCCL_IB_HCA=mlx5_0`, `NCCL_IB_GID_INDEX=3`, and `NCCL_CROSS_NIC=0`.
- Weight-update-specific env used the same HCA/GID/CROSS_NIC/PXN settings and `SLIME_WEIGHT_UPDATE_DIRECT_EXPERTS=1`.

## KV Routing And Cache

- KV routing was active: retained frontend logs include 2,861 `kv_router.select_worker` / `Selected worker` lines for this run.
- The retained frontend log starts at `2026-05-25T08:08:56Z`, so it does not cover the whole run.
- Copied Dynamo trace: `raw/dynamo-agent-trace.000010.jsonl.gz`.
- Dynamo `request_end` counters for this run recorded 5,980 inference requests:
  - total prompt/input tokens: 195,360,152
  - total cached tokens: 0
  - token-weighted prefix cache hit rate: 0.0
  - unweighted `kv_hit_rate`: mean 0.0, p95 0.0, max 0.0
  - requests with nonzero `cached_tokens`: 0
- Per-prefill-worker trace counters were also all zero: workers handled 1,420 / 1,477 / 1,489 / 1,594 requests respectively, with zero cached tokens on each.
- Logged rollout prefix-cache metrics were zero for all eight returned train batches:
  - average `rollout/prefix_cache_hit_rate`: 0.0
  - average `rollout/avg_cached_tokens_per_sample`: 0.0
- Caveat: these zero cache metrics now agree with Dynamo's explicit request-end counters. If SGLang internally reused prefix work, that reuse was not reflected in the `cached_tokens` / `kv_hit_rate` fields Dynamo emitted for this run.

## Rollout Behavior

- Rollout finish reasons: 114 `stop`, 73 `length`.
- Rollout statuses: 114 completed, 73 truncated.
- Model responses: 5,966 total; 5,901 stop, 65 length.
- Model stop reasons: 5,803 `</tool_call>`, 163 `None`.
- Tool calls requested: 5,799 total: 4,122 `bash`, 1,663 `str_replace_editor`, 13 `submit`, 1 `bell`.
- Submit responses: 174.
- Patch chars across finished rollouts: median 1,643, mean 8,970, p90 22,485, max 243,384.

## Artifacts

- `metrics.generated.json`: parsed metrics from the copied logs.
- `raw/ray-driver.log`: Ray driver log copied from the run directory.
- `raw/env.json`: run environment copied from the run directory.
- `raw/ray-status.log`: Ray status log copied from the run directory.
- `raw/frontend.log`: retained frontend log from the active frontend pod.
- `raw/dynamo-agent-trace.000010.jsonl.gz`: Dynamo JSONL trace copied from the eval pod.
