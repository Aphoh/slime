# Successful Slow SWE-Pro Run

Date recorded: 2026-05-12

This documents the first Qwen3.5-122B SWE-bench Pro RL run that completed end to end on GCP02. It was successful as a systems run, but slow. The key caveat is that the Slime online weight update was intentionally forced onto Socket/TCP for this run, so the repeated weight updates took about 79 seconds each.

## Source Of Truth

Cluster:

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
NS=warnold-dynamo
```

Run id:

```text
qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z
```

Durable log directory on the trainer pod:

```text
/data/swebench-pro/runs/qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z
```

Important files:

```text
/data/swebench-pro/runs/qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z/env.json
/data/swebench-pro/runs/qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z/ray-submit.log
/data/swebench-pro/runs/qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z/ray-status.log
/data/swebench-pro/runs/qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z/ray-driver.log
```

Ray status:

```text
submitted: 2026-05-12T08:39:58Z
succeeded: 2026-05-12T10:56:51Z
driver node: 10.0.0.28
driver pid: 6028
driver exit code: 0
```

## Live Stack Used

Trainer:

```text
warnold-swepro-trainer + 3 Ray worker pods
4 nodes x 4 GPUs = 16 trainer GPUs
```

Inference:

```text
warnold-swepro-engine-0, 2 GPUs
warnold-swepro-engine-1, 2 GPUs
2 engines x 2 GPUs = 4 rollout GPUs
```

Other services:

```text
warnold-swepro-frontend:3000
warnold-swepro-nats:4222
warnold-swepro-session-0..7
warnold-swepro-eval-0..1
```

Note: after this run, an interrupted later cleanup command deleted `warnold-swepro-engine-0` and `warnold-swepro-engine-1`. That deletion is not part of the successful run.

## Exact Launch Shape

The run was submitted from inside `warnold-swepro-trainer` from `/code/slime` through:

```bash
python3 examples/swebench-pro/launch_swepro_rl.py --repo-root /code/slime -- --recompute-loss-function
```

The effective environment from `env.json` was:

```bash
export SWEPRO_RUN_ID=qwen35_swepro_large_gbs8_inflight32_tp2pp8ep2_2engine_triton_bf16_cpuoffload_socketwu_serialinit_20260512T083856Z
export SWEPRO_ACTOR_NUM_NODES=4
export SWEPRO_ACTOR_NUM_GPUS_PER_NODE=4
export SWEPRO_MODEL_ARGS_SCRIPT=scripts/models/qwen3.5-122B-A10B.sh
export SWEPRO_HF_CHECKPOINT=/shared/Qwen3.5-122B-A10B
export SWEPRO_REF_LOAD=/shared/Qwen3.5-122B-A10B_torch_dist
export SWEPRO_PROMPT_DATA=/data/swebench-pro/swebench_pro_train_cached_images.jsonl
export SWEPRO_MODEL=/shared/Qwen3.5-122B-A10B

export SWEPRO_NUM_ROLLOUT=20
export SWEPRO_ROLLOUT_BATCH_SIZE=8
export SWEPRO_OVER_SAMPLING_BATCH_SIZE=32
export SWEPRO_N_SAMPLES_PER_PROMPT=1
export SWEPRO_GLOBAL_BATCH_SIZE=8
export SWEPRO_ASYNC_MAX_INFLIGHT=32
export SWEPRO_ASYNC_GROUP_MAX_ATTEMPTS=2

export SWEPRO_TP=2
export SWEPRO_PP=8
export SWEPRO_CP=1
export SWEPRO_EP=2
export SWEPRO_ETP=1
export SWEPRO_SEQUENCE_PARALLEL=1
export SWEPRO_QKV_FORMAT=thd
export SWEPRO_MOE_TOKEN_DISPATCHER_TYPE=alltoall

export SWEPRO_SEQ_LENGTH=131072
export SWEPRO_MAX_CONTEXT_LEN=131072
export SWEPRO_MAX_RESPONSE_LEN=131072
export SWEPRO_MAX_TOKENS_PER_GPU=131072
export SWEPRO_USE_DYNAMIC_BATCH_SIZE=1
export SWEPRO_LOG_PROBS_CHUNK_SIZE=64
export SWEPRO_DEFER_FP32_LOGITS=1

export SWEPRO_ROLLOUT_FUNCTION_PATH=fully_async_rollout.generate_rollout_fully_async
export SWEPRO_ROLLOUT_NUM_GPUS=4
export SWEPRO_ROLLOUT_NUM_GPUS_PER_ENGINE=2

export SWEPRO_DISABLE_SAVE=1
export SWEPRO_UPDATE_WEIGHTS_INTERVAL=2
export SWEPRO_USE_PRECISION_AWARE_OPTIMIZER=1
export SWEPRO_OPTIMIZER_CPU_OFFLOAD=1
export SWEPRO_OVERLAP_CPU_OPTIMIZER_D2H_H2D=1
export SWEPRO_EXP_AVG_DTYPE=bf16
export SWEPRO_EXP_AVG_SQ_DTYPE=bf16
```

Trainer transport environment passed by Ray:

```bash
export NCCL_NVLS_ENABLE=1
export NCCL_MNNVL_ENABLE=0
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_HOST_ENABLE=1
export NCCL_STORE_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NVIDIA_GDRCOPY=1
export UCX_TLS=cuda_ipc,cuda_copy,rc
export UCX_IB_GID_INDEX=3
export UCX_RC_TIMEOUT=600s
export UCX_KEEPALIVE_INTERVAL=300s
```

Weight-update environment that made this run complete, but slowly:

```bash
export SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET=1
export SLIME_WEIGHT_UPDATE_NCCL_IB_DISABLE=1
export SLIME_WEIGHT_UPDATE_NCCL_CROSS_NIC=0
export SLIME_WEIGHT_UPDATE_NCCL_IB_MERGE_NICS=0
export SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE=0
export SLIME_WEIGHT_UPDATE_NCCL_NET=Socket
export SLIME_WEIGHT_UPDATE_NCCL_SOCKET_IFNAME=eth0
export SLIME_WEIGHT_UPDATE_NCCL_DEBUG=INFO
export SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS=INIT,ENV,NET
```

The engine pods used for the successful run also had the Socket fallback shape in their entrypoint:

```bash
export NCCL_CROSS_NIC=0
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_MERGE_NICS=0
export NCCL_MNNVL_ENABLE=0
export NCCL_NET=Socket
```

This is different from the GCP02 runbook's preferred RoCE/GDRDMA setup. It was useful for proving the training loop can run end to end, but it is the likely reason weight updates were slow.

## Exact Ray Entrypoint

The exact Ray job entrypoint recorded in `ray-status.log` was:

```bash
python3 train_async.py --qkv-format thd --actor-num-nodes 4 --actor-num-gpus-per-node 4 --distributed-backend cpu:gloo,cuda:nccl --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec --disable-bias-linear --qk-layernorm --group-query-attention --num-attention-heads 32 --num-query-groups 2 --kv-channels 256 --num-layers 48 --hidden-size 3072 --ffn-hidden-size 1024 --use-gated-attention --normalization RMSNorm --apply-layernorm-1p --position-embedding-type rope --norm-epsilon 1e-6 --rotary-percent 0.25 --swiglu --untie-embeddings-and-output-weights --vocab-size 248320 --rotary-base 10000000 --moe-ffn-hidden-size 1024 --moe-shared-expert-intermediate-size 1024 --moe-router-score-function softmax --moe-token-dispatcher-type alltoall --moe-router-topk 8 --moe-layer-freq [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] --num-experts 256 --moe-grouped-gemm --moe-token-drop-policy probs --moe-router-dtype fp32 --moe-permute-fusion --moe-aux-loss-coeff 0.001 --attention-output-gate --moe-shared-expert-gate --seq-length 131072 --rollout-backend dynamo --rollout-function-path fully_async_rollout.generate_rollout_fully_async --dynamo-frontend-url http://warnold-swepro-frontend:3000 --dynamo-worker-system-port 30001 --dynamo-frontend-wait-timeout 300 --rollout-num-gpus 4 --rollout-num-gpus-per-engine 2 --sglang-server-concurrency 64 --hf-checkpoint /shared/Qwen3.5-122B-A10B --ref-load /shared/Qwen3.5-122B-A10B_torch_dist --prompt-data /data/swebench-pro/swebench_pro_train_cached_images.jsonl --input-key prompt --label-key instance_id --rollout-shuffle --num-rollout 20 --rollout-batch-size 8 --over-sampling-batch-size 32 --n-samples-per-prompt 1 --rollout-max-response-len 131072 --rollout-max-context-len 131072 --rollout-temperature 1 --global-batch-size 8 --balance-data --tensor-model-parallel-size 2 --sequence-parallel --expert-model-parallel-size 2 --expert-tensor-parallel-size 1 --pipeline-model-parallel-size 8 --context-parallel-size 1 --attention-backend flash --moe-token-dispatcher-type alltoall --use-dynamic-batch-size --max-tokens-per-gpu 131072 --log-probs-chunk-size 64 --defer-fp32-logits --disable-weights-backuper --advantage-estimator grpo --disable-rewards-normalization --update-weights-interval 2 --update-weight-buffer-size 536870912 --distributed-timeout-minutes 30 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --optimizer adam --lr 1e-6 --lr-decay-style constant --min-lr 1e-6 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --use-precision-aware-optimizer --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16 --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --custom-generate-function-path generate_with_swebench_pro.generate --custom-rm-path generate_with_swebench_pro.reward_func --recompute-loss-function
```

## What Completed

Observed counts from `ray-driver.log`:

```text
Final collected 8 samples: 20
WEIGHT UPDATE OUTER: 11
reward: 1.0 count: 0
reward: 0.0 count: 40
IBV_WC_RETRY_EXC_ERR count: 0
wrong type count: 0
OOM count: 0
```

The run completed 20 rollout batches and corresponding train metrics. It did not solve any tasks.

## Slow Pieces

Weight updates:

```text
11 updates total
each update total was about 78.3s to 79.3s
first connect path: get_engines=0.009s, connect=0.804s, sync=78.333s, total=79.150s
later updates: connect=0.000s, sync about 78s to 79s
num_engines=2
num_new=2 only on the first update, then 0
```

Trainer timings varied by sample length and cache/warm state. Representative metrics:

```text
perf 0:
  update_weights_time=79.15s
  log_probs_time=353.65s
  actor_train_time=722.39s
  train_time=1076.18s
  actor_train_tok_per_s=215.65

perf 1:
  log_probs_time=155.84s
  actor_train_time=257.61s
  train_time=413.57s
  actor_train_tok_per_s=552.57

perf 8:
  update_weights_time=79.30s
  log_probs_time=6.06s
  actor_train_time=33.72s
  train_time=39.91s
  actor_train_tok_per_s=4432.65

perf 16:
  update_weights_time=78.38s
  log_probs_time=104.46s
  actor_train_time=144.55s
  train_time=249.20s
  actor_train_tok_per_s=1233.13

perf 19:
  log_probs_time=103.17s
  actor_train_time=148.36s
  train_time=251.73s
  actor_train_tok_per_s=1015.00
```

## Important Interpretation

This run proved:

- Qwen3.5-122B can complete a 20-rollout SWE-Pro RL job on 16 trainer GPUs with 2 two-GPU inference engines.
- `--defer-fp32-logits` was needed for the 131k path.
- `--use-precision-aware-optimizer` plus `--optimizer-cpu-offload` was accepted and used.
- Flattened bucket online weight update worked repeatedly.
- Serializing initial engine weight-update group connections avoided the earlier `wrong type` failure.
- There was no `IBV_WC_RETRY_EXC_ERR`, no OOM, and no second-rollout crash.

This run did not prove:

- That RoCE/GDRDMA weight update is working for the mixed trainer+engine group.
- That the trainer is fast enough.
- That SWE-Pro reward quality is acceptable.

The next networking experiment should remove the Socket fallback and follow `examples/swebench-pro/gcp02-nccl-runbook.md`: keep `SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE=0`, leave `NCCL_NET` unset, leave `NCCL_IB_DISABLE` unset, and verify NCCL logs show the RDMA transport rather than Socket.
