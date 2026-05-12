# SWE-bench Pro Qwen3.5 131k Trainer Handoff

This is a handoff from the GCP dev-02 bringup work for SWE-bench Pro RL with Qwen3.5 122B-A10B. The main state lives in the conductor workspace branch `warnold/slynamo-2.0`; Kubernetes/example YAML changes should be pulled into this checkout after the branch is pushed.

## Cluster And Namespace

Use only this namespace/context for the work described here:

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
NS=warnold-dynamo
kubectl --context="$CTX" -n "$NS" get pods
```

Important resources used by the SWE-bench Pro stack:

- Namespace: `warnold-dynamo`
- Ray head / trainer pod: `warnold-swepro-trainer`
- Ray worker deployment: `warnold-swepro-trainer-worker`, 3 replicas for a 16 GPU trainer when combined with the head
- Session workers: `warnold-swepro-session` StatefulSet, 8 replicas
- Dynamo frontend service: `warnold-swepro-frontend:3000`
- NATS: `warnold-swepro-nats:4222`
- Model/cache PVCs:
  - `/shared` from `shared-model-cache`: shared model cache, should be disturbed minimally
  - `/data` from `model-cache`: run logs, debug rollout data, traces, writable working data
  - `/code` from `code-cache`: uploaded code snapshot used by cluster pods

## Current Qwen3.5 Trainer Status

Goal tested: forced trainer run with mocked rollout data, two 131072-token samples, Qwen3.5 122B-A10B, 16 GPUs, no rollout/inference dependency.

Synthetic rollout data:

```bash
/data/swebench-pro/debug/pp16_131k_smoke/rollout_0.pt
```

Data shape:

- `samples=2`
- `total_length=131072` per sample
- `response_length=8192` per sample
- includes rollout logprobs

Useful launch pattern, run inside `warnold-swepro-trainer`:

```bash
cd /code/slime
env \
  SLIME_SKIP_WEIGHT_UPDATES=1 \
  SWEPRO_RUN_ID=pp16_cp1_tp1_ep1_131k_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
  SWEPRO_ACTOR_NUM_NODES=4 \
  SWEPRO_ACTOR_NUM_GPUS_PER_NODE=4 \
  SWEPRO_MODEL_ARGS_SCRIPT=scripts/models/qwen3.5-122B-A10B.sh \
  SWEPRO_HF_CHECKPOINT=/shared/Qwen3.5-122B-A10B \
  SWEPRO_REF_LOAD=/shared/Qwen3.5-122B-A10B_torch_dist \
  SWEPRO_LOAD_DEBUG_ROLLOUT_DATA='/data/swebench-pro/debug/pp16_131k_smoke/rollout_{rollout_id}.pt' \
  SWEPRO_NUM_ROLLOUT=1 \
  SWEPRO_ROLLOUT_BATCH_SIZE=2 \
  SWEPRO_OVER_SAMPLING_BATCH_SIZE=2 \
  SWEPRO_N_SAMPLES_PER_PROMPT=1 \
  SWEPRO_GLOBAL_BATCH_SIZE=2 \
  SWEPRO_TP=1 \
  SWEPRO_PP=16 \
  SWEPRO_CP=1 \
  SWEPRO_EP=1 \
  SWEPRO_ETP=1 \
  SWEPRO_SEQUENCE_PARALLEL=0 \
  SWEPRO_QKV_FORMAT=thd \
  SWEPRO_USE_DYNAMIC_BATCH_SIZE=1 \
  SWEPRO_MAX_TOKENS_PER_GPU=131072 \
  SWEPRO_SEQ_LENGTH=131072 \
  SWEPRO_MAX_CONTEXT_LEN=131072 \
  SWEPRO_MAX_RESPONSE_LEN=131072 \
  SWEPRO_LOG_PROBS_CHUNK_SIZE=16 \
  SWEPRO_DEFER_FP32_LOGITS=1 \
  SWEPRO_MOE_TOKEN_DISPATCHER_TYPE=alltoall \
  SWEPRO_DISABLE_SAVE=1 \
  python3 examples/swebench-pro/launch_swepro_rl.py --repo-root /code/slime -- \
    --recompute-loss-function
```

Ray status/log commands:

```bash
ray job status <RUN_ID> --address=http://127.0.0.1:8265
ray job logs <RUN_ID> --address=http://127.0.0.1:8265 | tail -300
ls -lh /data/swebench-pro/runs/<RUN_ID>
tail -200 /data/swebench-pro/runs/<RUN_ID>/ray-driver.log
```

GPU memory snapshots:

```bash
kubectl --context="$CTX" -n "$NS" exec warnold-swepro-trainer -- \
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
kubectl --context="$CTX" -n "$NS" exec deploy/warnold-swepro-trainer-worker -- \
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## Findings From 131k Smoke Attempts

### 1. CP-heavy layout failed in GDN/FLA

Earlier full 131k trainer attempt used roughly `TP=2, PP=1, CP=8, EP=16` and failed in FLA gated delta rule:

```text
torch.OutOfMemoryError: Tried to allocate 1.47 GiB
/usr/local/lib/python3.12/dist-packages/fla/ops/common/chunk_scaled_dot_kkt.py
A = torch.empty(B, T, H, BT, ...)
```

Interpretation: context parallelism did not save the GDN activation path for Qwen3.5. For this model family, CP is not a reliable first lever for 131k training.

### 2. PP16 fixes the GDN/log_probs phase but exposes last-stage logits memory

PP16 config:

```text
TP=1, PP=16, CP=1, EP=1, ETP=1
```

This got through logprob forward and checkpoint load. Initial timings:

- `log_probs`: 93.2s before bf16 output patch
- `actor_train`: reached, then failed

Failure before patch:

```text
torch.OutOfMemoryError: Tried to allocate 121.25 GiB
/root/src/Megatron-LM/megatron/core/transformer/module.py float16_to_fp32
```

Root cause: Slime called the Megatron model with default `fp32_output=True`, causing the last pipeline stage to materialize fp32 logits with shape approximately `131072 x 248320`. Megatron RL utilities use `fp32_output=not (args.fp16 or args.bf16)`; Slime should do the same. The 397B config's `defer_fp32_logits: true` is the same class of fix.

Local patch tested in the conductor workspace:

- `slime/backends/megatron_utils/model.py`
  - pass `fp32_output=not (args.fp16 or args.bf16)` in both logprob and train forward calls
- `slime/backends/megatron_utils/loss.py`
  - allow bf16 logits in dtype assertions

After patch, `log_probs` dropped to roughly 14s.

### 3. Entropy should not be computed when `entropy_coef == 0`

After bf16 logits, `actor_train` still tried to allocate entropy/logprob scratch buffers and failed with only a few MiB free:

```text
calculate_log_probs_and_entropy
entropy_input = logits_chunk.clone()
torch.OutOfMemoryError: Tried to allocate 62.00 MiB
```

Local patch tested:

- In `policy_loss_function`, call `get_log_probs_and_entropy(..., with_entropy=args.entropy_coef != 0)`.
- If entropy is disabled, set `entropy_loss = logits.new_zeros(())`.

### 4. Chunk size 16 plus entropy skip still failed on compiled logprob scratch

Run id:

```text
pp16_cp1_tp1_ep1_131k_bf16_chunk16_20260511T051059Z
```

Failure:

```text
torchinductor generated empty_strided_cuda((16, 1, 248320), ..., torch.float32)
torch.OutOfMemoryError: Tried to allocate 16.00 MiB
```

This means the last pipeline stage was essentially full during backward recomputation. Lowering `SWEPRO_LOG_PROBS_CHUNK_SIZE` below 16 may avoid this exact allocation, but it is probably not the right long-term answer.

### 5. Precision-aware CPU optimizer offload did not fix the scratch-buffer OOM

Run id:

```text
pp16_cp1_tp1_ep1_131k_cpuoffload_chunk16_20260511T052127Z
```

Extra args:

```bash
--use-precision-aware-optimizer \
--optimizer-cpu-offload \
--overlap-cpu-optimizer-d2h-h2d \
--exp-avg-dtype bf16 \
--exp-avg-sq-dtype bf16
```

The args were accepted:

```text
optimizer_cpu_offload=True
use_precision_aware_optimizer=True
exp_avg_dtype=torch.bfloat16
exp_avg_sq_dtype=torch.bfloat16
```

But failure remained the same 16 MiB compiled logprob scratch allocation. This suggests the limiting memory is not Adam state; it is last-stage logits/loss/backward memory.

## Interpreting The 397B Example

The example config:

```yaml
megatron_cfg:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 8
  num_layers_in_first_pipeline_stage: 6
  num_layers_in_last_pipeline_stage: 6
  expert_model_parallel_size: 32
  activation_checkpointing: true
  sequence_parallel: true
  moe_token_dispatcher_type: allgather
  apply_rope_fusion: false
  defer_fp32_logits: true
```

Useful pieces for our 122B work:

- `defer_fp32_logits: true`: definitely relevant. Our manual `fp32_output=False` patch is the local equivalent, but a cleaner upstream-aligned flag should be added.
- Non-uniform PP split: likely relevant. The last stage has embeddings/output head pressure; reducing last-stage transformer layers may help. For 48 layers and PP16, try first/last stages with fewer layers if Megatron supports `--num-layers-in-first-pipeline-stage` / `--num-layers-in-last-pipeline-stage` or equivalent in this fork.
- More EP: likely relevant for expert weights and MoE memory. For Qwen3.5-122B, try `EP=16` or `EP=32` if world size allows. With 16 GPUs, valid options need to respect `TP * PP * CP * DP` and EP constraints. `EP=16` with `PP=16`, `TP=1`, `CP=1` should be checked against Megatron's process group rules before assuming it works.
- Activation checkpointing: already using `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`; keep this.
- `sequence_parallel: true`: only works with TP > 1. It was not used in the PP16 smoke because TP was 1. Since Qwen3.5/GDN did not like CP, TP/sequence parallel may still be useful but will require a different PP/TP/EP layout.
- `moe_token_dispatcher_type: allgather`: the Qwen3.5 model args script currently sets alltoall. The launcher default used to override this with allgather; for variable-length/dynamic batches Megatron warned that allgather does not support variable sequence length and recommended alltoall. Be careful copying `allgather` into our dynamic packed path.

## Recommended Next Trainer Experiments

1. Make `defer_fp32_logits` a real Slime option.
   - Add an argument similar to `--defer-fp32-logits`.
   - Use it to pass `fp32_output=False` in Megatron forward calls when training bf16/fp16.
   - Keep logprob/loss code compatible with bf16 logits.

2. Avoid last-stage pressure rather than shrinking logprob chunk forever.
   - Try PP16 with fewer layers on the last stage if the fork supports it.
   - Search flags:

```bash
kubectl --context="$CTX" -n "$NS" exec warnold-swepro-trainer -- \
  bash -lc 'python3 train_async.py --help | grep -E "first.*pipeline|last.*pipeline|pipeline.*layers|defer.*logits"'
```

3. Try higher EP with PP16.
   - Candidate smoke: `TP=1, PP=16, CP=1, EP=16, ETP=1`, dynamic batch size on, same 131k debug rollout.
   - If EP16 is rejected by process group validation, try `TP=2, PP=8, CP=1, EP=16` with sequence parallel on. This gives fewer PP stages but may cut output/logit shard size through TP.

4. If last-stage output logits remain the bottleneck, test TP on the output head.
   - `TP=2, PP=8, CP=1, EP=16`, `--sequence-parallel`.
   - This may be the most direct way to halve vocab-parallel logits per rank.

5. Do not use CP as the first answer for Qwen3.5 long context.
   - CP8 previously failed in the GDN/FLA kernel before reaching the same last-stage logprob bottleneck.

## Qwen3.5 Inference Worker Notes

Standalone engine YAMLs were created for Qwen3.5 122B FP8 rollout smoke:

- `examples/swebench-pro/k8s-gcp02-qwen35-engine-bf16-smoke.yaml`
- `examples/swebench-pro/k8s-gcp02-qwen35-engines-bf16-big.yaml`

The working inference shape was TP2 per engine, model path `/shared/Qwen3.5-122B-A10B`, with:

```bash
python3 -m dynamo.sglang \
  --model-path /shared/Qwen3.5-122B-A10B \
  --tp 2 \
  --trust-remote-code \
  --enable-rl \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557","enable_kv_cache_events":true}' \
  --page-size 64 \
  --attention-backend trtllm_mha \
  --moe-runner-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-scheduler-strategy extra_buffer \
  --mamba-track-interval 256 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 32 \
  --context-length 131072 \
  --max-prefill-tokens 16384 \
  --chunked-prefill-size 16384 \
  --stream-interval 50 \
  --scheduler-recv-interval 30 \
  --tokenizer-worker-num 1 \
  --skip-server-warmup \
  --mem-fraction-static 0.75
```

For GLM-4.7, the stack template should keep the `triton` attention path; for Qwen3.5, use the `qwen35` engine profile in the updated stack YAML. The validated Qwen3.5 BF16 path uses the Triton MoE runner by default; the FlashInfer TRT-LLM path needs the inference-side restore/repack hook before it should be treated as the default.

## Kubernetes Files To Pull From The Branch

Committed/pushed Kubernetes/example artifacts should include:

- `examples/swebench-pro/k8s-gcp02-swepro-stack.yaml`
  - parameterized frontend model path / KV block size
  - mounts `/shared`
  - StatefulSet session workers with persistent 500Gi Docker graph PVCs
  - trainer worker Deployment instead of fixed worker pod
  - Qwen3.5 engine profile support
- `examples/swebench-pro/k8s-gcp02-qwen35-engine-bf16-smoke.yaml`
  - one TP2 Qwen3.5 engine, lower concurrency
- `examples/swebench-pro/k8s-gcp02-qwen35-engines-bf16-big.yaml`
  - two TP2 Qwen3.5 engines, higher concurrency
- `examples/swebench-pro/k8s-ray-object-store-bench.yaml`
  - Ray object-store benchmark cluster for CPU nodes

Apply stack carefully; it uses real `warnold-*` names in `warnold-dynamo`.

## Useful Cleanup Commands

List warnold pods:

```bash
kubectl --context="$CTX" -n "$NS" get pods -o wide | grep warnold
```

Scale session workers down/up:

```bash
kubectl --context="$CTX" -n "$NS" scale statefulset/warnold-swepro-session --replicas=0
kubectl --context="$CTX" -n "$NS" scale statefulset/warnold-swepro-session --replicas=8
```

Scale trainer workers:

```bash
kubectl --context="$CTX" -n "$NS" scale deployment/warnold-swepro-trainer-worker --replicas=0
kubectl --context="$CTX" -n "$NS" scale deployment/warnold-swepro-trainer-worker --replicas=3
```

Check Ray jobs:

```bash
kubectl --context="$CTX" -n "$NS" exec warnold-swepro-trainer -- \
  ray job list --address=http://127.0.0.1:8265
```

Stop a Ray job:

```bash
kubectl --context="$CTX" -n "$NS" exec warnold-swepro-trainer -- \
  ray job stop <RUN_ID> --address=http://127.0.0.1:8265
```

## Caution

The local code patches in the conductor workspace are experimental and not all committed. The most important trainer patch concept is `defer_fp32_logits`; implement that cleanly before relying on full 131k training. The Kubernetes YAML commit is safe as infrastructure handoff, but the 131k Qwen3.5 trainer itself was not yet made to pass.
