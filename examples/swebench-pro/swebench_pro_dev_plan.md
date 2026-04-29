# SWE-bench Pro RL dev run — GLM-4.7-30B-A3B on 2× 8×B200 (nscale)

> **Audience**: a fresh agent who will execute this plan end-to-end. Assume nothing; follow the pointers. Every path below is real and checked in. Confirm cluster state with the listed `kubectl` commands before acting.

## Target

- **Model**: GLM-4.7-30B-A3B (30B total, 3B active, MoE with MLA). Model args: `scripts/models/glm4.7-30B-A3B.sh`. Conversion bridge: `slime/backends/megatron_utils/megatron_to_hf/glm4moe.py`. Existing code path exercises conversion — **do not re-implement**.
- **Task**: SWE-bench Pro (731 instances; ~50% JS/TS, ~36% Python, ~10% Go). Upstream: `~/proj/SWE-bench_Pro-os` on the dev host; canonical eval runner is `swe_bench_pro_eval.py` (Modal-based — we'll replace Modal with an in-cluster runner).
- **Agent scaffold**: SWE-agent (primary) or mini-swe-agent (fallback) with a small first-party completions model adapter that formats prompts locally and posts token IDs to slime's `/v1/completions` endpoint. **Do not use LiteLLM.** Use the saved SWE-agent `.traj` `history` or mini-swe-agent `messages` as the source of truth for training tokens.
- **Hardware**: 2× 8×B200 nscale nodes (~192 vCPU, ~2 TiB RAM each). 16 B200s total. x86_64. No separate CPU pool — eval workers steal spare CPU on the GPU nodes.
- **Scale**: ramp from 4–8 instances × 1 sample for smoke, then 16 instances × 1 sample, then 32 instances × 2 samples for the dev run. Keep eval concurrency low until CPU, Docker cache, and NCCL jitter are measured.

## Cluster and namespace

```
Context:    nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
Namespace:  warnold-dynamo
Node pool:  16 × g.192.b200.8  (8× B200, 192 vCPU, ~2 TiB RAM each, NVLink8/NVSwitch)
```

Switch context and verify before any apply:
```bash
kubectl config use-context nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
kubectl config set-context --current --namespace=warnold-dynamo
kubectl get nodes -o wide
kubectl get pods
```

Pick two nodes from the live free list immediately before launch. Do not trust stale candidate lists; other namespaces may consume GPUs at any time. Re-check with:
```bash
kubectl describe nodes | grep -E "Name:|nvidia.com/gpu" | grep -B1 "nvidia.com/gpu"
```

## Existing infrastructure to leverage

| Resource | What it gives you |
|---|---|
| Pod `workbench` | Long-lived dev pod, 4 GPU, mounts `model-cache` and `code-cache` PVCs. Use for `ray job submit` and for driving training. |
| Pod `sglang-bench` | Same shape as workbench; second driver if you want two concurrent experiments. |
| Pods `dynamo-worker-0`, `dynamo-worker-1` | Single-GPU dynamo workers already running qwen3-4b-sft; templates for the new worker pods (see `.context/k8s-dynamo-worker.yaml`). |
| Pod `docker-builder` | Prebuilt build environment for slime images. |
| PVC `model-cache` | Shared NFS, holds HF + mcore weights. Mount at `/data`. |
| PVC `code-cache` | Shared NFS, holds `slime` + `Megatron-LM` working trees. Mount at `/code`. |
| Image `aphoh/slime:dynamo-rl-support-v5` | Baseline slime+dynamo image used by all pods above. **Likely needs a rebuild to add SWE-agent deps** — check before P2. |

## Reference scripts and code (read these first)

All paths are relative to the repo root `/Users/warnold/conductor/workspaces/slime/tunis/` (and `/code/slime` inside the pods).

**Cluster YAML patterns** (`.context/`):
- `k8s-dynamo-worker.yaml` — pod template with `NODE_PLACEHOLDER` and `ETCD_HOST_PLACEHOLDER` substitution. Single-GPU worker; adapt for TP=4 by bumping `nvidia.com/gpu: 4`, `--tp 4`, and CPU/mem.
- `workbench-new.yaml` — sleep pod pattern with PVCs and GPU tolerations.
- `k8s-pvc.yaml`, `k8s-convert.yaml`, `k8s-train.yaml`, `k8s-docker-builder.yaml`, `k8s-download.yaml`, `k8s-sglang-bench.yaml` — other reference shapes.

**Launch patterns** (`.context/`):
- `launch_dynamo_external.sh` — **primary reference.** Full single-pod stack: pkill existing → start etcd + nats + dynamo.frontend → launch N workers (one per GPU) → `ray job submit` the trainer with `--rollout-backend dynamo --dynamo-frontend-url http://${POD_IP}:3000`. Mirror this topology but split workers across 2 pods on 2 nodes.
- `workbench_run_slime_retool_kvr.sh` — shows multi-pod pattern (frontend on workbench, workers as separate pods, trainer submitted via `ray job submit`). Use this when frontend/workers/trainer live on different pods.
- `launch_dynamo_3w_4s.sh` — same family, 3 workers + solo trainer.

**Code to mirror** (`examples/`):
- `examples/tau-bench/generate_with_tau.py` — close analogue for `async def generate(args, sample, sampling_params) -> Sample`: call an external agent harness, convert result → slime `Sample`, return.
- `examples/retool/generate_with_retool.py` — richer tool-loop example (402 LOC) if you need per-turn control.
- `examples/fully_async/fully_async_rollout.py` and `train_async.py` — reference for async rollout/training overlap and delayed weight updates. For SWE-bench Pro, start with `train_async.py` plus `--update-weights-interval > 1`; do **not** enable fully-async background rollout initially.
- `slime/rollout/sglang_rollout.py` and `examples/retool/generate_with_retool.py` — references for rollout logprob shape. Dynamo/OpenAI-compatible completions already sends token IDs to `/v1/completions`, requests `logprobs: 0` plus `return_tokens_as_token_ids: True`, and reads `choices[0].logprobs.token_logprobs`. The custom agent model adapter should mirror this path instead of using `/v1/chat/completions`.
- `slime/utils/mask_utils.py` and `slime/rollout/sft_rollout.py` — use `MultiTurnLossMaskGenerator` to convert SWE-agent `history`/mini-swe-agent `messages` into `tokens`, `response_length`, and assistant-only `loss_mask`.
- `slime/rollout/sglang_rollout.py:288` — confirms `logprobs: 0` is the baseline (prevents the regression hit last month).
- `slime/backends/dynamo_utils/dynamo_engine.py` — dynamo adapter; where `--dynamo-frontend-url` is consumed.

**SWE-bench Pro runner** (`~/proj/SWE-bench_Pro-os/`):
- `swe_bench_pro_eval.py` already has the local-Docker path we need: `assemble_workspace_files()`, `create_entryscript()`, `eval_with_docker()`, and final pass/fail logic over `FAIL_TO_PASS`/`PASS_TO_PASS`.
- Local `helper_code/sweap_eval_full_v2.jsonl` uses uppercase `FAIL_TO_PASS` / `PASS_TO_PASS` and mixed list/string encodings. Normalize these before feeding slime/eval; do not assume lowercase `fail_to_pass` / `pass_to_pass` are present.
- SWE-agent's default API model path is LiteLLM-based. Avoid it by adding a local custom model class or by using mini-swe-agent's `--model-class` hook with a direct `requests`/`aiohttp` OpenAI-compatible adapter.
- SWE-agent `.traj` files include `history`, `trajectory`, and `info.submission`; use `history` for train tokens and `info.submission` for eval. If `history` is missing in an older trajectory format, fall back to reconstructing messages from `trajectory[*].response` and `trajectory[*].observation`, but treat that as a smoke-only fallback.

**Model + conversion**:
- `scripts/models/glm4.7-30B-A3B.sh` — sourced as `MODEL_ARGS`. MLA (q-lora 768, kv-lora 512, qk-head-dim 192, v-head-dim 256), 64 experts/4 active, moe-grouped-gemm + moe-permute-fusion.
- `slime/backends/megatron_utils/megatron_to_hf/glm4moe.py` — conversion shim.

## Topology

```
Node A (8×B200)                           Node B (8×B200)
┌────────────────────────────────┐        ┌────────────────────────────────┐
│ slime-trainer    GPUs 0-3 TP=4 │        │ rollout-engine-1 GPUs 0-3 TP=4 │
│ rollout-engine-0 GPUs 4-7 TP=4 │ ◄────► │ rollout-engine-2 GPUs 4-7 TP=4 │
│ eval-workers   low concurrency │        │ eval-workers   low concurrency │
│ dockerd DaemonSet (privileged) │        │ dockerd DaemonSet (privileged) │
│ NATS broker (1 pod)            │        │                                │
│ etcd + dynamo.frontend         │        │                                │
└────────────────────────────────┘        └────────────────────────────────┘
                    │                                           │
                    └─── NCCL weight broadcast (16 ranks) ──────┘
```

NCCL weight-update group: 4 trainer + 12 engine ranks = 16-way, same shape as retool.
Initial rollout/eval concurrency: target 4–8 trajectories in flight. Ramp only after P3 shows stable CPU, Docker cache, and weight-sync behavior. Later target: 3 engines × ~20–30 in-flight = 60–90 concurrent trajectories.

## Resource budget

### Per-pod requests

| Pod | Node | GPU | vCPU req | RAM req | Priority |
|---|---|---:|---:|---:|---|
| slime-trainer | A | 4 | 48 | 256 GiB | high |
| rollout-engine-0 | A | 4 | 32 | 128 GiB | high |
| rollout-engine-1 | B | 4 | 32 | 128 GiB | high |
| rollout-engine-2 | B | 4 | 32 | 128 GiB | high |
| eval-workers "small" initial (2/node × 2 = 4) | A, B | 0 | 2 | 8 GiB | low |
| eval-workers "large" initial (1/node × 2 = 2) | A, B | 0 | 4 | 16 GiB | low |
| eval-workers ramp target | A, B | 0 | scale after P3 | scale after P3 | low |
| dockerd DaemonSet | A, B | 0 | 4 | 8 GiB | system |
| NATS broker | A | 0 | 2 | 2 GiB | system |
| etcd + dynamo.frontend | A | 0 | 2 | 4 GiB | system |

Initial per-node reserved: Node A/B ~90–100 vCPU before eval subprocess bursts. Keep headroom for Ray, kubelet, dockerd, NATS/etcd/frontend, and NCCL.
Initial eval capacity: **6 concurrent** (4 small + 2 large). Ramp to 12, then 30 only after queue depth and trainer jitter are measured.

### Instance scheduling classes (sampled profiles)

| Class | Dispatch heuristic | Resources | Example instances |
|---|---|---|---|
| small | default | 2 CPU / 8 GiB req, 4 CPU / 16 GiB lim | ansible, qutebrowser, flipt, vuls |
| large | `instance_id` matches `tutanota\|teleport\|element-web\|openlibrary` | 4 CPU / 16 GiB req, 4 CPU / 30 GiB lim | tutanota, teleport, element-web, openlibrary |

Dispatch heavies at the front of each rollout batch so they don't straggle wall-clock.

### Observed per-instance envelope (P90)

| Language | P50 RAM | P90 RAM | P50 CPU | P90 wall-clock |
|---|---:|---:|---:|---:|
| Python | 3 GiB | 6 GiB | 2 | 15 min |
| Go | 2 GiB | 8 GiB | 2 | 10 min |
| JS/TS | 5 GiB | 10 GiB | 2 | 30 min |

Composite P90: ~8 GiB / 3 CPU / 25 min. Worst-case (tutanota, teleport): 15 GiB / 4 CPU / 45 min.

## Slime config

Create `.context/workbench_run_slime_swebench_pro_kvr.sh` by copying `.context/workbench_run_slime_retool_kvr.sh`, but submit `train_async.py` instead of `train.py` and swap these args:

```bash
--actor-num-nodes 1 --actor-num-gpus-per-node 4
--tensor-model-parallel-size 4 --sequence-parallel
--expert-model-parallel-size 4 --expert-tensor-parallel-size 1
--pipeline-model-parallel-size 1 --context-parallel-size 1

--rollout-backend dynamo
--dynamo-frontend-url http://<frontend-pod>:3000
--rollout-num-gpus 12 --rollout-num-gpus-per-engine 4

--prompt-data /data/swebench-pro/swebench_pro_train.jsonl
--input-key prompt --label-key instance_id

--rollout-max-response-len 32768 --rollout-max-context-len 131072
--rollout-batch-size 8 --over-sampling-batch-size 8 --n-samples-per-prompt 1
--rollout-temperature 1 --num-rollout 5

--use-dynamic-batch-size --max-tokens-per-gpu 32768

--global-batch-size 8 --balance-data
--use-kl-loss --kl-loss-coef 0.0 --advantage-estimator grpo
--update-weights-interval 2
--recompute-granularity full --recompute-method uniform --recompute-num-layers 1

--hf-checkpoint /data/glm-4.7-30b-a3b
--ref-load /data/glm-4.7-30b-a3b_torch_dist
--custom-generate-function-path generate_with_swebench_pro.generate
--custom-rm-path generate_with_swebench_pro.reward_func
```

`MODEL_ARGS` comes from `source scripts/models/glm4.7-30B-A3B.sh` at the top of the script.

Do **not** enable `--partial-rollout` or `--mask-offpolicy-in-partial-rollout` for the first SWE-bench Pro runs. Existing custom-agent examples assert partial rollout is unsupported, and we only need ordinary async RL with delayed weight updates.

Do **not** enable `--use-tis` until P1c passes. TIS requires `sample.rollout_log_probs` aligned to `sample.response_length`. The rollout engine can emit logprobs, but the agent model adapter must propagate them into the saved trajectory.

Set `PYTHONPATH` in the Ray runtime env to include `/code/slime/examples/swebench-pro` in addition to `/root/Megatron-LM/` and `/code/slime`, matching the retool script pattern.

## Storage

| Path | Type | Size | Purpose |
|---|---|---:|---|
| `/data/glm-4.7-30b-a3b` | PVC `model-cache` | ~60 GB | HF weights |
| `/data/glm-4.7-30b-a3b_torch_dist` | PVC `model-cache` | ~60 GB | mcore-converted weights |
| `/data/checkpoints/` | PVC `model-cache` | 200 GB | training saves |
| `/data/swebench-pro/` | PVC `model-cache` | 5 GB | dataset + run_scripts |
| `/var/lib/docker` | **per-node local NVMe** (hostPath) | 500 GB+ | eval image cache — do not put on NFS |
| `/swepro-workspaces` | **per-node local NVMe** (hostPath) | 100 GB+ | shared bind-mount workspace visible to both eval pods and dockerd |

## Files to create

1. **`examples/swebench-pro/prepare_swebench_pro_data.py`** — one-time preprocessing script.
   - Input: `~/proj/SWE-bench_Pro-os/helper_code/sweap_eval_full_v2.jsonl` or the HF dataset export.
   - Output: `/data/swebench-pro/swebench_pro_train.jsonl`.
   - Each line should include top-level `prompt` and `instance_id`, plus `metadata` containing the normalized raw row: `repo`, `base_commit`, `selected_test_files_to_run`, `fail_to_pass`, `pass_to_pass`, Docker image/tag inputs, and any path needed by the agent.
   - Normalize `FAIL_TO_PASS`/`PASS_TO_PASS` into Python lists regardless of whether upstream encoded them as JSON strings or lists. Use existing `problem_statement` directly for the prompt when `requirements`/`interface` fields are absent.

2. **`examples/swebench-pro/completions_direct_model.py`** — direct `/v1/completions` model adapter, no LiteLLM dependency.
   - Implement the minimal interface required by the chosen scaffold:
     - mini-swe-agent: class with `query(messages, **kwargs) -> {"content": ..., "extra": ...}` and `get_template_vars()`.
     - SWE-agent: subclass/parallel implementation of `AbstractModel` compatible with `sweagent.agent.models.get_model`, or a small local patch that selects this adapter for `agent.model.name`.
   - Load the same HF tokenizer used by slime, render chat locally with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`, tokenize to `prompt_ids`, then POST to `<dynamo-frontend>/v1/completions`.
   - Request `{"prompt": prompt_ids, "max_tokens": ..., "temperature": ..., "top_p": ..., "logprobs": 0, "return_tokens_as_token_ids": true, "stream": false}`. This mirrors `slime/rollout/sglang_rollout.py::_generate_dynamo`.
   - Decode `choices[0].text` for the agent-facing message and preserve raw response metadata under `extra.response`, including generated token IDs from `choices[0].logprobs.tokens` and `choices[0].logprobs.token_logprobs`.
   - Add stop strings only if they match the scaffold parser behavior. Avoid server-side chat formatting to prevent template drift.
   - Keep retry/error handling small and explicit; no proxy/model-registry abstraction.

3. **`examples/swebench-pro/generate_with_swebench_pro.py`** — custom generate + reward module.
   - `generate()`: per-sample, spawn SWE-agent or mini-swe-agent subprocess configured to use `completions_direct_model.py`; capture `.traj`, final patch, and exit status.
   - Convert SWE-agent `.traj` `history` into slime training data with `MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=args.loss_mask_type)`. Set `sample.tokens`, `sample.response`, `sample.response_length`, `sample.loss_mask`, `sample.status`, and metadata (`instance_id`, `patch`, `traj_path`, `exit_status`, `repo`, eval fields). `loss_mask` must cover only assistant tokens from the first assistant token through the end; observations/user/tool output must be zero-masked.
   - If using mini-swe-agent fallback, convert saved `messages` the same way. This is likely the fastest P2 path because mini-swe-agent saves exact OpenAI-style messages.
   - Rollout logprobs: use `completions_direct_model.py` so each assistant response stores generated token ids and token logprobs from `/v1/completions` (`choices[*].logprobs.token_logprobs`). When converting the trajectory, build `sample.rollout_log_probs` with length exactly `sample.response_length`: real logprobs for generated assistant content tokens; `0.0` for zero-masked user/observation/tool/template tokens. Assert alignment before returning the sample.
   - Keep `--max_turns 50` / per-instance call limit hard cap to bound trajectory length. Return `Sample.Status.TRUNCATED` if the agent exits due to turn/call/context limits.
   - `reward_func()` (async): accept either one `Sample` or a `list[Sample]`. For a list, `asyncio.gather` per-sample calls and return rewards in order. For one sample, publish `{request_id, instance_id, patch, repo, fail_to_pass, pass_to_pass, selected_test_files_to_run}` to NATS (`nats://<nats-service>:4222`, subject `swepro.evals`), await reply with a long timeout, attach raw eval JSON to `sample.metadata["eval"]`, and return scalar `1.0` or `0.0`.

4. **`docker/swepro-eval` Docker image** — new image. Thin Python runner:
   - subscribes to NATS subject `swepro.evals`
   - for each message: reuse the local-Docker logic from `~/proj/SWE-bench_Pro-os/swe_bench_pro_eval.py` (`assemble_workspace_files`, `create_entryscript`, `get_dockerhub_image_uri`, parser output handling) instead of reimplementing parsing.
   - run `docker run jefzda/sweap-images:<instance_tag>` through `DOCKER_HOST=unix:///var/run/docker-swepro.sock`; execute `entryscript.sh`; parse `output.json`; reply with `{passed, tests, status_code, stdout_tail, stderr_tail, error}`.
   - implement worker-level timeout, container cleanup by label/request ID, and a small internal semaphore (`SWEPRO_EVAL_CONCURRENCY=1` initially).
   - Dockerfile lives at `docker/swepro-eval/Dockerfile`.

5. **`.context/k8s-dockerd.yaml`** — privileged dockerd DaemonSet, one per selected node.
   - Mount local-NVMe hostPath for Docker graph storage at `/var/lib/docker`.
   - Mount hostPath socket at `/var/run/docker-swepro.sock`.
   - Mount shared local-NVMe hostPath at `/swepro-workspaces`. This exact path must also be mounted in eval pods, because the Docker daemon can only bind-mount paths visible in its own mount namespace.
   - **Confirm nscale allows `privileged: true` before committing** (see Open Items #1).

6. **`.context/k8s-swepro-eval.yaml`** — two `Deployment`s (small/large replica sets), `nodeSelector` on the GPU-node hostnames, `priorityClassName: low`, no GPU request, mounts `/var/run/docker-swepro.sock` and `/swepro-workspaces`. Start with 2 small + 1 large total or 2 small + 1 large per node; ramp later.

7. **`.context/k8s-nats.yaml`** — single-node NATS on Node A (`Deployment` + `Service`).

8. **`.context/k8s-swepro-trainer.yaml`** — trainer pod (4 GPU, tp=4), based on `k8s-dynamo-worker.yaml` but with the slime trainer entrypoint instead of `dynamo.sglang`.

9. **`.context/k8s-swepro-engine.yaml`** — 3× engine pod template (parameterize node + etcd host). Again start from `k8s-dynamo-worker.yaml`; bump GPUs to 4 and add `--tp 4`.

10. **`.context/launch_swebench_pro.sh`** — top-level runner that applies the YAMLs in order and tails logs. Mirror `launch_dynamo_external.sh`.

## Phase plan

| Phase | Scope | Duration | Gate |
|---|---|---|---|
| **P0: Preflight** | Confirm kube context, pick 2 free B200 nodes, verify PVC mounts + image pull, verify privileged pods allowed. Pre-pull top-20 SWE-bench Pro images on both nodes. | 0.5 day | `kubectl apply --dry-run` passes; `docker pull` of top-20 images completes on both nodes |
| **P0b: Data prep** | Normalize SWE-bench Pro JSONL into slime prompt-data with metadata. | 0.5 day | `/data/swebench-pro/swebench_pro_train.jsonl` loads through slime `Dataset`; metadata has list-valued `fail_to_pass` / `pass_to_pass` |
| **P1: Eval infra** | Build `swepro-eval` image; apply dockerd DaemonSet + NATS + low-concurrency eval Deployments. Run 1 eval container end-to-end on a known-good gold patch. | 1–2 days | ≥5 green evals end-to-end via NATS round-trip; failed evals clean up containers/workspaces |
| **P1b: Direct model bridge** | Add `completions_direct_model.py`, run one SWE-agent or mini-swe-agent trajectory without importing LiteLLM, and convert saved `history`/`messages` to slime `Sample` fields. | 0.5–1 day | No LiteLLM import; `/v1/completions` receives token-ID prompts; `len(sample.loss_mask) == sample.response_length`; at least one assistant token is unmasked; no user/observation tokens are unmasked |
| **P1c: Logprob bridge** | Preserve generated token ids + token logprobs from the direct OpenAI-compatible response, then align them to slime `response_length`. | 0.5–1 day | `len(sample.rollout_log_probs) == sample.response_length`; TIS dry-run reaches loss computation without assertion |
| **P2: Rollout-only smoke** | Stand up trainer + 3 engines on 2 nodes using the new YAMLs. Run 4 instances × 1 sample via slime endpoint with `--debug-rollout-only`. **No training step.** | 1 day | trajectory → `Sample` → patch → eval → reward, full loop green |
| **P3: 1-rollout async training** | 4–8 instances × 1 sample × 1–2 steps with `train_async.py`, `--update-weights-interval 2`, no partial rollout. Enable `--use-tis` only if P1c passed. | 1 day | non-zero loss, step completes, delayed weight-sync succeeds, eval queue stays bounded |
| **P4: Scaled dev run** | 16 instances × 1 sample × 5 rollouts, then 32 instances × 2 samples × 5 rollouts if stable. | 1–2 days wall-clock | all steps complete, non-degenerate reward signal, no trainer starvation |
| **P5: Tuning** | Adjust `--over-sampling-batch-size`, `--max-tokens-per-gpu`, per-engine concurrency from P4 data. | 1 day | — |

Total: ~1 week focused work, 2 weeks with debugging.

## Risks

| Risk | Mitigation |
|---|---|
| NCCL weight-sync hang (seen on retool reruns) | Worker-pod recreation runbook kept from retool debugging |
| Docker image cold-pull tax (15+ GB for some JS/TS) | Pre-warm top-20 images on each node before P4 (P0 gate) |
| LiteLLM supply-chain/dependency risk | Do not install or import LiteLLM in the SWE-bench Pro path; use `completions_direct_model.py` with direct HTTP calls |
| Chat template drift between adapter and server | Do all prompt formatting locally with the training tokenizer and call `/v1/completions` with token IDs |
| SWE-agent trajectories blow past context (200+ turns) | Cap `--max_turns 50` / per-instance call limit; mark as `TRUNCATED`; do not rely on slime partial rollout initially |
| SWE-agent trajectory does not contain exact model context | Prefer `.traj` `history`. If unavailable, use mini-swe-agent `messages` for P2 or patch SWE-agent save hooks before training |
| Rollout logprobs misalign with chat-template tokens | Preserve token ids/logprobs at generation time and validate `len(rollout_log_probs) == response_length`; fill `0.0` only for zero-masked observation/template tokens |
| Eval queue depth blows up if trajectories finish simultaneously | NATS backpressure + low initial worker count; start `max_in_flight=6`, ramp to 12 then 30 |
| dockerd cannot see eval workspaces | Mount the same local-NVMe hostPath at the same `/swepro-workspaces` path in dockerd and eval pods |
| nscale doesn't allow privileged DaemonSets | Fallback: rootless Podman, or K8s ephemeral-pod-per-eval via Job API |
| Trainer CPU jitter from noisy eval neighbor during NCCL collectives | kubelet `--cpu-manager-policy=static` + `Guaranteed` QoS on trainer; cpuset isolation |
| `aphoh/slime:dynamo-rl-support-v5` missing SWE-agent deps | Extend image (new Dockerfile layer adding `sweagent`, docker CLI); build via `docker-builder` pod |

## Open items to confirm before P1

1. nscale allows `privileged: true` pods (required for dockerd host-socket pattern). Test with a minimal privileged pod first.
2. NVMe capacity per 8×B200 node ≥ 500 GB for Docker image cache plus ≥100 GB for `/swepro-workspaces`. Check with `kubectl debug node/<name> -- df -h` or equivalent.
3. 8×B200 nscale node has NVLink8 (single NVSwitch domain) — needed for TP=4 co-placement. `nvidia-smi topo -m` inside a running pod.
4. Ref model placement: colocate with trainer (needs ~30 GB extra per GPU) vs. offload to engines during idle. Default: colocate; revisit if OOM.
5. `aphoh/slime:dynamo-rl-support-v5` has `sweagent` or `mini-swe-agent` importable without needing LiteLLM on the execution path — if not, rebuild via `docker-builder` pod before P2.
6. SWE-agent `.traj` generated by the installed version includes `history`. If not, use mini-swe-agent first or patch SWE-agent to persist model query history before training.
