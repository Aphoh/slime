# DeepSeek V3 64-GPU FP8 Trainer Smoke Handoff

Date: 2026-05-28
Repo: `/Users/warnold/dev/slime`
Target cluster: `dynamo-gcp-dev-02`
Target namespace: `warnold-dynamo`

This handoff is for the next agent taking over the DeepSeek V3/R1 trainer fit experiment. The immediate question was whether we can smoke test a 64-GPU GB200 trainer using FP8 resident parameters and precision-aware Adam states, and whether there is a free rack.

## Current Conclusion

There is not currently a free rack for a 64-GPU / 16-node smoke test.

Each GB200 node in this cluster exposes 4 GPUs. A 64-GPU trainer needs 16 full nodes in one rack/compute domain if we want to avoid cross-rack surprises. The last inventory showed:

- `o7v`: 50/72 GPUs requested, only 3 fully free schedulable nodes: `cw71`, `q8sl`, `tj86`.
- `w0e`: 50/72 GPUs requested, only 1 fully free schedulable node: `qp15`; two additional nodes were free but cordoned: `d686`, `pmgm`.

Best reshuffle target is probably `w0e`, because `bis-rl` holds 8 full nodes / 32 GPUs there. However, `bis-rl` is active: 16 GPUs were pegged at 100% and another 16 trainer GPUs had large model memory resident.

## Correct Kubernetes Target

Use this context and namespace for every command:

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
NS=warnold-dynamo
kubectl --context="$CTX" -n "$NS" get pods
```

Do not use the nscale/B200 cluster. If `kubectl` errors with expired credentials, refresh Teleport:

```bash
tsh login --proxy=nv-prd-dgxc.teleport.sh
tsh kube login dynamo-gcp-dev-02
```

After login, verify:

```bash
kubectl config current-context
kubectl --context="$CTX" -n "$NS" get pods -o wide
```

Expected current context string:

```text
nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
```

## Relevant Local Files

- Main launcher:
  - `examples/swebench-pro/launch_swepro_rl.py`
- Current stack manifest:
  - `examples/swebench-pro/k8s-gcp02-swepro-stack.yaml`
- Existing GCP02 NCCL runbook:
  - `examples/swebench-pro/gcp02-nccl-runbook.md`
- Existing successful slow-run record:
  - `.context/successful-slow-run.md`
- DeepSeek model args:
  - `scripts/models/deepseek-v3.sh`
  - `scripts/models/deepseek-v3-5layer.sh`
  - `scripts/models/deepseek-v3-20layer.sh`
- Low precision docs:
  - `docs/en/advanced/low-precision.md`
  - `/Users/warnold/dev/nemorl/docs/fp8.md`
  - `/Users/warnold/dev/Megatron-Bridge/docs/performance-guide.md`

There are uncommitted local changes in this repo. Do not reset them. They include SWE-Pro tracing/limits/session changes and stack YAML changes.

## DeepSeek GPU Recipe Findings

GB200-specific Megatron-Bridge DeepSeek V3 recipes point to 256 GPUs as the normal trainer size:

- `/Users/warnold/dev/Megatron-Bridge/docs/performance-summary.md`
  - DGX-GB200 DeepSeekV3 perf rows use 256 GPUs.
  - 26.02 MXFP8 row: 256 GPUs, `TP=1`, `PP=4`, `CP=1`, `VP=4`, `EP=64`.
  - 25.11 BF16 row: 256 GPUs, `TP=1`, `PP=4`, `CP=1`, `VP=4`, `EP=64`.
  - 25.11 FP8-MX row: 256 GPUs, `TP=1`, `PP=8`, `CP=1`, `VP=4`, `EP=32`.

NeMoRL has smaller GRPO DeepSeek V3 recipes:

- `/Users/warnold/dev/nemorl/examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n4g.yaml`
  - 32 nodes x 4 GPUs = 128 total GPUs.
  - Inherits base config and overrides `pipeline_model_parallel_size: 8`.
- `/Users/warnold/dev/nemorl/examples/configs/recipes/llm/performance/grpo-deepseek-v3-32n8g.yaml`
  - 32 nodes x 8 GPUs = 256 total GPUs.
  - `PP=16`, `EP=16`, `TP=1`, `CP=1`.
- `/Users/warnold/dev/nemorl/examples/configs/recipes/llm/performance/dapo-deepseek-v3-64n8g.yaml`
  - 64 nodes x 8 GPUs = 512 total GPUs.
  - `TP=8`, `PP=8`, `CP=4`, `EP=32`.

Slime DeepSeek R1 doc:

- `docs/en/examples/deepseek-r1.md`
  - 128 x H100 example.
  - Megatron shape: `TP8`, `PP4`, `CP4`, `EP32`.
  - Uses CPU Adam to save HBM.

Takeaway:

- Recipe-backed "normal" GB200 answer: 256 trainer GPUs.
- Smaller RL recipe lower bound: 128 GPUs.
- 64 GPUs is experimental/cram-it-in territory.

## FP8 Params and Optimizer Theory

The current theory was to try making 64 GB200 GPUs fit with FP8 resident parameters and precision-aware optimizer states.

Important caveat:

- In current slime, `--fp8-param-gather` is the flag that actually keeps TransformerEngine weights resident in FP8.
- `docs/en/advanced/low-precision.md` says `--fp8-param-gather` is currently incompatible with CPU Adam.
- The same doc says that without `--fp8-param-gather`, TE weights remain BF16 resident and are only cast to FP8 for GEMMs.

Therefore, the smoke should test:

- `--fp8-param-gather`
- CPU optimizer offload disabled
- precision-aware optimizer enabled
- BF16 Adam moments via `--exp-avg-dtype bfloat16` and `--exp-avg-sq-dtype bfloat16`
- `--no-save-optim`
- huge save interval or saves disabled

Rough 64-GPU average memory estimate:

```text
FP8 params:          ~681 GB total   -> ~11 GB/GPU avg
BF16 grads:          ~1.36 TB total  -> ~21 GB/GPU avg
BF16 Adam m/v:       ~2.72 TB total  -> ~43 GB/GPU avg
PA master/remainder: ~1.3 TB-ish     -> ~20 GB/GPU avg
-------------------------------------------------------
Before activations:  ~95-120 GB/GPU avg
```

The first likely failure modes:

- TE/FusedAdam/precision-aware incompatibility.
- high-water memory spike during model or optimizer initialization.
- later, activation memory at long context.

## Smoke-Test Plan

Do not start with full 64 GPUs until a rack is reshuffled.

Suggested sequence:

1. Compatibility smoke with tiny DeepSeek:
   - Use `scripts/models/deepseek-v3-5layer.sh`.
   - Use a small number of nodes available in one rack.
   - Add `--fp8-param-gather`.
   - Disable CPU optimizer offload.
   - Run one tiny rollout/step.
   - Goal: catch argument, TransformerEngine, and optimizer compatibility quickly.

2. Full-model fit smoke:
   - Use `scripts/models/deepseek-v3.sh`.
   - Use 64 GPUs in one rack/compute domain after reshuffle.
   - Start with short sequence.
   - Then increase to 131k context.

Candidate 64-GPU shapes:

```text
TP8 PP2 CP4 EP32
```

Closest to the 128-GPU DeepSeek-R1-style shape while reducing PP. If long-context activations blow up, try:

```text
TP4 PP2 CP8 EP32
```

That gives more context parallelism.

## Launcher Env for Full 64-GPU Smoke

Use this only after reserving 16 full 4-GPU nodes in one rack:

```bash
export SWEPRO_RUN_ID=deepseekv3_fp8param_64gpu_tp8_pp2_cp4_ep32_smoke_$(date -u +%Y%m%dT%H%M%SZ)
export SWEPRO_MODEL_ARGS_SCRIPT=scripts/models/deepseek-v3.sh
export SWEPRO_ACTOR_NUM_NODES=16
export SWEPRO_ACTOR_NUM_GPUS_PER_NODE=4
export SWEPRO_TP=8
export SWEPRO_PP=2
export SWEPRO_CP=4
export SWEPRO_EP=32
export SWEPRO_ETP=1
export SWEPRO_SEQUENCE_PARALLEL=1
export SWEPRO_QKV_FORMAT=thd
export SWEPRO_USE_DYNAMIC_BATCH_SIZE=1
export SWEPRO_MAX_TOKENS_PER_GPU=16384
export SWEPRO_SEQ_LENGTH=65536
export SWEPRO_MAX_CONTEXT_LEN=65536
export SWEPRO_MAX_RESPONSE_LEN=65536
export SWEPRO_LOG_PROBS_CHUNK_SIZE=512
export SWEPRO_DEFER_FP32_LOGITS=1
export SWEPRO_MOE_TOKEN_DISPATCHER_TYPE=alltoall

# Optimizer experiment.
export SWEPRO_OPTIMIZER_CPU_OFFLOAD=0
export SWEPRO_USE_PRECISION_AWARE_OPTIMIZER=1
export SWEPRO_EXP_AVG_DTYPE=bfloat16
export SWEPRO_EXP_AVG_SQ_DTYPE=bfloat16
export SWEPRO_SAVE_INTERVAL=1000000
export SWEPRO_DISABLE_SAVE=1

# Keep the workload tiny for smoke.
export SWEPRO_NUM_ROLLOUT=1
export SWEPRO_ROLLOUT_BATCH_SIZE=1
export SWEPRO_OVER_SAMPLING_BATCH_SIZE=1
export SWEPRO_N_SAMPLES_PER_PROMPT=1
export SWEPRO_GLOBAL_BATCH_SIZE=1
```

Launch extra args:

```bash
python examples/swebench-pro/launch_swepro_rl.py -- \
  --transformer-impl transformer_engine \
  --bf16 \
  --fp8-format e4m3 \
  --fp8-recipe blockwise \
  --fp8-param-gather \
  --no-save-optim
```

For tiny compatibility smoke, change:

```bash
export SWEPRO_MODEL_ARGS_SCRIPT=scripts/models/deepseek-v3-5layer.sh
```

and reduce topology to what the currently free nodes can satisfy. Do not interpret a tiny-model pass as proof that the full model fits.

## Current Rack Usage Snapshot

This was sampled on 2026-05-28 after refreshing Teleport.

### o7v

Summary:

- 50/72 GPUs requested.
- 22 schedulable GPUs free.
- Fully free schedulable nodes: `cw71`, `q8sl`, `tj86`.

GPU users:

| Namespace | GPUs | Nodes |
| --- | ---: | --- |
| `ibhosale-dynamo` | 8 | `6z5p`, `cc2r` |
| `kavin` | 8 | `21z7`, `8wwv` |
| `ryan` | 8 | `60xv`, `nxfg` |
| `jothomson-llmd` | 6 | `9jvj`, `hqmb`, `xjjk` |
| `jegu` | 4 | `rv76` |
| `jihao` | 4 | `x4jh` |
| `nlevin-riptide` | 4 | `v6sj` |
| `tzulingk-ft-tests` | 4 | `v3gc` |
| `dynamo-qa-ci` | 3 | `0mb9`, `hrk5`, `xjjk` |
| `tzulingk-dis2138` | 1 | `9jvj` |

### w0e

Summary:

- 50/72 GPUs requested.
- 14 schedulable GPUs free.
- Fully free schedulable node: `qp15`.
- Free but cordoned nodes: `d686`, `pmgm`.

GPU users:

| Namespace | GPUs | Nodes |
| --- | ---: | --- |
| `bis-rl` | 32 | `06vd`, `5rzq`, `gjn2`, `grdk`, `gtwc`, `lv8c`, `mb4d`, `qjmr` |
| `jothomson-llmd` | 10 | `24wk`, `46k1`, `9crm`, `mc07`, `tqtb` |
| `jihao` | 4 | `69ln` |
| `nlevin-agentic` | 4 | `ss59` |

### bis-rl Activity Check

`bis-rl` was not idle.

I sampled `nvidia-smi` 3 times over about 20 seconds. Results:

Actively pegged at 100% util, 16 GPUs total:

| Node | Pod | GPU util | Memory |
| --- | --- | --- | --- |
| `5rzq` | `qwen235-rl-agg-1-0-worker-0-worker-ldr-fdgd4` | 100/100/100/100% | ~170 GiB each |
| `gjn2` | `qwen235-rl-agg-1-0-worker-0-worker-wkr-bqmvm` | 100/100/100/100% | ~170 GiB each |
| `lv8c` | `qwen235-rl-agg-1-0-worker-1-worker-ldr-d4vqw` | 100/100/100/100% | ~170 GiB each |
| `06vd` | `qwen235-rl-agg-1-0-worker-1-worker-wkr-hw8gg` | 100/100/100/100% | ~170 GiB each |

Allocated but compute-idle at sample time, 16 GPUs total, with model memory resident:

| Node | Pod | GPU util | Memory |
| --- | --- | --- | --- |
| `grdk` | `qwen235-rl-1-trainer-0` | 0/0/0/0% | ~138-145 GiB each |
| `qjmr` | `qwen235-rl-1-trainer-1` | 0/0/0/0% | ~138-139 GiB each |
| `mb4d` | `qwen235-rl-1-trainer-2` | 0/0/0/0% | ~138-139 GiB each |
| `gtwc` | `qwen235-rl-1-trainer-3` | 0/0/0/0% | ~138-139 GiB each |

Conclusion: ask `bis-rl` before touching those nodes. Their agg/worker side is actively busy.

## Commands To Re-Run Inventory

GPU request summary by rack:

```bash
python3 - <<'PY'
import json, subprocess, collections
CTX='nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02'
def kjson(*args):
    return json.loads(subprocess.check_output(['kubectl', f'--context={CTX}', *args], text=True))
nodes=kjson('get','nodes','-o','json')['items']
pods=kjson('get','pods','-A','-o','json')['items']
node_info={}
for n in nodes:
    name=n['metadata']['name']
    labels=n['metadata'].get('labels',{})
    cap=n['status'].get('capacity',{})
    alloc=n['status'].get('allocatable',{})
    gpu=int(cap.get('nvidia.com/gpu',0) or 0)
    if not gpu:
        continue
    rack='o7v' if '-o7v-' in name else 'w0e' if '-w0e-' in name else labels.get('cloud.google.com/gke-nodepool','unknown')
    node_info[name]={'rack':rack,'short':name.rsplit('-',1)[-1],'gpu_capacity':gpu,'gpu_allocatable':int(alloc.get('nvidia.com/gpu',gpu) or 0),'used':0,'unsched':bool(n.get('spec',{}).get('unschedulable')),'pods':[]}
for p in pods:
    if p.get('status',{}).get('phase') in ('Succeeded','Failed'):
        continue
    node=p.get('spec',{}).get('nodeName')
    if node not in node_info:
        continue
    gpu=0
    for c in p.get('spec',{}).get('containers',[]):
        for bucket in ('requests','limits'):
            val=c.get('resources',{}).get(bucket,{}).get('nvidia.com/gpu')
            if val:
                gpu=max(gpu,int(val))
    if gpu:
        ns=p['metadata']['namespace']
        pod=p['metadata']['name']
        node_info[node]['used'] += gpu
        node_info[node]['pods'].append({'namespace':ns,'pod':pod,'gpu':gpu})
for rack in ('o7v','w0e'):
    nodes_r=[(n,i) for n,i in node_info.items() if i['rack']==rack]
    cap=sum(i['gpu_capacity'] for _,i in nodes_r)
    used=sum(i['used'] for _,i in nodes_r)
    free_sched=sum(max(0,i['gpu_allocatable']-i['used']) for _,i in nodes_r if not i['unsched'])
    full_free=[i['short'] for _,i in sorted(nodes_r) if i['used']==0 and not i['unsched']]
    print(f'{rack}: {used}/{cap} GPUs requested, {free_sched} schedulable GPUs free, fully-free schedulable nodes={len(full_free)} ({", ".join(full_free)})')
PY
```

Check `bis-rl` activity:

```bash
python3 - <<'PY'
import subprocess, json, time
CTX='nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02'
NS='bis-rl'
def run(args, timeout=30):
    return subprocess.run(['kubectl', f'--context={CTX}', '-n', NS, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
pods_json=json.loads(run(['get','pods','-o','json'], timeout=60).stdout)
gpu_pods=[]
for p in pods_json['items']:
    if p.get('status',{}).get('phase') in ('Succeeded','Failed'):
        continue
    gpu=0
    for c in p.get('spec',{}).get('containers',[]):
        for bucket in ('requests','limits'):
            val=c.get('resources',{}).get(bucket,{}).get('nvidia.com/gpu')
            if val:
                gpu=max(gpu,int(val))
    if gpu:
        gpu_pods.append((p['metadata']['name'], gpu, p.get('spec',{}).get('nodeName','')))
print('bis-rl GPU pods:')
for name,gpu,node in gpu_pods:
    print(f'  {name} gpu_req={gpu} node={node.rsplit("-",1)[-1]}')
query='index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw'
for sample in range(3):
    print(f'\n=== nvidia-smi sample {sample+1}/3 ===')
    for name,gpu,node in gpu_pods:
        r=run(['exec', name, '--', 'nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'], timeout=25)
        out=(r.stdout or '').strip()
        if r.returncode != 0:
            print(f'{name}: ERROR rc={r.returncode}: {out[-500:]}')
            continue
        rows=[]
        for line in out.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts) >= 6:
                rows.append(parts)
        util=[int(float(row[1])) for row in rows if row[1].replace('.','',1).isdigit()]
        mem=[int(float(row[3])) for row in rows if row[3].replace('.','',1).isdigit()]
        print(f'{name}: util%={"/".join(str(u) for u in util)} memMiB={"/".join(str(m) for m in mem)}')
    if sample < 2:
        time.sleep(10)
PY
```

## What To Ask Other Users

If the goal is a 64-GPU single-rack smoke:

- Ask `bis-rl` first whether their `qwen235-rl-*` run on `w0e` can pause or move. They hold 8 full nodes.
- If `bis-rl` cannot move, `o7v` requires coordination with many more owners:
  - `kavin`, `ryan`, `ibhosale-dynamo`, `jothomson-llmd`, `jegu`, `jihao`, `nlevin-riptide`, `tzulingk-ft-tests`, `dynamo-qa-ci`, `tzulingk-dis2138`.
- `w0e` is cleaner if `bis-rl` and `jothomson-llmd` can reshuffle.

## Notes About MNNVL/NCCL

For prior SWE-Pro runs, the known-good GCP02 NCCL env was:

```bash
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_HOST_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_MNNVL_ENABLE=0
export NCCL_STORE_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NVIDIA_GDRCOPY=1

export UCX_TLS=cuda_ipc,cuda_copy,rc
export UCX_IB_GID_INDEX=3
export UCX_RC_TIMEOUT=600s
export UCX_KEEPALIVE_INTERVAL=300s

export SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET=1
export SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE=0
```

Things intentionally left unset:

```bash
unset NCCL_NET
unset MC_FORCE_MNNVL
unset SLIME_WEIGHT_UPDATE_MC_FORCE_MNNVL
unset NCCL_IB_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_P2P_DISABLE
unset NCCL_ALGO
unset NCCL_PROTO
```

This mattered for Slime online trainer-to-engine weight update. For the DeepSeek trainer-only smoke, first get model/optimizer fit working before adding rollout/weight-update complexity.

## Recommended Next Step

1. Ask users to free one rack, ideally `w0e` via `bis-rl`.
2. Re-run the inventory script above.
3. If there are at least 16 fully free 4-GPU nodes in one rack, launch the tiny 5-layer DeepSeek FP8-param compatibility smoke.
4. If that passes init and one optimizer step, launch full DeepSeek V3 with short sequence.
5. Only then scale sequence toward 131k.

