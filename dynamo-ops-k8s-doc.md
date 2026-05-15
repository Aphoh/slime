# Deploying Dynamo with NIXL Disaggregated Serving on Kubernetes

This guide covers how to deploy NVIDIA Dynamo with NIXL-based KV-cache transfer
for disaggregated prefill/decode serving across **nscale (InfiniBand)**, **GCP
(RoCE)**, and **AWS (EFA)** clusters. It documents the RDMA resource requests,
NIXL/UCX/libfabric environment variables, security contexts, and MNNVL
configuration needed for NVL72 multi-node NVLink topologies.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Deploying on nscale (InfiniBand)](#deploying-on-nscale-infiniband)
3. [Deploying on GCP (RoCE)](#deploying-on-gcp-roce)
4. [Deploying on AWS (EFA)](#deploying-on-aws-efa)
5. [MNNVL for NVL72 (Multi-Node NVLink)](#mnnvl-for-nvl72-multi-node-nvlink)
6. [Environment Variable Reference](#environment-variable-reference)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Dynamo disaggregated serving splits inference into **prefill** and **decode**
workers. The KV cache produced by prefill must be transferred to decode via
high-bandwidth RDMA. NIXL is the transfer library that abstracts the underlying
RDMA transport:

| Cloud  | GPU           | Runtime | Interconnect | NIXL Backend | Transport        |
| ------ | ------------- | ------- | ------------ | ------------ | ---------------- |
| nscale | B200 (x86_64) | TRT-LLM | InfiniBand   | UCX          | RC verbs         |
| GCP    | GB200 (arm64) | TRT-LLM | RoCE v2      | UCX          | RC verbs (GID 3) |
| AWS    | GB200 (arm64) | TRT-LLM | EFA          | LIBFABRIC    | SRD over EFA     |
| AWS    | H100 (x86_64) | vLLM    | EFA          | LIBFABRIC    | SRD over EFA     |

Each deployment has three service components:

- **Frontend** — stateless HTTP gateway (no GPU, no RDMA)
- **Prefill workers** — GPU + RDMA, run `--disaggregation-mode prefill`
- **Decode workers** — GPU + RDMA, run `--disaggregation-mode decode`

All workers use `--disaggregation-transfer-backend nixl` and
`--disaggregation-bootstrap-port 30001`.

---

## Deploying on nscale (InfiniBand)

| Property      | Value                                 |
| ------------- | ------------------------------------- |
| Architecture  | x86_64                                |
| GPU           | NVIDIA B200                           |
| Interconnect  | InfiniBand                            |
| NIXL backend  | UCX (default)                         |
| RDMA resource | `rdma/ib`                             |
| Security      | `IPC_LOCK` capability                 |
| Node selector | `nvidia.com/gpu.product: NVIDIA-B200` |
| MNNVL         | Not applicable (no cross-node NVLink) |

### 1. Request RDMA resources

The RDMA device plugin exposes `rdma/ib` resources. Request one per GPU:

```yaml
resources:
  limits:
    gpu: "4"
    custom:
      rdma/ib: "4"
```

No pod annotations are needed. InfiniBand devices (`/dev/infiniband/*`) are
injected automatically by the device plugin.

### 2. Set the security context

Add `IPC_LOCK` capability so NIXL/UCX can pin GPU memory for RDMA registration.
Without this, transfers fail with `waiting_timeout`:

```yaml
securityContext:
  capabilities:
    add:
      - IPC_LOCK
```

### 3. Set environment variables

These go on **both prefill and decode** worker containers:

```yaml
env:
  # --- UCX (RDMA transport) ---
  - name: UCX_TLS
    value: "cuda_ipc,cuda_copy,rc"
  - name: UCX_RC_TIMEOUT
    value: "600s"
  - name: UCX_KEEPALIVE_INTERVAL
    value: "300s"

  # --- NCCL ---
  - name: NCCL_IB_DISABLE
    value: "0"
  - name: NCCL_STORE_TIMEOUT
    value: "7200"
  - name: NCCL_DEBUG
    value: INFO

  # --- NIXL ---
  - name: NIXL_LOG_LEVEL
    value: INFO           # use DEBUG during bringup
  - name: NIXL_TELEMETRY_ENABLE
    value: "y"
  - name: NIXL_TELEMETRY_EXPORTER
    value: prometheus
  - name: NIXL_TELEMETRY_PROMETHEUS_PORT
    value: "19090"
```

### 4. Set common networking env vars (spec-level)

These go in `spec.envs` (shared by all services):

```yaml
spec:
  envs:
    - name: GLOO_SOCKET_IFNAME
      value: eth0
    - name: NCCL_SOCKET_IFNAME
      value: eth0
    - name: NATS_SERVER
      value: nats://dynamo-platform-nats.<namespace>.svc.cluster.local:4222
```

---

## Deploying on GCP (RoCE)

| Property      | Value                                                               |
| ------------- | ------------------------------------------------------------------- |
| Architecture  | arm64 (Grace Blackwell)                                             |
| GPU           | NVIDIA GB200                                                        |
| Interconnect  | RoCE v2                                                             |
| NIXL backend  | UCX (default)                                                       |
| RDMA resource | `networking.gke.io.networks/rdma-{0..N}`                            |
| Security      | `IPC_LOCK` capability                                               |
| Node selector | `kubernetes.io/arch: arm64`, `nvidia.com/gpu.product: NVIDIA-GB200` |
| MNNVL         | ComputeDomain + nodeAffinity (for wide EP/TP)                       |

### 1. Request RDMA resources

GKE GB200 clusters expose RDMA via **multi-network** resources. Each GPU maps
to a separate RDMA network (`rdma-0` through `rdma-3`). You need **both**
resource requests and pod annotations.

**4 GPUs** (e.g. TP4):

```yaml
annotations:
  networking.gke.io/default-interface: eth0
  networking.gke.io/interfaces: |
    [
      {"interfaceName":"eth0","network":"default"},
      {"interfaceName":"rdma0","network":"rdma-0"},
      {"interfaceName":"rdma1","network":"rdma-1"},
      {"interfaceName":"rdma2","network":"rdma-2"},
      {"interfaceName":"rdma3","network":"rdma-3"}
    ]
resources:
  limits:
    gpu: "4"
    custom:
      networking.gke.io.networks/rdma-0: "1"
      networking.gke.io.networks/rdma-1: "1"
      networking.gke.io.networks/rdma-2: "1"
      networking.gke.io.networks/rdma-3: "1"
```

**1 GPU** (e.g. for debugging):

```yaml
annotations:
  networking.gke.io/default-interface: eth0
  networking.gke.io/interfaces: |
    [
      {"interfaceName":"eth0","network":"default"},
      {"interfaceName":"rdma0","network":"rdma-0"}
    ]
resources:
  limits:
    gpu: "1"
    custom:
      networking.gke.io.networks/rdma-0: "1"
```

### 2. Set the security context

Same as nscale — add `IPC_LOCK` capability:

```yaml
securityContext:
  capabilities:
    add:
      - IPC_LOCK
```

Set ulimits in the container entrypoint (GB200 arm64 images may have lower
defaults):

```bash
ulimit -l unlimited && ulimit -n 1048576 && exec python3 -m dynamo.sglang ...
```

### 3. Set environment variables

These go on **both prefill and decode** worker containers:

```yaml
env:
  # --- UCX (RDMA transport) ---
  - name: UCX_TLS
    value: "cuda_ipc,cuda_copy,rc"
  - name: UCX_IB_GID_INDEX
    value: "3"                    # RoCEv2 GID index — GCP-specific
  - name: UCX_RC_TIMEOUT
    value: "600s"
  - name: UCX_KEEPALIVE_INTERVAL
    value: "300s"

  # --- NCCL ---
  - name: NCCL_IB_DISABLE
    value: "0"
  - name: NCCL_CUMEM_ENABLE
    value: "1"                    # required for GB200
  - name: NCCL_NVLS_ENABLE
    value: "1"                    # required for NVL72
  - name: NVIDIA_GDRCOPY
    value: "1"
  - name: NCCL_STORE_TIMEOUT
    value: "7200"
  - name: NCCL_DEBUG
    value: INFO

  # --- NIXL ---
  - name: NIXL_LOG_LEVEL
    value: INFO                   # use DEBUG during bringup
  - name: NIXL_TELEMETRY_ENABLE
    value: "y"
  - name: NIXL_TELEMETRY_EXPORTER
    value: prometheus
  - name: NIXL_TELEMETRY_PROMETHEUS_PORT
    value: "19090"
```

### 4. Set common networking env vars (spec-level)

```yaml
spec:
  envs:
    - name: GLOO_SOCKET_IFNAME
      value: eth0
    - name: NCCL_SOCKET_IFNAME
      value: eth0
    - name: NATS_SERVER
      value: nats://dynamo-platform-nats.<namespace>.svc.cluster.local:4222
    - name: ETCD_ENDPOINTS
      value: dynamo-platform-etcd.<namespace>.svc.cluster.local:2379
```

### 5. (Optional) Configure MNNVL for wide EP/TP

If you are running **multi-node** decode (e.g. EP16 across 4 nodes), you need
MNNVL to pin pods to the same NVLink clique. See [MNNVL for NVL72](#mnnvl-for-nvl72-multi-node-nvlink).

### What's different from nscale

- Pod annotations required for GKE multi-network RDMA.
- `UCX_IB_GID_INDEX=3` required (RoCE GID selection).
- `NCCL_CUMEM_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, `NVIDIA_GDRCOPY=1` required
  for GB200 NVL72.
- MNNVL configuration needed for wide EP/TP across nodes.

---

## Deploying on AWS (EFA)

Two validated paths, depending on runtime:

| Property         | TRT-LLM on GB200                                                    | vLLM on H100                                                  |
| ---------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| Architecture     | arm64 (Grace Blackwell)                                             | x86_64                                                        |
| GPU              | NVIDIA GB200                                                        | NVIDIA H100                                                   |
| Instance type    | p6e-gb200.36xlarge                                                  | p5.48xlarge / p5en.48xlarge                                   |
| Interconnect     | EFA (Elastic Fabric Adapter)                                        | EFA (Elastic Fabric Adapter)                                  |
| NIXL backend     | **UCX** (SRD transport over EFA, auto-discovered)                   | **LIBFABRIC** (explicit via `kv_connector_extra_config`)      |
| RDMA resource    | `vpc.amazonaws.com/efa`                                             | `vpc.amazonaws.com/efa`                                       |
| Security context | `IPC_LOCK` capability (no privileged)                               | `IPC_LOCK` capability (no privileged)                         |
| Validated image  | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1`                | `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.1.0rc6-efa-amd64` |
| Node selector    | `kubernetes.io/arch: arm64`, `nvidia.com/gpu.product: NVIDIA-GB200` | x86_64 EFA-enabled p5/p5en nodes                              |
| MNNVL            | ComputeDomain + ResourceClaims (same as GCP)                        | N/A (single-node TP, no NVL72)                                |

> **Backend selection:**
>
> - **TRT-LLM** (GB200): NIXL auto-selects **UCX** with **SRD (Scalable
>   Reliable Datagram)** transport over EFA. UCX's `ib` transport discovers
>   `rdmap*` devices and uses SRD automatically — no `FI_PROVIDER` needed.
>   The `SGLANG_DISAGGREGATION_NIXL_BACKEND` variable is *only* for SGLang.
> - **vLLM** (H100): pass `"backends":["LIBFABRIC"]` in
>   `kv_connector_extra_config` to use libfabric directly (with
>   `FI_PROVIDER=efa`). UCX is not used in this path.

> **No EFA driver mounts in the pod.** The EFA Kubernetes device plugin
> (`aws-efa-k8s-device-plugin` DaemonSet) injects all required device
> nodes (`/dev/infiniband/uverbs*`, `/dev/infiniband/rdma_cm`) and the
> `/sys/class/infiniband/*` paths into the container based on the
> `vpc.amazonaws.com/efa` resource request. Do **not** add manual
> `hostPath` mounts for `/opt/amazon/efa`, `/dev/infiniband`, or any
> kernel module — the EFA SDK libraries are already baked into the
> validated runtime images and the device plugin handles the rest.

### 1. Request RDMA resources

EFA devices are exposed by the **EFA device plugin** via
`vpc.amazonaws.com/efa`. **One unit corresponds to one EFA NIC**, not one
GPU. Request the count that matches the instance topology:

| Instance       | EFA NICs | GPUs | Request `vpc.amazonaws.com/efa` |
| -------------- | -------- | ---- | ------------------------------- |
| p5.48xlarge    | 32       | 8    | `"32"` (full node)              |
| p5en.48xlarge  | 16       | 8    | `"16"` (full node)              |
| p6e-gb200.36xl | 4        | 4    | `"4"` (full node)               |

```yaml
resources:
  limits:
    gpu: "8"
    custom:
      vpc.amazonaws.com/efa: "32"   # all NICs on a p5.48xlarge
  requests:
    custom:
      ephemeral-storage: "2Gi"
      vpc.amazonaws.com/efa: "32"
```

> Requesting fewer than the full NIC count caps fabric bandwidth and
> leaves PCIe-affined NICs idle. To saturate EFA, request all NICs on
> the node and use a TP size that covers all GPUs.

> **GPU↔NIC affinity caveat:** the EFA device plugin and the NVIDIA GPU
> device plugin allocate independently. Requesting a *partial* set
> (e.g. `gpu: "4"` + `vpc.amazonaws.com/efa: "16"` on a p5) does **not**
> guarantee that the allocated NICs sit on the same PCIe rail as the
> allocated GPUs — the kubelet may hand you GPUs on rail 1 and NICs on
> rail 2, forcing cross-rail traffic. To get topology-aware allocation,
> all of the following must be true:
>
> 1. Kubelet `--topology-manager-policy=single-numa-node` (or `restricted`).
> 2. NVIDIA GPU plugin and `aws-efa-k8s-device-plugin` both report NUMA
>    hints (set `--numa-aware=true` on the EFA plugin DaemonSet,
>    requires ≥ v0.4).
> 3. Pod is `Guaranteed` QoS (CPU/memory request == limit).
>
> Without TopologyManager, **request the whole node** (`gpu: "8"` +
> `vpc.amazonaws.com/efa: "32"` on p5.48xlarge) — that's the only way
> to be sure NICs and GPUs are on matching rails.

No pod annotations and no driver volume mounts are needed. The EFA device
plugin injects `/dev/infiniband/uverbs*` (one per requested NIC) and
exposes `/sys/class/infiniband/*` automatically.

### 2. Set the security context

EFA on Kubernetes does **not** require `privileged: true`. The EFA device
plugin gives the container the right device nodes; libfabric and ibverbs
only need the ability to pin memory for RDMA registrations:

```yaml
securityContext:
  capabilities:
    add:
      - IPC_LOCK
```

> **Validated:** `IPC_LOCK` alone is sufficient for both UCX/SRD (TRT-LLM)
> and LIBFABRIC (vLLM) paths on `aws-efa-k8s-device-plugin` ≥ v0.5. If
> ibverbs registration fails with `Cannot allocate memory`, raise the
> node-level `memlock` ulimit (default `64KB` is too low) — do **not**
> fall back to privileged.

### 3. Set environment variables

These go on **both prefill and decode** worker containers:

```yaml
env:
  # --- UCX (RDMA transport via SRD over EFA) ---
  - name: UCX_TLS
    value: "cuda_ipc,cuda_copy,rc"
  - name: UCX_RC_TIMEOUT
    value: "600s"
  - name: UCX_KEEPALIVE_INTERVAL
    value: "300s"

  # --- NCCL ---
  - name: NCCL_MNNVL_ENABLE
    value: "1"
  - name: NCCL_CUMEM_ENABLE
    value: "1"
  - name: NCCL_NVLS_ENABLE
    value: "1"
  - name: NVIDIA_GDRCOPY
    value: "1"
  - name: TLLM_LOG_LEVEL
    value: "info"
  - name: TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER
    value: "1"
  - name: TRTLLM_ENABLE_PDL
    value: "1"
  - name: NCCL_STORE_TIMEOUT
    value: "7200"
  - name: NCCL_DEBUG
    value: INFO

  # --- NIXL ---
  - name: NIXL_LOG_LEVEL
    value: INFO                   # use DEBUG during bringup
  - name: NIXL_TELEMETRY_ENABLE
    value: "y"
  - name: NIXL_TELEMETRY_EXPORTER
    value: prometheus
  - name: NIXL_TELEMETRY_PROMETHEUS_PORT
    value: "19090"
```

### 4. Fix NIXL_PLUGIN_DIR for decode multinode

> **Critical for disaggregated multi-node decode.** The container image sets
> `NIXL_PLUGIN_DIR` globally to the system NIXL plugin path
> (`/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/plugins`). This works for
> prefill workers (which load `nixl_cu13`), but decode multinode workers launch
> via `mpirun → mgmn_worker_node` which loads **TRT-LLM's bundled NIXL** —
> a different NIXL build that is ABI-incompatible with the system plugins.
> The result is a crash:
>
> ```
> RuntimeError: [TensorRT-LLM][ERROR] Assertion failed:
>   status == NIXL_SUCCESS (transferAgent.cpp:416)
> getPluginParams: backend 'UCX' not found
> ```

Override `NIXL_PLUGIN_DIR` **only on the decode service** to point to
TRT-LLM's own matching plugins:

```yaml
# In the DynamoGraphDeployment decode service definition:
extraPodSpec:
  mainContainer:
    env:
      - name: NIXL_PLUGIN_DIR
        value: "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/plugins"
```

Do **not** override `NIXL_PLUGIN_DIR` globally (in `spec.envs`) or on the
prefill service — prefill workers use `nixl_cu13` which is compatible with
the system plugins.

**How it works:** Two separate NIXL builds exist in the container:

| Worker type      | Process                      | NIXL library loaded  | Compatible plugins              |
| ---------------- | ---------------------------- | -------------------- | ------------------------------- |
| Prefill          | `mpi4py.futures.server`      | `nixl_cu13` (system) | System plugins (default path)   |
| Decode multinode | `mgmn_worker_node` (TRT-LLM) | TRT-LLM bundled NIXL | TRT-LLM plugins (override path) |

When `NIXL_PLUGIN_DIR` is not overridden, the decode worker tries to load
system plugins into TRT-LLM's bundled NIXL, hits an ABI mismatch, and fails
to find the `UCX` backend. The targeted override ensures each NIXL build
discovers its own ABI-compatible plugins.

### 5. Set common networking env vars (spec-level)

```yaml
spec:
  envs:
    - name: GLOO_SOCKET_IFNAME
      value: eth0
    - name: NCCL_SOCKET_IFNAME
      value: eth0
    - name: TRTLLM_ENABLE_PDL
      value: "1"
    - name: NATS_SERVER
      value: nats://dynamo-platform-nats.<namespace>.svc.cluster.local:4222
    - name: ETCD_ENDPOINTS
      value: dynamo-platform-etcd.<namespace>.svc.cluster.local:2379
```

### 6. TRT-LLM cache_transceiver_config

In the ConfigMaps for both prefill and decode, use `backend: DEFAULT` so
TRT-LLM lets NIXL auto-select the backend (UCX on AWS):

```yaml
cache_transceiver_config:
  max_tokens_in_buffer: 4608
  backend: DEFAULT
```

Using `backend: UCX` also works but bypasses NIXL's backend auto-selection
and was used as a temporary workaround before the `NIXL_PLUGIN_DIR` fix.

### 7. vLLM with LIBFABRIC backend (H100 / p5)

> **Sample DynamoGraphDeployment:**
> [`qwen3-32b-vllm-efa/aws-h100-disagg-efa.yaml`](https://gitlab-master.nvidia.com/saperiyasamy/dynamo-recipes/-/blob/main/qwen3-32b-vllm-efa/aws-h100-disagg-efa.yaml?ref_type=heads)
> in `dynamo-recipes`. Validated end-to-end on `p5.48xlarge`
> (H100, 32 EFA NICs) with the
> `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.1.0rc6-efa-amd64` image,
> `IPC_LOCK`-only security context, and full-node EFA allocation
> (`vpc.amazonaws.com/efa: "32"`). Sustained ~91 Gbps RDMA-read
> aggregate across all 32 NICs at TP=8 with 0 drops/retransmits.

When using **vLLM** (not TRT-LLM), select the LIBFABRIC backend explicitly
in the NIXL connector config — vLLM does not use UCX-over-SRD. Pass it
through `--kv-transfer-config` on both prefill and decode workers:

```yaml
args:
  - --disaggregation-mode
  - prefill        # or "decode" on the decode worker
  - --kv-transfer-config
  - '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
  - --tensor-parallel-size
  - "8"
```

Required env (replaces the UCX block in §3 — vLLM/LIBFABRIC ignores
`UCX_*` variables):

```yaml
env:
  # --- Libfabric / EFA provider ---
  - name: FI_PROVIDER
    value: efa
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
  - name: FI_EFA_ENABLE_SHM_TRANSFER
    value: "0"           # MUST be 0 — SHM breaks NIXL GPU buffer registrations
  - name: FI_LOG_LEVEL
    value: info          # use "debug" during bringup
  - name: LD_LIBRARY_PATH
    value: /opt/amazon/efa/lib:/opt/amazon/efa/lib64:/usr/local/lib
  - name: PATH
    value: /opt/dynamo/venv/bin:/opt/amazon/efa/bin:/usr/local/bin:/usr/bin:/bin

  # --- vLLM ---
  - name: VLLM_LOGGING_LEVEL
    value: INFO
```

`§4 (NIXL_PLUGIN_DIR override)` and `§6 (cache_transceiver_config)` are
**TRT-LLM-only** and do not apply to the vLLM path.

### What's different from nscale/GCP

- **TRT-LLM path:** NIXL uses the **UCX** backend, but UCX transports over
  **SRD** (EFA's high-performance datagram protocol) instead of RC verbs.
- **vLLM path:** NIXL uses the **LIBFABRIC** backend directly
  (`backends: ["LIBFABRIC"]` in `kv_connector_extra_config`); UCX is not
  involved.
- `NIXL_PLUGIN_DIR` must be overridden on the **decode service only**
  (TRT-LLM path) to point to TRT-LLM's bundled NIXL plugins (ABI mismatch
  with system plugins).
- No `UCX_IB_GID_INDEX` needed (SRD doesn't use RoCE GID selection;
  LIBFABRIC doesn't use UCX at all).
- `NCCL_CUMEM_ENABLE=1`, `NCCL_NVLS_ENABLE=1`, `NVIDIA_GDRCOPY=1` required
  for GB200 NVL72 (TRT-LLM path).
- **No privileged mode required** — `IPC_LOCK` capability is sufficient
  on `aws-efa-k8s-device-plugin` ≥ v0.5 for both UCX/SRD and LIBFABRIC
  paths.
- MNNVL uses **ComputeDomain + ResourceClaims** (same mechanism as GCP) to
  pin pods to the same NVLink clique (GB200 only).
- `NCCL_MNNVL_ENABLE=1` must be set on GB200 NVL72 (in addition to
  `NCCL_NVLS_ENABLE` and `NCCL_CUMEM_ENABLE`).
- `TRTLLM_ENABLE_PDL=1` enables pipelined disaggregated serving in
  TRT-LLM (TRT-LLM path only).

---

## MNNVL for NVL72 (Multi-Node NVLink)

GB200 NVL72 racks have an NVLink domain spanning up to 72 GPUs across multiple
nodes. When using **wide Expert Parallelism (EP)** or **wide Tensor Parallelism
(TP)** across nodes, NCCL all-to-all traffic should traverse NVLink — not
RoCE/EFA. This requires pinning all pods in a multi-node job to the **same
NVLink clique** (set of nodes in the same NVLink domain).

### When you need MNNVL

You need this configuration when your decode service uses `multinode.nodeCount > 1`
with wide TP/EP (e.g. `--tensor-parallel-size 16 --expert-parallel-size 16`).
Single-node deployments do **not** need MNNVL.

### GCP — ComputeDomain + nodeAffinity

GKE exposes MNNVL topology via the **NVIDIA DRA driver** (Dynamic Resource
Allocation). You need three pieces:

**Step 1: Create a ComputeDomain CR** that defines the NVLink clique size:

```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: <name>-compute-domain
  namespace: <namespace>
spec:
  channel:
    resourceClaimTemplate:
      name: <name>-compute-domain-channel
  numNodes: 5   # number of nodes in the NVLink domain
```

**Step 2: Add a ResourceClaim to the decode service** to bind pods to the
same clique:

```yaml
# In the DynamoGraphDeployment decode service:
resources:
  claims:
    - name: compute-domain-channel

# In extraPodSpec:
extraPodSpec:
  resourceClaims:
    - name: compute-domain-channel
      resourceClaimTemplateName: <name>-compute-domain-channel
```

**Step 3: Add nodeAffinity** to pin decode pods to the correct node pool
(= NVLink clique):

```yaml
extraPodSpec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-nodepool
                operator: In
                values:
                  - <nodepool-name>   # e.g. customer-gpu-w0e
```

All three pieces work together: the ComputeDomain ensures NCCL discovers the
NVLink topology, the ResourceClaim ties all pods in the multi-node job to the
same domain, and the nodeAffinity ensures they land on the right physical nodes.

**Additional NCCL var for wide EP:** Set `NCCL_GRAPH_MIXING_SUPPORT=0` on
decode workers when using multi-node EP.

### AWS — ComputeDomain + ResourceClaims

On AWS GB200 (p6e instances), MNNVL uses the **same ComputeDomain mechanism**
as GCP. The NVIDIA DRA driver is available on EKS GB200 clusters and handles
NVLink clique scheduling.

**Step 1: Create a ComputeDomain CR** sized to include all nodes in the
deployment (prefill + decode):

```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: <name>-compute-domain
spec:
  numNodes: 3   # e.g. 1 prefill node + 2 decode nodes
  channel:
    resourceClaimTemplate:
      name: <name>-compute-domain-channel
```

**Step 2: Add ResourceClaims to both prefill and decode services** to bind
all pods to the same NVLink clique:

```yaml
# In both prefill and decode service definitions:
resources:
  claims:
    - name: compute-domain-channel

extraPodSpec:
  resourceClaims:
    - name: compute-domain-channel
      resourceClaimTemplateName: <name>-compute-domain-channel
```

**Required NCCL env vars** (set in `spec.envs`):

```yaml
- name: NCCL_MNNVL_ENABLE
  value: "1"
- name: NCCL_NVLS_ENABLE
  value: "1"
- name: NCCL_CUMEM_ENABLE
  value: "1"
```

> **Note:** Unlike GCP, AWS does not require `nodeAffinity` for node pool
> selection — the DRA driver and ComputeDomain handle clique placement
> directly.

### nscale — Not Applicable

nscale B200 (x86_64) nodes use InfiniBand for inter-node communication.
There is no NVLink domain spanning nodes, so MNNVL configuration does not
apply. Multi-GPU TP within a single node uses NVLink/NVSwitch natively.

---

## Environment Variable Reference

Detailed reference for all variables used across platforms.

### NIXL Variables

Set on **all platforms**, on both prefill and decode workers.

| Variable                             | Value            | Description                                                                                                                               |
| ------------------------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `NIXL_LOG_LEVEL`                     | `DEBUG` / `INFO` | NIXL logging verbosity. Use `DEBUG` during bringup.                                                                                       |
| `NIXL_TELEMETRY_ENABLE`              | `y`              | Enable Prometheus metrics export from NIXL.                                                                                               |
| `NIXL_TELEMETRY_EXPORTER`            | `prometheus`     | Telemetry export format.                                                                                                                  |
| `NIXL_TELEMETRY_PROMETHEUS_PORT`     | `19090`          | Port for the NIXL metrics endpoint.                                                                                                       |
| `SGLANG_DISAGGREGATION_NIXL_BACKEND` | `LIBFABRIC`      | **SGLang on AWS only.** Switches to libfabric. Not needed for TRT-LLM (uses UCX over SRD).                                                |
| `NIXL_PLUGIN_DIR`                    | *(see below)*    | **AWS decode only.** Override to TRT-LLM's plugin path for decode multinode. See [§4 above](#4-fix-nixl_plugin_dir-for-decode-multinode). |

### UCX Variables (nscale, GCP, AWS)

Used where NIXL uses the UCX backend (all platforms with TRT-LLM).

| Variable                 | Value                   | Description                                                                                        |
| ------------------------ | ----------------------- | -------------------------------------------------------------------------------------------------- |
| `UCX_TLS`                | `cuda_ipc,cuda_copy,rc` | Transport layers: `cuda_ipc` for same-node GPU-GPU, `cuda_copy` for host-device, `rc` for RDMA RC. |
| `UCX_IB_GID_INDEX`       | `3`                     | **GCP only.** RoCEv2 GID entry. Not needed on nscale IB or AWS EFA.                                |
| `UCX_RC_TIMEOUT`         | `600s`                  | Timeout for RC transport before failure.                                                           |
| `UCX_KEEPALIVE_INTERVAL` | `300s`                  | Keepalive probe interval for idle connections.                                                     |
| `UCX_LOG_LEVEL`          | `debug` / `info`        | UCX logging. Use `debug` during bringup.                                                           |

### Libfabric Variables (SGLang on AWS)

Used **only on AWS with SGLang** when using the LIBFABRIC NIXL backend. **Not
needed for TRT-LLM** deployments (which use UCX over SRD).

| Variable                     | Value                     | Description                                               |
| ---------------------------- | ------------------------- | --------------------------------------------------------- |
| `FI_PROVIDER`                | `efa`                     | Selects the EFA libfabric provider.                       |
| `FI_EFA_USE_DEVICE_RDMA`     | `1`                       | GPUDirect RDMA (requires `efa_nv_peermem` kernel module). |
| `FI_EFA_ENABLE_SHM_TRANSFER` | `0`                       | **Must be 0.** SHM breaks NIXL GPU buffer registrations.  |
| `FI_LOG_LEVEL`               | `info` / `debug`          | Libfabric logging verbosity.                              |
| `LD_LIBRARY_PATH`            | `/opt/amazon/efa/lib:...` | Must include EFA SDK lib path.                            |

### NCCL Variables

Set on **all platforms**, on both prefill and decode workers.

| Variable                    | Value  | Platforms   | Description                                                   |
| --------------------------- | ------ | ----------- | ------------------------------------------------------------- |
| `NCCL_DEBUG`                | `INFO` | all         | NCCL debug logging.                                           |
| `NCCL_IB_DISABLE`           | `0`    | nscale, GCP | Ensure IB/RoCE is enabled for NCCL.                           |
| `NCCL_CUMEM_ENABLE`         | `1`    | GCP, AWS    | cuMem allocator for NCCL buffers. Required on GB200.          |
| `NCCL_NVLS_ENABLE`          | `1`    | GCP, AWS    | NVLink SHARP for multi-node NVLink. Required on NVL72.        |
| `NVIDIA_GDRCOPY`            | `1`    | GCP, AWS    | GDRCopy for low-latency GPU-host copies.                      |
| `NCCL_STORE_TIMEOUT`        | `7200` | all         | Bootstrap store timeout (seconds). Set high for large models. |
| `NCCL_GRAPH_MIXING_SUPPORT` | `0`    | GCP, AWS    | Disable graph mixing. Only needed for wide EP multi-node.     |

---

## Troubleshooting

### NIXL transfer hangs / `waiting_timeout`

- Verify `IPC_LOCK` capability is set on the worker pods (all clusters).
- Check that `ulimit -l unlimited` is in the container entrypoint.
- Confirm RDMA devices are visible: `ls /dev/infiniband/`.
- On AWS, confirm `FI_EFA_ENABLE_SHM_TRANSFER=0`.

### UCX connection failures on GCP

- Verify `UCX_IB_GID_INDEX=3`. Wrong GID index causes connection failures.
- Confirm GKE multi-network annotations are present and `rdma0` interface
  exists: `ip addr show rdma0`.

### NCCL timeout during model loading

- Increase `NCCL_STORE_TIMEOUT` (default is too low for large models).
- Verify `GLOO_SOCKET_IFNAME=eth0` and `NCCL_SOCKET_IFNAME=eth0`.

### NIXL plugin ABI mismatch on AWS (decode crash with `backend 'UCX' not found`)

**Symptoms:** Decode leader pod crashes during startup with:
```
RuntimeError: [TensorRT-LLM][ERROR] Assertion failed:
  status == NIXL_SUCCESS (transferAgent.cpp:416)
getPluginParams: backend 'UCX' not found
```

**Cause:** The container image has two NIXL builds:
1. **System NIXL** (`nixl_cu13`) — loaded by prefill workers (`mpi4py.futures.server`).
2. **TRT-LLM bundled NIXL** — loaded by decode multinode workers (`mgmn_worker_node`).

The container's default `NIXL_PLUGIN_DIR` points to system plugins
(`/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/plugins`), which are
ABI-compatible with `nixl_cu13` but **not** with TRT-LLM's bundled NIXL.
When the decode worker loads these mismatched plugins, the UCX backend fails
to initialize.

**Fix:** Override `NIXL_PLUGIN_DIR` on the decode service only:
```yaml
extraPodSpec:
  mainContainer:
    env:
      - name: NIXL_PLUGIN_DIR
        value: "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/libs/nixl/plugins"
```

**Verification:** After the fix, decode logs should show:
```
[NIXL Infra] Loaded backend plugin: UCX
[NIXL XFER] UCX backend init...
```
And prefill logs should independently show the same via system plugins.

**Debugging tips:**
- Check which NIXL library each worker loads:
  `cat /proc/<pid>/maps | grep nixl`
- List available plugins: `ls $NIXL_PLUGIN_DIR`
- Set `NIXL_LOG_LEVEL=DEBUG` to see plugin discovery details.

### EFA not detected on AWS

- Check that the EFA device plugin DaemonSet is running:
  `kubectl get ds -n kube-system | grep efa`.
- Verify `vpc.amazonaws.com/efa` is in node allocatable:
  `kubectl describe node <node> | grep efa`.

### Verifying NIXL is using SRD over EFA (AWS)

After deployment, confirm NIXL is using EFA with SRD transport:
```bash
# In decode/prefill worker logs, look for:
[NIXL XFER] UCX backend init...
# And in UCX debug output:
ucp_worker.c:*  UCX transport: srd
```

You can also verify from inside a pod:
```bash
# List EFA devices
ls /dev/infiniband/
# Should show: rdmap0s21, rdmap160s21, rdmap32s21, rdmap64s21 (4 devices for 4 GPUs)
```

### MNNVL not working on GCP

- Verify the `ComputeDomain` CR exists and `numNodes` matches your topology.
- Confirm `resourceClaimTemplateName` matches the ComputeDomain's channel name.
- Check that decode pods landed on nodes in the same NVLink clique (same
  node pool).
