# GCP02 GB200 NCCL Runbook

This records the NCCL and Kubernetes settings that made the SWE-bench Pro
trainer plus online Dynamo/SGLang weight sync work on
`dynamo-gcp-dev-02` in `warnold-dynamo`.

There are two NCCL scopes:

1. Megatron trainer collectives across the trainer Ray workers.
2. Slime online weight update, where a trainer rank broadcasts weights to the
   Dynamo/SGLang engine ranks.

Do not collapse these into one setting set without retesting. The weight-update
group includes engine processes, and its topology constraints can differ from
the trainer-only group.

## Kubernetes requirements

For each 4-GPU GB200 trainer pod:

```yaml
metadata:
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
spec:
  resourceClaims:
    - name: compute-domain-channel
      resourceClaimTemplateName: warnold-swepro-compute-domain-channel
  nodeSelector:
    cloud.google.com/gke-nodepool: customer-gpu-w0e
    nvidia.com/gpu.product: NVIDIA-GB200
  containers:
    - securityContext:
        capabilities:
          add:
            - IPC_LOCK
      resources:
        requests:
          nvidia.com/gpu: "4"
          networking.gke.io.networks/rdma-0: "1"
          networking.gke.io.networks/rdma-1: "1"
          networking.gke.io.networks/rdma-2: "1"
          networking.gke.io.networks/rdma-3: "1"
        limits:
          nvidia.com/gpu: "4"
          networking.gke.io.networks/rdma-0: "1"
          networking.gke.io.networks/rdma-1: "1"
          networking.gke.io.networks/rdma-2: "1"
          networking.gke.io.networks/rdma-3: "1"
      volumeMounts:
        - name: shm
          mountPath: /dev/shm
  volumes:
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 64Gi
```

The entrypoint must raise the lock and file-descriptor limits before starting
Ray or Dynamo:

```bash
ulimit -l unlimited
ulimit -n 1048576
```

The 64 GiB `/dev/shm` mount is required. The small default container shm caused
NCCL bootstrap failures with messages like `No space left on device` for
`/dev/shm/nccl-*`.

## Trainer runtime env

Known-good for trainer replay and live weight update:

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
```

For Slime's online weight-update group:

```bash
export SLIME_WEIGHT_UPDATE_FLATTENED_BUCKET=1
export SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE=0

# Enable only while debugging or collecting durable logs.
export SLIME_WEIGHT_UPDATE_NCCL_DEBUG=INFO
export SLIME_WEIGHT_UPDATE_NCCL_DEBUG_SUBSYS=INIT,ENV,NET
```

The `SLIME_WEIGHT_UPDATE_*` variables are copied into the temporary NCCL
environment used to create the trainer-to-engine weight-update process group.
Keep them explicit so training collectives and weight sync can be tuned
independently.

## Engine runtime env

Known-good engine entrypoint env:

```bash
export UCX_TLS=cuda_ipc,cuda_copy,rc
export UCX_IB_GID_INDEX=3
export UCX_RC_TIMEOUT=600s
export UCX_KEEPALIVE_INTERVAL=300s

export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_HOST_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_MNNVL_ENABLE=0
export NCCL_STORE_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NVIDIA_GDRCOPY=1

export NIXL_LOG_LEVEL=INFO
export NIXL_TELEMETRY_ENABLE=y
export NIXL_TELEMETRY_EXPORTER=prometheus
export NIXL_TELEMETRY_PROMETHEUS_PORT=19090
```

`NCCL_MNNVL_ENABLE=0` is intentional for the current working topology. The
trainer pods are in a ComputeDomain, but the engine pods are not currently in
the same ComputeDomain and do not have the same IMEX channel device. For the
online weight-update group, enabling MNNVL in that mixed topology made NCCL try
an unavailable path. The validated path uses RoCE/GDRDMA; NCCL logs show
`MNNVL 0` and `NET/IBext.../GDRDMA`.

If engines are later moved into the same ComputeDomain/IMEX domain as the
trainer ranks, retest `NCCL_MNNVL_ENABLE=1` and
`SLIME_WEIGHT_UPDATE_NCCL_MNNVL_ENABLE=1` with
`examples/swebench-pro/weight_transfer_smoke.py` before using it in an RL run.

## Variables to leave unset

These were not part of the known-good setup:

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

In particular, do not set `NCCL_NET=IB`. It can force a transport choice that
fights NCCL's topology selection. Also avoid `MC_FORCE_MNNVL=1` unless the
entire process group is proven to be in the same MNNVL-capable domain.

## Validation

The working validation sequence was:

1. Recreate trainer pods with the 64 GiB `/dev/shm` mount.
2. Run clipped real rollout debug data with live weight updates enabled.
3. Confirm the run logs include a full online update, actor train, and a second
   online update.

Expected log shape:

```text
Timer update_weights start
[WEIGHT UPDATE] ... total_sync=...
[WEIGHT UPDATE OUTER] ... num_engines=1 ...
Timer update_weights end
Timer train start
Timer actor_train start
perf 0: ...
Timer actor_train end
Timer train end
Timer update_weights start
[WEIGHT UPDATE] ... total_sync=...
Timer update_weights end
```

For an isolated weight-sync smoke from the trainer head pod:

```bash
python3 examples/swebench-pro/weight_transfer_smoke.py \
  --engine-url http://ENGINE_POD_IP:30001 \
  --preset small \
  --load-format flattened_bucket \
  --world-size 3 \
  --weight-version-prefix post-train-smoke \
  --http-timeout 240 \
  --pg-timeout 240
```

This smoke intentionally mutates the live engine weights. Recreate the engine or
perform a full trainer update before using it for rollouts.
